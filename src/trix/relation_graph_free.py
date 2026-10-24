"""Relation-graph-free message passing for RelNet.

TRIX's relation module (`RelNet`) runs an NBFNet over the *relation graph* built by
`tasks.build_relation_graph`: one node per relation type and one edge (r_1, r_2, v) per entity v
shared by relations r_1 and r_2, in four flavours (hh / ht / th / tt) according to whether v is
the head or the tail of r_1 and of r_2. That graph has sum_v k_v^2 edges (at most 4 |V| alpha^2),
typically 4 to 30 times the number of triples of the original graph, and it is built in Python
when the dataset is processed.

With the DistMult message and sum aggregation used by TRIX, a `GeneralizedRelationalConv` over
one flavour computes, with p_v = relation_projection(e_v) the projected state of entity v,

    m[r_1] = sum_{(r_1, r_2, v)} x[r_2] * p_v + boundary[r_1]

The message is bilinear in x and p and the aggregation is a sum, so the sum over edges factors
through the shared entities. Let S(v) be the relations v sends from and D(v) the relations v
delivers to (heads(v) or tails(v) according to the flavour, see FLAVOURS):

    y_v    = sum_{r in S(v)} x[r]                      (relations -> entities)
    m[r_1] = sum_{v : r_1 in D(v)} p_v * y_v           (entities -> relations)
           - x[r_1] * sum_{v : r_1 in D(v)} p_v        (hh and tt have no edge r -> r)

i.e. two sparse products by the binary entity x relation incidence matrices (at most 2 |E|
non-zeros) instead of one pass over sum_v k_v^2 relation-graph edges. The relation graph is never
built, the parameters are the same, and the result is identical up to floating-point rounding
(`check_relation_graph_free.py` verifies scores and gradients against the released checkpoints;
the hh / tt flavours are accumulated in float64 so that the subtraction above does not lose
precision).
"""

import torch

# flavour -> (source incidence, destination incidence). In `tasks.build_relation_graph`, an edge
# (r_1, r_2, v) of flavour hh has r_1, r_2 in heads(v); ht has r_1 in heads(v), r_2 in tails(v);
# th has r_1 in tails(v), r_2 in heads(v); tt has r_1, r_2 in tails(v). The rspmm kernel aggregates
# towards edge_index[0] = r_1, so messages flow from r_2 (source) to r_1 (destination).
FLAVOURS = {"hh": ("H", "H"), "ht": ("T", "H"), "th": ("H", "T"), "tt": ("T", "T")}

_CACHE = "_relation_incidence"


def incidence(data):
    """Binary entity x relation incidence matrices of `data`, for entities as heads ("H") and as
    tails ("T"): {"H": (A_H, A_H^T), "T": (A_T, A_T^T)} with A of shape [num_nodes, num_relations]
    (sparse COO, coalesced). Built from the same edge_index / edge_type as
    `tasks.build_relation_graph` (presence of a (entity, relation) pair, not its multiplicity),
    and cached on `data`."""
    key = (int(data.num_edges), str(data.edge_index.device), data.edge_index.data_ptr())
    cached = getattr(data, _CACHE, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    num_nodes, num_relations = int(data.num_nodes), int(data.num_relations)
    out = {}
    for side, entity in (("H", data.edge_index[0]), ("T", data.edge_index[1])):
        index = torch.stack([entity, data.edge_type]).unique(dim=1)
        values = torch.ones(index.size(1), device=index.device, dtype=torch.get_default_dtype())
        A = torch.sparse_coo_tensor(index, values, (num_nodes, num_relations), check_invariants=False).coalesce()
        out[side] = (A, A.t().coalesce())
    setattr(data, _CACHE, (key, out))
    return out


def conv(layer, input, boundary, relation, inc, flavour):
    """One `GeneralizedRelationalConv` of a RelNet over the relation graph of one flavour, computed
    without the relation graph. `input`, `boundary`: [batch_size, num_relations, dim]; `relation`:
    entity states [batch_size, num_nodes, dim], already projected by `layer.relation_projection`."""
    if layer.message_func != "distmult" or layer.aggregate_func != "sum":
        raise ValueError("relation-graph-free message passing requires message_func=distmult and "
                         "aggregate_func=sum (got %s / %s)" % (layer.message_func, layer.aggregate_func))
    src, dst = FLAVOURS[flavour]
    A_src = inc[src][0]                                               # [num_nodes, num_relations]
    AT_dst = inc[dst][1]                                              # [num_relations, num_nodes]
    batch_size, num_relations, dim = input.shape
    x = input.transpose(0, 1).reshape(num_relations, batch_size * dim)        # [R, B * d]
    p = relation.transpose(0, 1).reshape(relation.shape[1], batch_size * dim)  # [V, B * d]
    if src == dst:
        # hh / tt: the relation graph has no edge r -> r, so the term r_2 = r_1 is removed by subtracting
        # x[r_1] * sum_v p_v from sum_v p_v * y_v. Both sums are large and nearly cancel on relations whose
        # entities mostly carry that one relation; in float32 this loses up to 40% of some gradients
        # (checked on FB15k237 v1 against a float64 reference), so these two flavours run in float64.
        x64, p64 = x.double(), p.double()
        A_src, AT_dst = A_src.double(), AT_dst.double()
        y = torch.sparse.mm(A_src, x64)                               # y_v = sum_{r in S(v)} x[r]
        m = torch.sparse.mm(AT_dst, p64 * y) - x64 * torch.sparse.mm(AT_dst, p64)
        m = m.to(x.dtype)
    else:
        y = torch.sparse.mm(A_src, x)                                 # y_v = sum_{r in S(v)} x[r]
        m = torch.sparse.mm(AT_dst, p * y)                            # m[r] = sum_{v : r in D(v)} p_v * y_v
    m = m.reshape(num_relations, batch_size, dim).transpose(0, 1) + boundary
    return layer.update(m, input)


def relnet_layer(relnet, i, layer_input, boundary, node_representations, inc):
    """Layer i of a RelNet: sum over the four flavours, same as the materialized version
    (`layers_hh[i] + layers_ht[i] + layers_th[i] + layers_tt[i]`, each with its own projection
    of the entity states)."""
    hidden = 0
    for flavour in FLAVOURS:
        layer = getattr(relnet, "layers_" + flavour)[i]
        relation = layer.relation_projection(node_representations)
        hidden = hidden + conv(layer, layer_input, boundary, relation, inc, flavour)
    return hidden
