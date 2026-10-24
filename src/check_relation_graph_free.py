"""Check that relation-graph-free RelNet (`relation_model.relation_graph_free: yes`) reproduces
TRIX exactly, with the released checkpoints, and compare sizes and timings.

    python ./src/check_relation_graph_free.py --task entity --ckpt ./entity_prediction.pth \
        --dataset FB15k237Inductive --version v1 --root ./kg-datasets --device cuda:0
    python ./src/check_relation_graph_free.py --task relation --ckpt ./relation_prediction.pth \
        --dataset WN18RRInductive --version v1 --root ./kg-datasets

For one split of one dataset: size of the materialized relation graph against the entity x relation
incidences it is replaced by, time to build it, maximum score difference between the two
implementations on a few batches of test triples (all negatives), time per forward pass of each,
and maximum gradient difference after one training step on the same batch.
"""

import argparse
import copy
import os
import sys
import time

import torch
from easydict import EasyDict
from torch.nn import functional as F
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trix import datasets, tasks
from trix.tasks import build_relation_graph
from trix.relation_graph_free import FLAVOURS, incidence

NBF = dict(message_func="distmult", aggregate_func="sum", short_cut=True, layer_norm=True)
# config/pretrain_entity.yaml and config/pretrain_relation.yaml (the released checkpoints)
ENTITY_CFG = dict(relation_model=dict(input_dim=32, hidden_dims=[32, 32, 32], **NBF),
                  entity_model_1=dict(input_dim=32, hidden_dims=[32, 32], **NBF),
                  entity_model_2=dict(input_dim=32, hidden_dims=[32, 32, 32, 32], **NBF))
RELATION_CFG = dict(trix=dict(feature_dim=32, num_layer=3, num_mlp_layer=2),
                    relation_model=dict(input_dim=32, hidden_dims=[32, 32], **NBF),
                    entity_model=dict(input_dim=32, hidden_dims=[32, 32], **NBF))


def build_model(task, relation_graph_free):
    if task == "entity":
        from trix.models_entity import TRIX
        rel_cfg = dict(ENTITY_CFG["relation_model"], relation_graph_free=relation_graph_free)
        return TRIX(rel_model_cfg=rel_cfg, entity_model_1_cfg=dict(ENTITY_CFG["entity_model_1"]),
                    entity_model_2_cfg=dict(ENTITY_CFG["entity_model_2"]))
    from trix.models_relation import TRIX
    rel_cfg = dict(RELATION_CFG["relation_model"], relation_graph_free=relation_graph_free)
    return TRIX(rel_model_cfg=rel_cfg, entity_model_cfg=dict(RELATION_CFG["entity_model"]),
                trix_cfg=EasyDict(RELATION_CFG["trix"]))


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["entity", "relation"], default="entity")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True, help="dataset class name, e.g. FB15k237Inductive")
    parser.add_argument("--version", default=None)
    parser.add_argument("--root", default="./kg-datasets")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bs", type=int, default=8, help="positive triples per batch")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--num_negative", type=int, default=32, help="negatives for the gradient check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    kwargs = dict(root=args.root)
    if args.version is not None:
        kwargs["version"] = args.version
    dataset = getattr(datasets, args.dataset)(**kwargs)
    data = dataset[["train", "valid", "test"].index(args.split)]
    name = args.dataset + (":" + args.version if args.version else "") + "/" + args.split

    # cost of the materialized relation graph: build it again on a bare copy of the graph
    bare = Data(edge_index=data.edge_index.cpu(), edge_type=data.edge_type.cpu(),
                num_nodes=data.num_nodes, num_relations=data.num_relations)
    _, t_build = timed(lambda: build_relation_graph(bare), torch.device("cpu"))
    if "relation_adj" not in data:
        data.relation_adj = bare.relation_adj
    data = data.to(device)
    num_rel_edges = {f: int(data.relation_adj[f].edge_index.size(1)) for f in FLAVOURS}
    inc = incidence(data)
    num_inc = int(inc["H"][0]._nnz() + inc["T"][0]._nnz())
    print(f"{name}: |V| = {int(data.num_nodes)}, |R| = {int(data.num_relations)}, |E| = {int(data.num_edges)}")
    print(f"  relation graph: {sum(num_rel_edges.values())} edges "
          f"({', '.join(f'{f} {n}' for f, n in num_rel_edges.items())}), built in {t_build:.1f} s")
    print(f"  entity x relation incidences: {num_inc} non-zeros "
          f"(relation graph / incidences = {sum(num_rel_edges.values()) / num_inc:.1f})")

    state = torch.load(args.ckpt, map_location="cpu")
    state = state.get("model", state)
    ref, free = build_model(args.task, False), build_model(args.task, True)
    ref.load_state_dict(state)
    free.load_state_dict(state)
    ref, free = ref.to(device).eval(), free.to(device).eval()

    triples = torch.cat([data.target_edge_index, data.target_edge_type.unsqueeze(0)]).t()
    worst, rel_worst, t_ref, t_free, passes = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        warm_up = tasks.all_negative(data, triples[:1])[0] if args.task == "entity" else tasks.all_negative_relation(data, triples[:1])
        ref(data, warm_up), free(data, warm_up)       # loads the rspmm extension, builds the incidences
        for b in range(args.batches):
            batch = triples[b * args.bs:(b + 1) * args.bs]
            if args.task == "entity":
                negatives = tasks.all_negative(data, batch)
            else:
                negatives = (tasks.all_negative_relation(data, batch),)
            for neg in negatives:
                s_ref, dt = timed(lambda: ref(data, neg), device)
                t_ref += dt
                s_free, dt = timed(lambda: free(data, neg), device)
                t_free += dt
                passes += 1
                diff = (s_ref - s_free).abs().max().item()
                worst = max(worst, diff)
                rel_worst = max(rel_worst, diff / s_ref.abs().max().item())
    print(f"  scores ({passes} forward passes, {args.bs} positives each, all negatives): "
          f"max abs diff {worst:.2e}, max rel diff {rel_worst:.2e}")
    print(f"  forward time per pass: materialized {t_ref / passes:.3f} s, "
          f"relation-graph-free {t_free / passes:.3f} s (x{t_ref / max(t_free, 1e-9):.1f})")

    # one training step on the same batch
    torch.manual_seed(args.seed)
    batch = triples[:args.bs]
    if args.task == "entity":
        batch = tasks.negative_sampling(data, batch, args.num_negative, strict=True)
    else:
        batch = tasks.negative_sampling_relation(data, batch, args.num_negative, strict=True)
    grads, losses, times = {}, {}, {}
    for label, model in (("materialized", ref), ("relation-graph-free", free)):
        model.train()
        model.zero_grad()

        def step():
            pred = model(data, batch)
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            loss = F.binary_cross_entropy_with_logits(pred, target)
            loss.backward()
            return loss.item()
        losses[label], times[label] = timed(step, device)
        grads[label] = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        model.eval()
    g_ref, g_free = grads["materialized"], grads["relation-graph-free"]
    assert set(g_ref) == set(g_free), "the two models do not have the same parameters with gradients"
    # relative gradient difference per parameter, in norm; worst parameters listed with their gradient scale
    rel_diff = {n: ((g_ref[n] - g_free[n]).norm() / g_ref[n].norm().clamp(min=1e-12)).item() for n in g_ref}
    worst = sorted(rel_diff, key=rel_diff.get, reverse=True)[:3]
    print(f"  training step ({args.num_negative} negatives): loss {losses['materialized']:.6f} vs "
          f"{losses['relation-graph-free']:.6f}; max rel grad diff (norm) {rel_diff[worst[0]]:.2e}")
    for n in worst:
        print(f"    {n}: rel diff {rel_diff[n]:.2e}, |grad| max {g_ref[n].abs().max().item():.2e}, "
              f"|diff| max {(g_ref[n] - g_free[n]).abs().max().item():.2e}")
    print(f"  forward + backward time: materialized {times['materialized']:.3f} s, "
          f"relation-graph-free {times['relation-graph-free']:.3f} s "
          f"(x{times['materialized'] / max(times['relation-graph-free'], 1e-9):.1f})")


if __name__ == "__main__":
    main()
