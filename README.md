# TRIX

## Overview
![TRIX](figure/trix.png)

This is the code for [TRIX: A More Expressive Model for Zero-shot Domain Transfer in Knowledge Graphs](https://openreview.net/forum?id=mRB0XkewKW). TRIX (Transferable Relation-Entity Interactions in crossing patterns (X-patterns)) is a fully inductive model for entity prediction tasks and relation prediction tasks over knowledge graphs. It is strictly more expressive than previous fully inductive methods and achieves state-of-the-art prediction performance over 57 knowledge graph datasets.

## Installation
```bash
pip install -r requirements.txt
```

## Pretraining
The pretraining of TRIX is done on the FB15k237, WN18RR, CoDExMedium datasets. The entity prediction model and the relation prediction model are trained separately.

Pre-training on 3 graphs for the entity prediction model:

```bash
python ./src/pretrain_entity.py -c ./config/pretrain_entity.yaml --gpus [0]
```

Pre-training on 3 graphs for the relation prediction model:

```bash
python ./src/pretrain_relation.py -c ./config/pretrain_relation.yaml --gpus [0]
```
## Inference
The run_entity.py and run_relation.py are the scripts for the entity prediction and the relation prediction respectively.

* `--dataset`: the dataset name.
* `--version`: a version of the dataset if it has multiple versions.
* `--epochs`: number of epochs to train. If it is 0, it's zero-shot inference; if it is larger than 0, it's inference with fine-tuning.
* `--bpe`: batches per epoch.

### Zero-shot Inference
Zero-shot Inference for the entity prediction task:
```bash
python ./src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset CoDExSmall --epochs 0 --bpe null --gpus [0] --ckpt /path/to/checkpoint

python ./src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FB15k237Inductive --version v1 --epochs 0 --bpe null --gpus [0] --ckpt /path/to/checkpoint
```

Zero-shot Inference for the relation prediction task:
```bash
python ./src/run_relation.py -c ./config/run_relation_transductive.yaml --dataset CoDExSmall --epochs 0 --bpe null --gpus [0] --ckpt /path/to/checkpoint

python ./src/run_relation.py -c ./config/run_relation_inductive.yaml --dataset FB15k237Inductive --version v1 --epochs 0 --bpe null --gpus [0] --ckpt /path/to/checkpoint
```

### Inference with Fine-tuning
Inference with fine-tuning for the entity prediction task:
```bash
python ./src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset CoDExSmall --epochs 3 --bpe 1000 --gpus [0] --ckpt /path/to/checkpoint

python ./src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FB15k237Inductive --version v1 --epochs 3 --bpe 1000 --gpus [0] --ckpt /path/to/checkpoint
```

Inference with fine-tuning for the relation prediction task:
```bash
python ./src/run_relation.py -c ./config/run_relation_transductive.yaml --dataset CoDExSmall --epochs 3 --bpe 1000 --gpus [0] --ckpt /path/to/checkpoint

python ./src/run_relation.py -c ./config/run_relation_inductive.yaml --dataset FB15k237Inductive --version v1 --epochs 3 --bpe 1000 --gpus [0] --ckpt /path/to/checkpoint
```

## Relation-graph-free message passing

TRIX's relation module (`RelNet`) runs an NBFNet over a *relation graph* built by `build_relation_graph`: one node per relation type and one edge per entity shared by two relations, in four flavours (head-head, head-tail, tail-head, tail-tail). This graph has `sum_v k_v^2` edges (at most `4|V|α²`, where `k_v` is the number of relations entity `v` appears in), typically 4 to 30 times the number of triples of the original graph; it is built in Python when a dataset is processed and stored with it.

Because TRIX uses the DistMult message and sum aggregation, the sum over relation-graph edges factors through the shared entities: one layer of `RelNet` can be computed with two sparse products by the binary entity × relation incidence matrices (at most `2|E|` non-zeros), relations → entities → relations, without building the relation graph. Same parameters, same output up to floating-point rounding; the `|V|α²` term of the time and memory complexity disappears. This is the alternate relational message passing of PathCon ([Wang, Ren, Leskovec, KDD 2021](https://arxiv.org/abs/2002.06757), edges → nodes → edges instead of the line graph) applied to the relation types of `RelNet`, where it happens to be exact. See `src/trix/relation_graph_free.py` for the derivation.

Enable it with `relation_graph_free: yes` in the `relation_model` section of a config, and `build_relation_graph: no` in the `dataset` section to skip building (and caching) the relation graph altogether:

```yaml
dataset:
  class: FB15k237Inductive
  version: v1
  build_relation_graph: no   # processed without the relation graph (separate cache file)
model:
  relation_model:
    relation_graph_free: yes
```

Both options work with the released checkpoints and with pre-training and fine-tuning (`relation_graph_free` is read by `RelNet` of both the entity and the relation prediction models). `check_relation_graph_free.py` compares the two implementations on one graph with a checkpoint: sizes, scores on test triples with all negatives, gradients after one training step, and timings.

```bash
python ./src/check_relation_graph_free.py --task entity --ckpt ./entity_prediction.pth --dataset FB15k237Inductive --version v1 --root ./kg-datasets --device cuda:0
python ./src/check_relation_graph_free.py --task relation --ckpt ./relation_prediction.pth --dataset WN18RRInductive --version v1 --root ./kg-datasets
```

With the released entity-prediction checkpoint (CPU, float32, 8 positives per batch with all negatives for the forward pass and 32 sampled negatives for the training step), the maximum relative score difference is below 2e-5 on every graph below, on CPU and on an A100. Gradients depend on the platform's float32 summation order: on macOS (torch 2.12) the two implementations agree to 5e-4 and are both within 1e-4 of a float64 reference; on Linux (torch 2.14, CPU and A100) the materialized implementation itself is off the float64 reference by up to 0.4 in relative norm on 33 tensors of the relation module on FB15k237 v1 (cancellation in the r = r' self term of the head-head and tail-tail flavours, where the gradients are small), while the relation-graph-free gradients stay within 1e-5 of the reference on every graph and platform. `--float64-reference` prints this comparison; the relation-prediction checkpoint gives the same picture.

| graph (test split) | triples | relation graph edges | incidences | forward (materialized → free) | forward + backward (materialized → free) |
|---|---:|---:|---:|---:|---:|
| WN18RR v1 | 3 236 | 19 304 | 4 272 | 0.019 s → 0.020 s | 0.28 s → 0.11 s |
| FB15k237 v1 | 3 986 | 34 724 | 5 228 | 0.031 s → 0.033 s | 0.43 s → 0.15 s |
| NELL v1 | 1 666 | 12 616 | 1 696 | 0.011 s → 0.011 s | 0.16 s → 0.06 s |
| FB15k237 (transductive, training graph) | 544 230 | 8 689 114 | 299 378 | 1.85 s → 0.72 s | 72.4 s → 9.6 s |

End-to-end, the zero-shot evaluation scripts (`run_entity.py` / `run_relation.py` with `--epochs 0` and the released checkpoints) give identical metrics with and without the relation graph on WN18RR v1 (entity and relation prediction) and NELL v1 (relation prediction). On NELL v1 entity prediction the MRR moves from 0.809 to 0.819: this split has many candidates with exactly or nearly equal scores, and `compute_ranking` is pessimistic (a tie counts against the positive), so rounding differences of 1e-5 change 29 of the 402 rankings. The materialized implementation is just as sensitive: the same model on the same graph with its edges listed in another order gives an MRR of 0.821 (31 rankings change for a maximum score difference of 7e-6).

On an A100 80 GB (torch 2.14, PyG 2.4, training graphs of the transductive datasets, batch of 32 positives × 512 negatives as in pre-training, median of 5 runs, released checkpoint):

| graph (training) | triples | relation graph edges | incidences | relation module (materialized → free) | forward + backward (materialized → free) | peak memory (materialized → free) |
|---|---:|---:|---:|---:|---:|---:|
| CoDExSmall | 65 776 | 314 158 | 23 734 | 0.040 → 0.019 s | 0.109 → 0.061 s | 0.7 → 0.8 GiB |
| WN18RR | 173 670 | 1 050 906 | 207 018 | 0.581 → 0.213 s | 1.525 → 0.423 s | 12.0 → 14.4 GiB |
| ConceptNet100k | 200 000 | 1 094 878 | 216 358 | 0.387 → 0.393 s | 0.783 → 0.758 s | 22.7 → 27.5 GiB |
| CoDExMedium | 371 168 | 2 623 864 | 202 272 | 0.455 → 0.112 s | 1.363 → 0.294 s | 5.3 → 6.1 GiB |
| YAGO3-10 | 2 158 080 | 5 364 460 | 784 244 | 1.846 → 0.733 s | 5.435 → 2.133 s | 36.5 → 43.7 GiB |
| Hetionet | 4 050 354 | 6 958 938 | 407 266 | 1.016 → 0.293 s | 3.184 → 1.029 s | 14.6 → 16.8 GiB |
| CoDExLarge | 1 102 386 | 8 217 030 | 760 606 | 2.006 → 0.477 s | 5.619 → 1.130 s | 23.5 → 27.6 GiB |
| FB15k237 | 544 230 | 8 689 114 | 299 378 | 0.378 → 0.116 s | 1.128 → 0.288 s | 5.4 → 5.4 GiB |
| DBpedia100k | 1 195 144 | 10 616 830 | 908 478 | 1.246 → 0.599 s | 3.306 → 1.267 s | 30.3 → 35.5 GiB |
| AristoV4 | 485 134 | 30 908 970 | 343 138 | 0.766 → 0.266 s | 2.041 → 0.548 s | 17.4 → 17.0 GiB |

Training with the same 300 batches from the released checkpoint (AdamW, 5e-4): 1.13 → 0.29 s per step on FB15k237 and 2.05 → 0.53 s on AristoV4, validation MRR of the two trained models within 0.001.

The gain follows the ratio of relation-graph edges to incidences, except on ConceptNet100k (78 k entities for 200 k triples): one layer costs O(|E| + |V|) here (the entity representations are a dense [num_nodes, batch × dim] tensor) against O(Σ_v k_v²) for the relation graph, so the gain is bounded by |E| / |V| and vanishes on graphs with few relations per entity. Peak memory of a training step is dominated by the entity models in both cases; the relation graph itself goes from up to 707 MiB (AristoV4) to 8 MiB of incidences, and the free version keeps a [num_nodes, batch × dim] intermediate per flavour, hence the +5 to +20% on some graphs.

On the small GraIL test graphs the relation module is a minor part of the forward pass; the saving grows with the size of the relation graph relative to the original graph (hubs, many relations per entity), as on the FB15k237 training graph where the relation graph has 16× more edges than the graph itself (29× more than the incidences), and where building and storing it is the limiting factor.

## LLM Experiment
Please download the CoDEx dataset from [CoDEx](https://github.com/tsafavi/codex).

Inferences of relation prediction tasks and entity prediction tasks with Gemini on CoDEx-S dataset:
```bash
python ./llm/run_relation_task_1.py

python ./llm/run_relation_task_2.py

python ./llm/run_relation_task_3.py

python ./llm/run_entity_task_1.py

python ./llm/run_entity_task_2.py

python ./llm/run_entity_task_3.py
```

## Citation ##

If you find this codebase useful in your research, please cite the original paper.

The main TRIX paper:

```bibtex
@inproceedings{zhang2024trix,
    title={TRIX: A More Expressive Model for Zero-shot Domain Transfer in Knowledge Graphs},
    author={Yucheng Zhang and Beatrice Bevilacqua and Mikhail Galkin and Bruno Ribeiro},
    booktitle={The Third Learning on Graphs Conference},
    year={2024},
    url={https://openreview.net/forum?id=mRB0XkewKW}
}
```