---
license: mit
pretty_name: RandCSPBench
task_categories:
- graph-ml
tags:
- constraint-satisfaction
- graph-coloring
- boolean-satisfiability
- combinatorial-optimization
- graph-neural-networks
- benchmark
size_categories:
- 100K<n<1M
configs:
- config_name: 3COL
  data_files:
  - split: train
    path: coloring/3COL/train-*.parquet
  - split: test
    path: coloring/3COL/test-*.parquet
  - split: test_ood
    path: coloring/3COL/test_ood-*.parquet
- config_name: 5COL
  data_files:
  - split: train
    path: coloring/5COL/train-*.parquet
  - split: test
    path: coloring/5COL/test-*.parquet
  - split: test_ood
    path: coloring/5COL/test_ood-*.parquet
---

# RandCSPBench

Random instances of hard constraint satisfaction problems, namely q-coloring of Erdős–Rényi graphs and random k-SAT, for benchmarking graph neural networks against classical solvers. This is the dataset of the paper [Benchmarking Graph Neural Networks in Solving Hard Constraint Satisfaction Problems](https://arxiv.org/abs/2602.18419).

Instances are drawn at connectivities on both sides of the satisfiability threshold, so every set mixes easy, hard and unsatisfiable problems. Code, solvers, and per-instance results of all benchmarked algorithms are on [GitHub](https://github.com/ArtLabBocconi/RandCSPBench).

## Contents

| Benchmark | Raw files | `datasets` config | Train | Test | Test OOD |
|---|---|---|---:|---:|---:|
| 3-coloring | `coloring/3COL.zip` | `3COL` | 60,000 | 20,000 | 24,000 |
| 5-coloring | `coloring/5COL.zip` | `5COL` | 60,000 | 20,000 | 24,000 |
| 3-SAT | `kSAT.zip` | – | 168,000 | 42,000 | 42,000 |
| 4-SAT | `kSAT.zip` | – | 84,000 | 21,000 | 12,600 |

Train and test instances have N = 16, 32, 64, 128, 256 variables (nodes). The out-of-distribution (OOD) test sets go from N = 512 up to N = 16384.

## q-coloring

### Loading with `datasets`

```python
from datasets import load_dataset

ds = load_dataset("CarloLucibello/CSPBenchmarks", "3COL", split="test")
```

Each row is one graph:

| Column | Description |
|---|---|
| `name` | File name of the instance inside the zip archive |
| `q` | Number of colors |
| `N` | Number of nodes |
| `M` | Number of edges |
| `c` | Mean degree on the generation grid (`M = round(c * N / 2)`) |
| `id` | Instance index within each (N, c) pair, starting from 1 |
| `edges` | `M x 2` list of edges with **0-based** node indices |
| `solved` | Whether at least one algorithm benchmarked in the paper found a proper coloring (null for train), from `results_evaluation/analysis-results/{q}COL-best_per_sample.csv` in the GitHub repository. An unsolved instance is either uncolorable or very hard. |

For example, with PyTorch Geometric: `edge_index = torch.tensor(row["edges"]).T`.

### Raw files

These are the exact files read by the solvers in the GitHub repository:

```bash
wget https://huggingface.co/datasets/CarloLucibello/CSPBenchmarks/resolve/main/coloring/3COL.zip
unzip 3COL.zip
```

```
3COL/train/       ErdosRenyi_N_{N}_M_{M}_id_{id}.txt   edge lists
3COL/test/
3COL/test_ood/
3COL/train-sat/   COLSAT_N{N}_M{M}_id{id}.cnf          SAT reductions (DIMACS)
3COL/test-sat/
```

Edge lists have a two-line header followed by one line per edge, with **1-based** node indices:

```
N	16
M	27
e	1	5
...
```

The CNF files encode q-coloring as SAT ([cnfgen](https://massimolauria.net/cnfgen/) `GraphColoringFormula`): variable `q*(v-1) + k` is true if and only if node `v` takes color `k`. CNF reductions of the OOD graphs are not included because of their size. Generate them with `python gen_graphs_coloring.py --gen-ood --to-cnf` from the GitHub repository.

### Generation

Graphs are sampled uniformly with N nodes and `M = round(c * N / 2)` edges (`networkx.gnm_random_graph`).

| | c grid | Train / test graphs per (N, c) |
|---|---|---|
| 3-coloring | 3.32, 3.50, …, 4.94 (10 values) | 1200 / 400 |
| 5-coloring | 9.9, 10.3, …, 13.5 (10 values) | 1200 / 400 |

Every OOD test set has 400 graphs per (N, c). The random seed is reset to 2 for train and to 1 for test and test OOD at the start of each (N, c) block.

The data were produced with `datasets/gen_graphs_coloring.py` (networkx 3.7, cnfgen 0.9.6) and checked against the instances evaluated in the paper. See `datasets/hf/` in the GitHub repository.

## k-SAT

`kSAT.zip` contains random k-SAT formulas in DIMACS CNF format, with N variables and `M = round(alpha * N)` clauses:

- 3-SAT: alpha = 3.0, 3.1, …, 5.0.
- 4-SAT: alpha = 9.0, 9.05, …, 10.0 for train and alpha = 8.0, 8.1, …, 10.0 for test.

```bash
wget https://huggingface.co/datasets/CarloLucibello/CSPBenchmarks/resolve/main/kSAT.zip
unzip kSAT.zip
```

```
3SAT/train/  3SAT/test/  3SAT/test_ood/                 N{N}_M{M}_id{id}.cnf
4SAT/train/  4SAT/test/  4SAT/test-ood/
3SAT/train_labels.csv  3SAT/test_labels.csv
4SAT/train_labels.csv  4SAT/test_labels.csv
```

The label files list for each formula whether CaDiCaL found a satisfying assignment within a short time limit, together with the assignment (a minus sign means negation).

## Citation

```bibtex
@article{skenderi2026benchmarking,
  title   = {Benchmarking Graph Neural Networks in Solving Hard Constraint Satisfaction Problems},
  author  = {Skenderi, Geri and Buffoni, Lorenzo and D'Amico, Francesco and Machado, David and Marino, Raffaele and Negri, Matteo and Ricci-Tersenghi, Federico and Lucibello, Carlo and Angelini, Maria Chiara},
  journal = {arXiv preprint arXiv:2602.18419},
  year    = {2026}
}
```
