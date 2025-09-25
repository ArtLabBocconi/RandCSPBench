# RandSATBench: Benchmarks for Constraint Satisfaction Problems

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)


This repository contains all the code and data required to reproduce the benchmarks presented in the paper titled ["Benchmarking Graph Neural Networks in Solving Hard Constraint Satisfaction Problems"](TODO-ARXIV). 

Our benchmark, **RandSATBench**, provides a mixed set of easy, hard, and unsatisfiable instances of constraint satisfaction problems, 
in particular *q-coloring* and *Boolean satisfiability* problems. 

The goal is to have a challenging setting where to compare the performance of deep learning methods (in particular Graph Neural Networks) versus classical exact and heuristic solvers. 

## Structure of the Repository

The repository is structured as follows:

- `datasets/`: Scripts for generating the dataset.
- `algorithms/`: Contains the source code with the implementation of classical and GNN-based algorithms.
- `results_evaluation/`: Contains the results of the evaluation of the algorithms on the dataset. The result files in this repo are used to produce all plots and tables in the paper.

## Installation

Clone the repo with 
```
git clone https://github.com/ArtLabBocconi/RandSATBench
```

**Python**. We recommend [uv](https://docs.astral.sh/uv/) for managing the Python dependencies, provided in the `pyproject.toml` file. 
Install `uv` with 
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Julia**. The most convenient way to install Julia is through [juliaup](https://github.com/JuliaLang/juliaup). 
Download and execute it as follows:
```
curl -fsSL https://install.julialang.org | sh
```


## Datasets

We provide 2 datasets for Boolean satisfiability, where each clause in each instance contains 3 and 4 literals, respectively.
We also provide 2 datasets for coloring problems (where 3 and 5 colors are allowed, respectively).

Each dataset contains a split for the train and test sets. The problem sizes are $N=16,32,64,256$, and the datasets span different connectivities. 
Optionally, out-of-distribution test datasets with sizes up to $N=16384$ can be created. Not all instances of the problems admit solutions.

| Dataset | # Train Instances |# Test Instances |
|---------|-------------|---------------|
| 3-SAT    |     168000    |   42000    | 
| 4-SAT    |     84000    |   21000    | 
| 3-col    |      60000   | 20000   | 
| 5-col    |      60000   | 20000   | 

Below, we explain how to generate the datasets.

### K-SAT

The datasets containing random instances of 3-SAT or 5-SAT problems in CNF format can be either downloaded or generated.

Download with
```
wget https://huggingface.co/datasets/CarloLucibello/kSAT-Benchmarks/resolve/main/kSAT.zip
unzip kSAT.zip
```

Otherwise, generate the instances executing
```
cd datasets
julia gen_graphs_sat.jl
# or 
julia gen_graphs_sat.jl --test-ood # to also generate larger problem sizes
```

We also provide ground truth solutions obtained by running the CaDiCal solver for a limited amount of time on each instance. 
The solutions can be found in the files `datasets/3SAT/train_labels.csv` and `datasets/4SAT/train_labels.csv`. A labels' file contains in each row a cnf file name, whether it could be solved or not by CaDiCal, and the assignment of variables (minus denotes negation).



### q-coloring

In order to generate the random graphs for the 3-coloring and 5-coloring benchmarks, run the following:
```
cd datasets
python gen_graphs_coloring.py
# or 
uv run python gen_graphs_coloring.py # if using uv
```

The script can take the option `--test-ood` to generate a test set with larger system sizes as well. The option `--to-cnf` also generates Boolean SAT reductions in CNF format for each problem.

No ground truth configurations are available for supervised training on this data.


## Algorithms

Algorithms' implementations can be found in the `algorithms/` folder.
Check each algorithm's folder for instructions on how to run it.

The repo contains implementations of the following SAT solvers:

**SAT: Classical Algorithms**
- Belief Propagation
- CaDiCal
- Focused Metropolis Search
- Simulated Annealing
- Survey Propagation
- WalkSAT

**SAT: Neural Solvers**
- QuerySAT
- NeuroSAT

For coloring problems, either reduction to a SAT instance can be used, or the following specialized algorithms are available:

**Coloring: Classical Algorithms**
- Belief Propagation
- Focused Metropolis Search
- Simulated Annealing

**Coloring: Neural Algorithms**
- QuerySAT
- rPI-GNN

## Results

The folder `results_evaluation/` contains the results of the evaluation of the algorithms on the dataset. The result files in this repo are used to produce all plots and tables in the paper.

The folder `results_evaluation/analysis-results/` is produced by the notebook `results_evaluation/analysis.ipynb` and contains aggregated statistics in csv format.

The files `results_evaluation/analysis-results/[DATASET]_best_per_sample.csv` are particularly useful, as they contain the best performance of any algorithm on each instance of the dataset.

In the table below, we report for each test dataset the number of instances and the number of instances solved by at least one algorithm within the time limit.

| Dataset | # Test Instances | # Solved |
|---------|-------------|----------|
| 3SAT    | 42000        | 28657   |
| 4SAT    |     21000    |    20032   |
| 3COL    |      20000   |    11760   |
| 5COL    |      20000   |    9409    |


The third column can be used to normalize the score obtained by each algorithm, so that the maximum score is 1 (the score of an algorithm that solves any solvable instance), and the minimum score is 0 (the score of a dummy algorithm that never solves any instance). Notice that 1 is not a strict upper bound, as it could be the case that some very hard instances have not been solved by any algorithm we tested within the time limit.

## Contact

For any questions or issues, please open an issue on this repository or contact the authors.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
