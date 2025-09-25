# Solving BenchmarkSAT with the CaDiCal solver 

## Setup
Make sure to first build CaDiCal as indicated in the ReadME file in `algorithms/CaDiCal/README.md`. By default, this requires you to run `./configure && make` to configure and build `cadical` and the library `libcadical.a` in the default `build` sub-directory, i.e., `algorithms/CaDiCal/build/`. This will naturally require a working C++ compiler. For more information refer to the CaDiCal Documentation

## Solving
Navigate to `algorithms/CaDiCal/solve_benchmarks/` and then run `solve.sh` as follows:
```
sh solve.sh k partition 
```
E.g., for the training and testing datasets of 3SAT one would run:
```
sh solve.sh 3 train && sh solve.sh 3 test
```

The script will run cadical and then extract the results in a csv file with the following format:
```
CNF FileName, SAT/UNSAT (Binary label), SAT Assignment (If SAT)
```
which is then automatically moved in the dataset directory

*Note that this whole procedure assumes you have generated the CNF files as specified in `datasets/generate_all.sh`*