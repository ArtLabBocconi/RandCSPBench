# Solving BenchmarkSAT with the CaDiCal solver 

## Setup
Make sure to first build CaDiCal as indicated in the ReadME file in `algorithms/CaDiCal/README.md`. By default, this requires you to run `./configure && make` to configure and build `cadical` and the library `libcadical.a` in the default `build` sub-directory, i.e., `algorithms/CaDiCal/build/`. This will naturally require a working C++ compiler. For more information refer to the CaDiCal Documentation

## Solving
Navigate to `algorithms/CaDiCal/solve_coloring/` and then run `solve.sh` as follows:
```
sh solve.sh q
```

*Note that this whole procedure assumes you have generated the CNF files as specified in through `generate_graphs_local.sh`*