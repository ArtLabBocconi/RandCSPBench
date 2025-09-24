# Focused Metropolis Search implemented for the q-coloring problem

## INTRODUCTION

The C program 'FMS_from_file.c' implements the FMS for q-coloring.


## Compilation

To compile the code, run:

```bash
gcc FMS_from_file.c -o FMS_from_file.out -lm
```


## Running the algorithm

To run the solver, use the following command:

```bash
./FMS_from_file.out [eta] [maxIter] [graphname] [id]
```

where

  - **eta** (double) is the noise parameter. We set eta=0.37 for 3-COL and 5-COL, please adjust this parameter for your specific problem
  - **maxIter** (long) maximum number of iterations. We include results for the quadratic scalings maxIter=20N^2 and maxIter=100N^2, where N is the number of nodes in the graph.
  - **graphname** (char*) name of the file that contains the graph
  - **id** (int) is the identification of that file in your dataset.

## Output

The program outputs the following numbers:

1:N  2:M  3:id  4:lowestUnsat  5:c  6:eta  7:MCStoLowest  8:timeSpent(sec) 

where

 - **N** is the number of nodes in the graph.
 - **M** is the number of edges in the graph.
 - **id** is the identification number of the graph.
 - **lowestUnsat** is the lowest number of unsatisfied edges found by FMS this run
 - **c** is the connectivity of the graph (c = 2M / N).
 - **eta** noise parameter used this run.
 - **MCStoLowest** is the number of Monte Carlo Sweeps (each one consists of **N** proposed flips) until the assigment with the lowest energy was found.
 - **timeSpent(sec)** total clock time in seconds.