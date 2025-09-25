SAforKSAT.c runs the Simulated Annealing algorithm to solve a K-SAT formula.
After compiling the code, you can run it as follows:

```
./SAforKSAT formula.cnf A
```

where `formula.cnf` is a file in CNF format containing the K-SAT formula
to be solved and the integer number `A` sets the maximum number of
MCMC sweeps to `maxIter = A * N`, being N the size of the K-SAT instance
The C code must be compiled with the right value of K set through the
first #define directive.
