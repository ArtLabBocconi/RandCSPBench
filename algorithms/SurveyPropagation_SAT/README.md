# Survey Propagation Algorithm for Solving K-SAT Instances

This repository contains a C++ implementation of the **Survey Propagation (SP)** algorithm for solving random K-SAT instances.

---

## Compilation

To compile the code, run:

```bash
g++ -O3 main.cpp Graph.cpp Vertex.cpp walksat.cpp -o sp -std=c++11
```

---

## Running the Algorithm

To execute the solver, use:

```bash
./sp -l [instance_name.cnf]
```

### Notes:
- You can adjust the **fraction of variables to decimate** by modifying the appropriate value in the header file.
- In this version, the **message convergence flag has been removed**, meaning the algorithm runs without stopping if the convergence is not found.

---

## Extracting and Formatting Results

Use the following script to extract and log the results from the output:

```bash
file=[instance_name.cnf]
./sp -l $file > outfile.txt

running_time=$(grep "Tempo di esecuzione totale" outfile.txt | sed 's/[^0-9.]*\([0-9]\+\.[0-9]\+\).*/\1/')

if grep -q "ASSIGNMENT FOUND" outfile.txt ; then 
    min_value=$(grep -A 1000 "lowest" outfile.txt | grep -E '^\s*[0-9]' | awk '{print $1}' | sort -n | head -n 1)
    echo "$(basename "$file" | sed -E 's/N([0-9]+)_M([0-9]+)_id([0-9]+).cnf/\1 \2 \3/')  $min_value  $running_time" >> SAToutput_test.txt
elif grep -q "ASSIGNMENT NOT FOUND" outfile.txt ; then 
    min_value=$(grep -A 1000 "lowest" outfile.txt | grep -E '^\s*[0-9]' | awk '{print $1}' | sort -n | head -n 1)
    echo "$(basename "$file" | sed -E 's/N([0-9]+)_M([0-9]+)_id([0-9]+).cnf/\1 \2 \3/')  $min_value $running_time" >> SAToutput_test.txt
elif grep -q "The number of unsat clauses is:" outfile.txt ; then 
    min_value=$(grep "The number of unsat clauses is:" outfile.txt | sed 's/[^0-9]*\([0-9]\+\).*/\1/')
    echo "$(basename "$file" | sed -E 's/N([0-9]+)_M([0-9]+)_id([0-9]+).cnf/\1 \2 \3/')  $min_value $running_time" >> SAToutput_test.txt
else 
    exit
fi

# Cleanup
rm *.cnf
rm whitening*
rm Sol*
rm Value_Anal_C*
```

---

## Notes

- This implementation is optimized for **random K-SAT** instances.
- Be sure to adjust parameters and output scripts depending on the dataset or research focus.


## Contact

Feel free to open an issue or submit a pull request for suggestions, improvements, or questions. 
