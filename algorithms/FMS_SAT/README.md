# FMS Algorithm for Solving K-SAT Instances

This repository contains a C implementation of the **FMS (Focused Metropolis Search)** algorithm for solving random K-SAT instances using local search techniques.

---

## Compilation

To compile the code, run:

```bash
gcc -O3 wfacwsat.c -o FMS -lm
```

---

## Running the Algorithm

To run the solver, use the following command:

```bash
./FMS -noise [A] [B] -cutoff [C] -fms < [instance_name.cnf]
```

Where:
- `[A]` and `[B]` are noise parameters optimized for different SAT instance types:
  - **3-SAT**: `A = 37`, `B = 100`
  - **4-SAT**: `A = 293`, `B = 1000`
- `[C]` is the cutoff value, usually defined as `g * N²`, with `g = 100` for a balance between speed and precision.

---

## Extracting and Formatting Results

To extract and log solver results, use the following script:

```bash
file=[instance_name.cnf]
./FMS -noise [A] [B] -cutoff [C] -fms < $file > output.txt

running_time=$(grep "Tempo di esecuzione totale" output.txt | sed 's/[^0-9.]*\([0-9]\+\.[0-9]\+\).*/\1/')
min_value=$(grep -A 1000 "lowest" output.txt | grep -E '^\s*[0-9]' | awk '{print $1}' | sort -n | head -n 1)

export LC_NUMERIC=C

if grep -q "ASSIGNMENT FOUND" output.txt ; then 
    echo "$(basename "$file" | sed -E 's/sat_([0-9]+)_([0-9]+)_([0-9]+).cnf/\1 \2 \3/') $min_value $running_time 1 $g [C]" >> SAT_test.txt
else
    echo "$(basename "$file" | sed -E 's/sat_([0-9]+)_([0-9]+)_([0-9]+).cnf/\1 \2 \3/') $min_value $running_time 0 $g [C]" >> SAT_test.txt
fi
```

---

## Notes

- This implementation is optimized for random 3-SAT and 4-SAT instances.
- Adjust `g` and `[A]  [B]  [C]` values to fine-tune performance and runtime depending on instance size.

## Contact

Feel free to open an issue or submit a pull request for suggestions, improvements, or questions. 
