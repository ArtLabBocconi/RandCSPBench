#!/bin/bash
SEED=55

# List of N values
N_VALUES=(16 32 64 128 256 512 1024 2048 4096 8192)

# Other fixed parameters
C_MIN=2.96
C_MAX=5.0
DELTA_C=0.18
N_SAMPLES=400

# OPTIMIZED, DON'T CHANGE
NITER=10000

# Directory base
DIR_BASE="../../datasets/3COL/Q3/Q3_N_"

# Output base
OUTPUT_BASE="./Q3_solutions/DECIM_N_"

# Maximum number of concurrent jobs
MAX_JOBS=35

# Function to run a single job
run_job() {
    local N=$1
    local DIR="${DIR_BASE}${N}/"
    local OUTPUT="${OUTPUT_BASE}${N}_Q3.txt"
    ./colBP_decim_quadratic.exe $N $C_MIN $C_MAX $DELTA_C $NITER $N_SAMPLES $DIR >> $OUTPUT
}

# Function to wait for jobs if maximum concurrency is reached
wait_for_jobs() {
    while [ $(jobs | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
        jobs > /dev/null
    done
}

# Loop through all combinations of N and nIter
for N in "${N_VALUES[@]}"; do
    run_job $N &
    wait_for_jobs
done

# Wait for all background jobs to complete
wait
