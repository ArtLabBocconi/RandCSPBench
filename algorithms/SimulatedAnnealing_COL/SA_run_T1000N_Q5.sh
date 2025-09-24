#!/bin/bash

# List of N values
N_VALUES=(16 32 64 128 256 512 1024)

# List of nIter values
A=1000

# Other fixed parameters
C_MIN=9.9
C_MAX=13.5
DELTA_C=0.4
START_TEMP=1
N_SAMPLES=400

# Standard simulated annealing correspond to these two values
R=1
GAMMA=1

# Directory base
DIR_BASE="../../datasets/5COL/Q5/Q5_N_"

# Output base
OUTPUT_BASE="./Q5_solutions/A1000_SA_N_"

# Maximum number of concurrent jobs
MAX_JOBS=35

# Function to run a single job
run_job() {
    local N=$1
    local NITER=$((N * A))
    local DIR="${DIR_BASE}${N}/"
    local OUTPUT="${OUTPUT_BASE}${N}_Q5.txt"
    ./RSA.exe $N $C_MIN $C_MAX $DELTA_C $START_TEMP $NITER $N_SAMPLES $R $GAMMA $DIR >> $OUTPUT
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
