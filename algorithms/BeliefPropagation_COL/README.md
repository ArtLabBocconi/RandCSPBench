# Belief Propagation algorithms for Q-COL

To replicate the results in the paper, two .sh files are presented. The options to change accordingly to local resources and folders structure are:

## Directory base
DIR_BASE="../../datasets/3COL/Q3/Q3_N_"

## Output base
OUTPUT_BASE="./Q3_solutions/DECIM_N_"

## Maximum number of concurrent jobs
MAX_JOBS=35

# Recomandations
Reinforcement has a time complexity linear in N, while Decimation is quadratic. For N>1024 Decimation becomes very expensive to run.
