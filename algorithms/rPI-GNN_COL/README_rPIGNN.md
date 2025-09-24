# Recurrent Physically Inspired Graph Neural Network (rPI-GNN) for q-coloring

## INTRODUCTION

The scripts:

    1. MainColor_rec_single_less_hardloss.py
    2. Utils_rec_single_less_hardoss.py
    3. MainColor_rec_parallel.py
    4. Utils_rec_parallel.py

implement the recurrent Physically-Inspired Graph Neural Network (rPI-GNN) applied to q-coloring instances.

Scripts (1) and (2) are built to receive a single graph and to train the network for that specific graph. 

With scripts (3) and (4) the user can simultaneously train over batches of graphs  



## SETTING UP THE ENVIRONMENT

The rPI-GNN is implemented using the modules pytorch and dgl. To use the computing resources available to us we installed 
the versions pytorch=2.4.1 and dgl=2.4.0 with CUDA=11.8. In case the user wants to use the exact same environment we provide
the file "pi-gnn_env.yml". To install the requirements, simply run the command:

```bash
conda env create -f pi-gnn_env.yml
```


## TRAINING

### I- Training for a single graph

File: MainColor_rec_single_less_hardloss.py
Positional argummients:

    1. q (number of colors, must be a positive integer)
    2. filegraph (name of the file with the information about the graph)
    3. nepochs (number of epochs, must be a positive integer)
    4. path_loss (path to the output file with the evolution of the training loss)
    5. path_colorings (path to the output file with the final coloring)
    6. path_others (path to the output file with other statistics about training, see below for more details)
    7. fileparams (name of the file with the parameters necessary to build the network, see below for more details)
    8. init_seed (initial seed for the random number generators used to create the network)
    9. ntries (number of independent attemps to obtain an optimal coloring of the graph)

Files with the information about the graphs:
    
    - Please take a look to our dataset to see the structure of these files.

Files with the parameters:
    
    - We considered 4 relevant parameters:
        a. Number of random entries in the embeddings of the first layer (randdim in the code)
        b. Dimension of the embeddings in the hidden layer (hiddim in the code)
        c. dropout
        d. learning rate
    - The parameters are read from the file in this order.
    - We provide the file "params_paper_recurrence.txt" as an example. It contains the default parameters used by us.

Output:

    - besides the evolution of the loss and the final coloring, we print a separate file with a single line containing the following numbers:
        a. minimum cost achieved
        b. clock time spent in the first try (the rPI-GNN makes 'ntries' attemps to color the graph)
        c. last seed used to initialize the network
        d. number of epochs in the last attempt
        e. list of clock times spent on previous attempt
    - the program stops whenever it finds a coloring without errors.


### II- Training for several graphs in parallel

File: MainColor_rec_parallel.py
Positional argummients:

    1. q (number of colors, must be a positive integer)
    2. folder_graphs (folder containing all the files with the information about the graphs. The program will load all of them simultaneously)
    3. nepochs (number of epochs, must be a positive integer)
    4. path_loss (path to the output file with the evolution of the training loss)
    5. path_colorings (path to the output file with the final coloring)
    6. path_others (path to the output file with other statistics about training, see above for more details)
    7. fileparams (name of the file with the parameters necessary to build the network, see above for more details)
    8. init_seed (initial seed for the random number generators used to create the network)
    9. ntries (number of independent attemps to obtain an optimal coloring of the graph)
    10. unique_id (string to be appended at the end of the names of the output files, to identify the batch)


In this case the files with the final colorings also contain an extra row with:

    a. seed that lead to the minimum cost
    b. minimum cost


## PROCESSING RESULTS


We provide the python script "parse_output.py", which we used to process the results of the rPI-GNN.
We hope the code is short, simple and clear enough. Since the shape of the output necessarily depends
on whether we train over a single graph or we train simultaneously over a batch of graphs, the function 
"parse_all()" receives a flag (default is set to True) that allows the user to select one of two options.
    
    - flag=True is for processing results from several trainings, each one done over a single graph. In this
    case the information is extracted from the files starting with 'others...'
    - flag=False is for processing results from training in parallel over a batch of graphs. In this case
    the information is extracted from the files starting with 'coloring...'