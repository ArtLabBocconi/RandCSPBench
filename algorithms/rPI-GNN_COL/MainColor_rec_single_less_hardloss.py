import torch
import sys
import os

from time import time


from Utils_rec_single_less_hardloss import(get_adjacency_matrix, saver_loss, saver_colorings, saver_others, get_gnn, 
                             run_gnn_training_early_stop, SyntheticDataset)


# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')


def read_params(fileparams):
    fpar = open(fileparams, "r")
    fpar.readline()
    line = fpar.readline().split()
    fpar.close()
    return int(line[0]), int(line[1]), float(line[2]), float(line[3])

# Specify the problem instance to solve and where to find the dataset(s) here:
q = int(sys.argv[1])
filename = sys.argv[2]
nepochs = int(float(sys.argv[3]))
path_loss = sys.argv[4]
path_colorings = sys.argv[5]
path_others = sys.argv[6]

fileparams = sys.argv[7]

init_seed = int(sys.argv[8])
ntries = int(sys.argv[9])

randdim, hiddim, dout, lrate = read_params(fileparams)

print("Parameters are:")
print(f'randdim={randdim}   hiddim={hiddim}    dout={dout}    lrate={lrate}')

filename_without_ext = os.path.splitext(os.path.basename(filename))[0]

data_train = SyntheticDataset(filename, q)
print('Dataset ready\n')


hypers = {
        'model': 'GraphSAGE',
        'dim_embedding': randdim + 2 * data_train.chr_n,
        'dim_rand_input': randdim,
        'dropout': dout,
        'learning_rate': lrate,
        'hidden_dim': hiddim,
        'seed': init_seed
}

torch.set_printoptions(threshold=3705,linewidth=160)


# Retrieve known optimizer hypers
opt_hypers = {
    'lr': hypers.get('learning_rate', None)
}



# Default meta parameters
solver_hypers = {
    'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
    'number_epochs': nepochs,   # Max number training steps
    'patience': 10000,             # Number early stopping triggers before breaking loop
    'graph_file': data_train.fname,  # Which problem is being solved
    'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
    'number_classes': data_train.chr_n  #data_train.GetChrom(i)#
}

        # Combine into a single set
hypers.update(solver_hypers)

        # Get adjacency matrix for use in calculations
adj_ = get_adjacency_matrix(data_train.nxgraph, TORCH_DEVICE, TORCH_DTYPE)

cond = True
min_cost = data_train.nxgraph.number_of_edges()
times = []
while hypers['seed'] < init_seed + ntries and cond:
    t_start = time()
    print("\nTrying seed=", hypers['seed'])
    net, embed, optimizer = get_gnn(data_train.chr_n, data_train.fname, 
                                    data_train.nxgraph.number_of_nodes(), hypers, opt_hypers, 
                                    TORCH_DEVICE, TORCH_DTYPE)

    name,losses, prob,best_coloring,best_loss,final_coloring,final_loss,epoch_num, best_cost = run_gnn_training_early_stop(
            hypers['graph_file'], data_train.edges_list, data_train.graph, adj_, net, embed, 
            optimizer, randdim, hypers['number_epochs'], hypers['patience'], hypers['tolerance'], hypers['seed'])
    
    hypers['seed'] += 1

    if best_cost < 0.5:
        cond = False
        print("Success with seed=", hypers['seed'] - 1)
        min_cost = 0
    elif min_cost > best_cost:
        min_cost = best_cost
    runtime_gnn = round(time() - t_start, 4)
    times.append(runtime_gnn)

        # report results
print(f'GNN runtime: {sum(times)}s')

str_file = f'recurrent_less_hardloss_q_{data_train.chr_n}_randdim_{randdim}_hidim_{hiddim}_dout_{"{0:.3f}".format(dout)}_lrate_{"{0:.3f}".format(lrate)}_ntrials_{ntries}_nep_{nepochs}_filename_{filename_without_ext}'

loss_filename = "loss_" + str_file
cols_filename = "coloring_" + str_file
others_filename = "others_" + str_file
saver_loss(losses, path_loss, loss_filename)
saver_colorings(best_coloring, path_colorings, cols_filename, data_train.nx_orig, final_coloring)
others = [min_cost, times[0], hypers['seed'] - 1, epoch_num]
for i in range(len(times)):
    others.append(times[i])
saver_others(others, path_others, others_filename)