import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN1D
import random
import numpy as np
import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.data import DGLDataset
from time import time
import psutil
import os


def set_seed(seed):
    """
    Sets random seeds for training.
    :param seed: Integer used for seed.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_adjacency_matrix(nx_graph, torch_device, torch_dtype):
    """
    Pre-load adjacency matrix, map to torch device
    :param nx_graph: Graph object to pull adjacency matrix for
    :type nx_graph: networkx.Graph
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str
    :return: Adjacency matrix for provided graph
    :rtype: torch.tensor
    """

    adj = nx.linalg.graphmatrix.adjacency_matrix(nx_graph).todense()
    adj_ = torch.tensor(adj).type(torch_dtype).to(torch_device) #torch_dtype is float32 in their case
    return adj_


def saver_loss(lista1, path, name): #IGNORE
    print(f'saving losses in {name}')
    # Write-Overwrites
    file1 = open(f'{path}/{name}.txt', "w")  # write mode
    epoch_i = 0
    while epoch_i < min(100, len(lista1)):
        file1.write(f'{epoch_i}\t{"{0:.4f}".format(lista1[epoch_i])}\n')
        epoch_i += 1
    while epoch_i < len(lista1):
        file1.write(f'{epoch_i}\t{"{0:.4f}".format(lista1[epoch_i])}\n')
        epoch_i = int(epoch_i * 1.02)
    file1.close()



def update_best(folder, best_colorings, best_costs, coloring_seed, cost_seed, seed, best_seeds):    
    for fname in os.listdir(folder):
        if cost_seed[fname] < best_costs[fname]:
            best_costs[fname] = cost_seed[fname]
            best_colorings[fname] = coloring_seed[fname]
            best_seeds[fname] = seed


def get_full_colors(best_coloring, all_isolates, nnodes_orig, folder):
    best_colors_full = {}
    for fname in os.listdir(folder):
        isol = all_isolates[fname]
        offset = 0
        inner = []
        for i in range(nnodes_orig[fname]):
            if i in isol:
                inner.append(0)
                offset += 1
            else:
                inner.append(best_coloring[fname][i - offset].item())
        best_colors_full[fname] = inner
    return best_colors_full


def saver_colorings(best_colorings_full, best_costs, path, name_start, 
                    folder, node_offset, seed):    

    for fname in os.listdir(folder):
        cost = 0
        with open(f'{folder}/{fname}', 'r') as f:
                content = f.read().strip()
        lines = content.split('\n')  # skip comment line(s)
        nedges=int(lines[1].split()[1])
        for line in lines[2:nedges+2]:
            edge = parse_line(line, node_offset)
            if best_colorings_full[fname][edge[0]] == best_colorings_full[fname][edge[1]]:
                cost += 1
        if cost != best_costs[fname]:
            print("There is a problem with the inter computation of the costs")
        elif cost == 0:
            fname_without_ext = os.path.splitext(os.path.basename(fname))[0]
            file3 = open(f'{path}/{name_start}_filename_{fname_without_ext}.txt', "w")  # write mode
            file3.write(str(best_colorings_full[fname])+'\n')#final coloring
            file3.write(f'{seed}\t{best_costs[fname]}')
            file3.close()
            os.remove(f'{folder}/{fname}')


def saver_colorings_final(best_colorings, best_costs, path, name_start, 
                            folder, best_seed):    

    for fname in os.listdir(folder):
        fname_without_ext = os.path.splitext(os.path.basename(fname))[0]
        file3 = open(f'{path}/{name_start}_filename_{fname_without_ext}.txt', "w")  # write mode
        file3.write(str(best_colorings[fname])+'\n')#final coloring
        file3.write(f'{best_seed[fname]}\t{best_costs[fname]}')
        file3.close()
        os.remove(f'{folder}/{fname}')


    


def saver_others(others, path, name):
    file3 = open(f'{path}/{name}.txt', "w")
    file3.write(str(others[0]))
    for i in range(1, len(others)):
        file3.write("\t" + str(others[i]))
    file3.close()


def parse_line(file_line, node_offset=0):
    splitted = file_line.split()
    x = int(splitted[1])
    y = int(splitted[2])
    x, y = x+node_offset, y+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y


def init_best(folder):
    best_costs = {}
    best_colorings = {}
    best_seeds = {}
    for fname in os.listdir(folder):
        best_costs[fname] = np.inf
        best_seeds[fname] = np.inf
        best_colorings[fname] = []
    return best_colorings, best_costs, best_seeds


def get_dgl_graph(folder, fname, node_offset):
    nx_orig = nx.Graph()
    print(f'Building graph from contents of file: {fname}', flush=True)
    with open(f'{folder}/{fname}', 'r') as f:
        content = f.read().strip()
    lines = content.split('\n')  # skip comment line(s)
    n=int(lines[0].split()[1])
    nedges=int(lines[1].split()[1])
    edgesnx=[parse_line(line, node_offset) for line in lines[2:nedges+2]]
    for i in range(n):
        nx_orig.add_node(i)
    for edge in edgesnx:
        nx_orig.add_edge(edge[0], edge[1])

    nx_clean = nx_orig.copy()
    nx_clean.remove_nodes_from(list(nx.isolates(nx_clean)))
    nx_clean = nx.convert_node_labels_to_integers(nx_clean)

    return nx_orig, nx_clean
    


class SyntheticDataset(DGLDataset):
    def __init__(self, folder, torch_device, torch_dtype, node_offset=-1):
        self.folder=folder
        self.node_offset=node_offset
        self.torch_device=torch_device
        self.torch_dtype=torch_dtype
        self.isolated_nodes = {}
        self.nnodes_orig = {}
        self.nnodes_clean = {}
        self.nx_clean_edges = {}
        self.all_adj_matrix = {}
        super().__init__(name="synthetic")

    def process(self):

        fname_list = os.listdir(self.folder)
        if len(fname_list) > 0:
            nx_orig, nx_clean = get_dgl_graph(self.folder, fname_list[0], self.node_offset)
            self.nnodes_orig[fname_list[0]] = nx_orig.number_of_nodes()
            self.nnodes_clean[fname_list[0]] = nx_clean.number_of_nodes()
            isol = list(nx.isolates(nx_orig))
            self.isolated_nodes[fname_list[0]] = isol
            self.nx_clean_edges[fname_list[0]] = np.array(nx_clean.edges())
            self.all_adj_matrix[fname_list[0]] = get_adjacency_matrix(nx_clean, self.torch_device, self.torch_dtype)
            self.graph = dgl.batch([dgl.from_networkx(nx_clean, device=self.torch_device)])
            for i in range(1, len(fname_list)):
                nx_orig, nx_clean = get_dgl_graph(self.folder, fname_list[i], self.node_offset)
                self.nnodes_orig[fname_list[i]] = nx_orig.number_of_nodes()
                self.nnodes_clean[fname_list[i]] = nx_clean.number_of_nodes()
                isol = list(nx.isolates(nx_orig))
                self.isolated_nodes[fname_list[i]] = isol
                self.nx_clean_edges[fname_list[i]] = np.array(nx_clean.edges())
                self.all_adj_matrix[fname_list[i]] = get_adjacency_matrix(nx_clean, self.torch_device, self.torch_dtype)
                self.graph = dgl.batch([self.graph, dgl.from_networkx(nx_clean, device=self.torch_device)])
            


# Define GNN GraphSage object
class GNNSage(nn.Module):
    """
    Basic GraphSAGE-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """

    def __init__(self, in_feats, hidden_size, num_classes, dropout, TORCH_DEV, TORCH_DTYPE):
        """
        Initialize the model object. Establishes model architecture and relevant hypers (`dropout`, `num_classes`, `agg_type`)
        :param g: Input graph object
        :type g: dgl.DGLHeteroGraph
        :param in_feats: Size (number of nodes) of input layer
        :type in_feats: int
        :param hidden_size: Size of hidden layer
        :type hidden_size: int
        :param num_classes: Size of output layer (one node per class)
        :type num_classes: int
        :param dropout: Dropout fraction, between two convolutional layers
        :type dropout: float
        :param agg_type: Aggregation type for each SAGEConv layer. All layers will use the same agg_type
        :type agg_type: str
        """
        
        super(GNNSage, self).__init__()

        self.num_classes = num_classes
        
        # input layers
        self.l1_mean = SAGEConv(in_feats, hidden_size, "mean", activation=F.relu)
        self.l1_pool = SAGEConv(in_feats, hidden_size, "pool", activation=F.relu)
        
        # output layer
        self.outlayer = SAGEConv(hidden_size, num_classes, "mean")
        self.dropout = nn.Dropout(p=dropout)

        self.batch_norm_mean = BN1D(hidden_size, device=TORCH_DEV, dtype=TORCH_DTYPE)
        self.batch_norm_pool = BN1D(hidden_size, device=TORCH_DEV, dtype=TORCH_DTYPE)

        self.relu = nn.ReLU()

    def forward(self, g, features):
        """
        Define forward step of netowrk. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.
        :param features: Input node representations
        :type features: torch.tensor
        :return: Final layer representation, pre-activation (i.e. class logits)
        :rtype: torch.tensor
        """
        h = features

        h = self.relu(torch.add(self.batch_norm_mean(self.l1_mean(g, h)), 
                             self.batch_norm_pool(self.l1_pool(g, h))))
        h = self.dropout(h)
        h = self.outlayer(g, h)

        return h


# Construct graph to learn on #
# Construct graph to learn on #
def get_gnn(n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    """
    Helper function to load in GNN object, optimizer, and initial embedding layer.
    :param n_nodes: Number of nodes in graph
    :type n_nodes: int
    :param gnn_hypers: Hyperparameters to provide to GNN constructor
    :type gnn_hypers: dict
    :param opt_params: Hyperparameters to provide to optimizer constructor
    :type opt_params: dict
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str
    :return: Initialized GNN instance, embedding layer, initialized optimizer instance
    :rtype: GNN_Conv or GNN_SAGE, torch.nn.Embedding, torch.optim.AdamW
    """

    try:
        print(f'Function get_gnn(): Setting seed to {gnn_hypers["seed"]}', flush=True)
        set_seed(gnn_hypers['seed'])
    except KeyError:
        print('!! Function get_gnn(): Seed not specified in gnn_hypers object. Defaulting to 0 !!', flush=True)
        set_seed(0)

    model = gnn_hypers['model']
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']
    randdim = gnn_hypers['dim_rand_input']

    # instantiate the GNN
    print(f'Building {model}...', flush=True)
    net = GNNSage(dim_embedding, hidden_dim, number_classes, dropout, torch_device, torch_dtype)
    
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)
    with torch.no_grad():
        embed.weight[:, randdim:] = 0

    # set up Adam optimizer
    # params = chain(net.parameters())

    print(f'Building ADAM-W optimizer...', flush=True)
    optimizer = torch.optim.AdamW(net.parameters(), **opt_params, weight_decay=1e-2)

    return net, embed, optimizer


# helper function for graph-coloring loss
def loss_func_mod(probs, all_adj_tensor, nnodes, folder, all_loss):
    """
    Function to compute cost value based on soft assignments (probabilities)
    :param probs: Probability vector, of each node belonging to each class
    :type probs: torch.tensor
    :param adj_tensor: Adjacency matrix, containing internode weights
    :type adj_tensor: torch.tensor
    :return: Loss, given the current soft assignments (probabilities)
    :rtype: float
    """

    # Multiply probability vectors, then filter via elementwise application of adjacency matrix.
    #  Divide by 2 to adjust for symmetry about the diagonal
    cumul = 0
    loss_ = 0
    for fname in os.listdir(folder):
        n = nnodes[fname]
        probs_part = probs[cumul:cumul + n, :]
        loss_part = torch.mul(all_adj_tensor[fname], (probs_part @ probs_part.T)).sum() / 2
        all_loss[fname] = loss_part
        loss_ += loss_part
        cumul += n

    return loss_


# helper function for custom loss according to Q matrix
def loss_func_color_hard(coloring, all_edges, nnodes, all_loss, best_colorings, found_sol, best_costs, folder):
    """
    Function to compute cost value based on color vector (0, 2, 1, 4, 1, ...)
    :param coloring: Vector of class assignments (colors)
    :type coloring: torch.tensor
    :param nx_graph: Graph to evaluate classifications on
    :type nx_graph: networkx.Graph
    :return: Cost of provided class assignments
    :rtype: torch.tensor
    """

    cumul = 0
    for fname in os.listdir(folder):
        n = nnodes[fname]
        if (not found_sol[fname]) and all_loss[fname] < 2:
            coloring_part = coloring[cumul:cumul + n]
            cost_ = len(all_edges[fname]) - torch.count_nonzero(coloring_part[all_edges[fname][:, 0]]-coloring_part[all_edges[fname][:, 1]])
            if cost_ == 0:
                best_costs[fname] = int(cost_)
                best_colorings[fname] = coloring_part
                found_sol[fname] = True
            elif cost_ < best_costs[fname]:
                best_costs[fname] = int(cost_)
                best_colorings[fname] = coloring_part
        cumul += n


# helper function for custom loss according to Q matrix
def loss_func_color_hard_final(coloring, all_edges, nnodes, best_colorings, found_sol, best_costs, folder):
    """
    Function to compute cost value based on color vector (0, 2, 1, 4, 1, ...)
    :param coloring: Vector of class assignments (colors)
    :type coloring: torch.tensor
    :param nx_graph: Graph to evaluate classifications on
    :type nx_graph: networkx.Graph
    :return: Cost of provided class assignments
    :rtype: torch.tensor
    """

    cumul = 0
    for fname in os.listdir(folder):
        n = nnodes[fname]
        if not found_sol[fname]:
            coloring_part = coloring[cumul:cumul + n]
            cost_ = len(all_edges[fname]) - torch.count_nonzero(coloring_part[all_edges[fname][:, 0]]-coloring_part[all_edges[fname][:, 1]])
            if cost_ == 0:
                best_costs[fname] = int(cost_)
                best_colorings[fname] = coloring_part
                found_sol[fname] = True
            elif cost_ < best_costs[fname]:
                best_costs[fname] = int(cost_)
                best_colorings[fname] = coloring_part
        cumul += n
    return sum(best_costs.values())


def run_gnn_training(all_edges, all_nnodes_clean, graph_dgl, all_adj_mat, net, embed, optimizer,
                     randdim, torch_device, folder, number_epochs=int(1e5), patience=1000, tolerance=1e-4, seed=1):
    t_start = time()

    """
    Function to run model training for given graph, GNN, optimizer, and set of hypers.
    Includes basic early stopping criteria. Prints regular updates on progress as well as
    final decision.
    :param nx_graph: Graph instance to solve
    :param graph_dgl: Graph instance to solve
    :param adj_mat: Adjacency matrix for provided graph
    :type adj_mat: torch.tensor
    :param net: GNN instance to train
    :type net: GNN_Conv or GNN_SAGE
    :param embed: Initial embedding layer
    :type embed: torch.nn.Embedding
    :param optimizer: Optimizer instance used to fit model parameters
    :type optimizer: torch.optim.AdamW
    :param number_epochs: Limit on number of training epochs to run
    :type number_epochs: int
    :param patience: Number of epochs to wait before triggering early stopping
    :type patience: int
    :param tolerance: Minimum change in cost to be considered non-converged (i.e.
        any change less than tolerance will add to early stopping count)
    :type tolerance: float
    :return: Final model probabilities, best color vector found during training, best loss found during training,
    final color vector of training, final loss of training, number of epochs used in training
    :rtype: torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, int
    """
    losses=[]
    # Ensure RNG seeds are reset each training run
    print(f'Function run_gnn_training(): Setting seed to {seed}', flush=True)
    set_seed(seed)

    best_colorings, best_costs, all_loss = init_best(folder)
    found_sol = {}
    for fname in os.listdir(folder):
        found_sol[fname] = False

    # Tracking
    
    # Early stopping to allow NN to train to near-completion
    prev_loss = 1.  # initial loss value (arbitrary)
    cnt = 0  # track number times early stopping is triggered

    # Training logic
    epoch = 0
    while epoch < number_epochs:
        # get soft prob assignments
        logits = net(graph_dgl, embed.weight)
        # apply softmax for normalization
        probs = F.softmax(logits, dim=1)

        # get cost value with POTTS cost function
        #weight_classes=weight_classes_orig*factor
        loss = loss_func_mod(probs, all_adj_mat, all_nnodes_clean, folder, all_loss)
        coloring = torch.argmax(probs, 1)
        loss_func_color_hard(coloring, all_edges, all_nnodes_clean, all_loss, best_colorings, found_sol, best_costs, folder)
        
        if (abs(loss - prev_loss) <= tolerance) | ((loss - prev_loss) > 0):
            cnt += 1
        else:
            cnt = 0
        # print(f'epoch: {epoch}, cost_hard.item(): {cost_hard.item()}')
        
        losses.append(loss.item())
        '''
        if cost_hard.item()==0:#epoch>=int(2e3) and 
            print('Epoch %d | Soft Loss: %.5f | Discrete Cost: %.5f | calculated ChroNu: %d | ChroNu: %d' % (epoch, loss.item(), cost_hard.item(), torch.max(coloring)+1, chromatic_numbers[graphname]))
            break
        '''
        # if cost_hard.item()==0:
        #     print("0 reached")

        # update loss tracking
        prev_loss = loss

        if cnt >= patience:
            print(f'Stopping early on epoch {epoch}. Patience count: {cnt}', flush=True)
            break

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()  # calculate gradient through compute graph
        optimizer.step()  # take step, update weights

        # Rebuild input parameters for recurrence
        with torch.no_grad():
            embed.weight[:, randdim:] = torch.cat((logits, probs), dim=1)

        # tracking: print intermediate loss at regular interval
        if epoch % 500 == 0:
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info(device=0)
                print(f'Epoch {epoch} | Soft Loss: {loss.item():.5f} \
                    | time: {round(time() - t_start, 4)} |  CPU Usage: {psutil.cpu_percent()} \
                    | RAM Usage: {psutil.virtual_memory().used / (1024 ** 3)} GB  \
                    |  GPU memory {(total_mem - free_mem) / (1024 ** 3)} GB  \
                    |  GPU memory free {free_mem / (1024 ** 3)} GB \
                    |  GPU memory total {total_mem / (1024 ** 3)} GB', flush=True)
            else:
                print(f'Epoch {epoch} | Soft Loss: {loss.item():.5f} \
                    | time: {round(time() - t_start, 4)} |  CPU Usage: {psutil.cpu_percent()} \
                    | RAM Usage: {psutil.virtual_memory().used / (1024 ** 3)} GB', flush=True)
        epoch += 1
    # Print final loss
    # Final coloring
    final_loss = loss
    final_coloring = torch.argmax(probs, 1)
    print(f'Final soft loss: {final_loss:.3f}, chromatic_number: {torch.max(final_coloring)+1}', flush=True)

    final_cost = loss_func_color_hard_final(final_coloring, all_edges, all_nnodes_clean, best_colorings, 
                                            found_sol, best_costs, folder)

    print('Epoch %d | Final loss: %.5f | Final cost: %.5f' % (epoch, loss.item(), final_cost), flush=True)

    
    
    return losses, best_colorings, best_costs