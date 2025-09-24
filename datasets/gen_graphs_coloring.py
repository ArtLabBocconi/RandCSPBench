import networkx as nx
import os
import sys
import numpy as np
import random
import argparse
from cnfgen import GraphColoringFormula 

def print_graph(graph, path, filename):
    graph_to_print = nx.Graph()
    graph_to_print.add_nodes_from(sorted(graph.nodes()))
    graph_to_print.add_edges_from(graph.edges)
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    fout = open(f'{path}/{filename}', "w")
    fout.write(f'N\t{graph_to_print.number_of_nodes()}\nM\t{graph_to_print.number_of_edges()}\n')
    for i, j in graph_to_print.edges():
        fout.write(f'e\t{i+1}\t{j+1}\n')
    fout.close()

def create_dataset(seed, ngraphs_each, Ns, cs, path, Q, to_cnf=False):
    for N in Ns:
        for c in cs:        
            random.seed(seed)
            np.random.seed(seed)
            for id in range(ngraphs_each):
                m = int(round(c * N / 2))
                g = nx.gnm_random_graph(N, m)
                filename = f'ErdosRenyi_N_{N}_M_{m}_id_{id+1}.txt'
                print_graph(g, path, filename)
                if to_cnf:
                    F = GraphColoringFormula(g, Q)
                    dimacs = F.to_dimacs()
                    outname = f"COLSAT_N{N}_M{m}_id{id+1}.cnf"
                    final_path = os.path.normpath(os.path.join(path, outname))
                    outfile = open(final_path, "w+")
                    print(dimacs, file=outfile)
                    outfile.close()
                    
            print(f'  N={N}   c={c:.2f}   done')


if __name__ == "__main__":
    ## ArgParsing
    parser = argparse.ArgumentParser(
        description="Create datasets for graph coloring problems (3-COL and 5-COL) on Erdos-Renyi random graphs. The datasets will be created in the folder '3COL' and '5COL' respectively, with subfolders 'train', 'test' and optionally 'test_ood'."
    )
    parser.add_argument('--to-cnf', action='store_true', help='If set, write also the SAT reductions for coloring in CNF format')
    parser.add_argument('--gen-ood', action='store_true', help='If set, generate out-of-distribution test dataset with larger graphs')
    args = parser.parse_args()
    to_cnf = args.to_cnf
    gen_ood = args.gen_ood
    
    print("Creating 3-COL train dataset...")
    Q = 3
    seed = 2
    ngraphs_each = 1200
    Ns = [2**exp for exp in range(4, 9)]
    cs = np.arange(3.32, 4.95, 0.18)
    path = '3COL/train/'
    create_dataset(seed, ngraphs_each, Ns, cs, path, Q, to_cnf)
    
    print("Creating 3-COL test dataset...")
    Q = 3
    seed = 1
    ngraphs_each = 400
    Ns = [2**exp for exp in range(4, 9)]
    cs = np.arange(3.32, 4.95, 0.18)
    path = '3COL/test/'
    create_dataset(seed, ngraphs_each, Ns, cs, path, Q, to_cnf)

    if gen_ood:
        print("Creating 3-COL test-ood dataset...")
        Q = 3
        seed = 1
        ngraphs_each = 400
        Ns = [2**exp for exp in range(9, 15)]
        cs = np.arange(3.32, 4.95, 0.18)
        path = '3COL/test_ood/'
        create_dataset(seed, ngraphs_each, Ns, cs, path, Q, to_cnf)
    
    print("Creating 5-COL train dataset...")
    Q = 5
    seed = 2
    ngraphs_each = 1200
    Ns = [2**exp for exp in range(4, 9)]
    cs = np.arange(9.9, 13.51, 0.4)
    path = '5COL/train/'
    create_dataset(seed, ngraphs_each, Ns, cs, path, Q, to_cnf)

    print("Creating 5-COL test dataset...")
    Q = 5
    seed = 1
    ngraphs_each = 400
    Ns = [2**exp for exp in range(4, 9)]
    cs = np.arange(9.9, 13.51, 0.4)
    path = '5COL/test/'
    create_dataset(seed, ngraphs_each, Ns, cs, path, Q, to_cnf)

    if gen_ood:
        print("Creating 5-COL test-ood dataset...")
        Q = 5
        seed = 1
        ngraphs_each = 400
        Ns = [2**exp for exp in range(9, 15)]
        cs = np.arange(9.9, 13.51, 0.4)
        path = '5COL/test_ood/'
        create_dataset(seed, ngraphs_each, Ns, cs, path)
