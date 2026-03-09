import os
import sys
import zipfile
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy.random as rnd
from scipy.special import comb

def read_graph_dimacs(file, q):
    with open(file, 'r') as f:
        lines = f.readlines()
    clauses_compl = []
    clauses_col = []
    header = lines[0].split()
    Nsat = int(header[2])
    Ncol = Nsat // q
    Mcompl = Ncol * (1 + comb(q, 2, exact=True))

    # We have three types of clauses:
    # 1) At least one color per node (AOL) - Ncol clauses
    # 2) At most one color per node (NTC) - Ncol + Ncol*(q choose 2) clauses
    # 3) No same color for adjacent nodes (NSC)
    AOL_lines, NTC_lines, NSC_lines = lines[1:Ncol+1], lines[Ncol+1:Mcompl+1], lines[Mcompl+1:-1]
    j = 0
    for line in AOL_lines:
        clause = list(map(int, line.split()))
        clause.pop()
        clauses_compl.append(clause)
        for i in range(j, j + comb(q, 2, exact=True)):
            ntc_clause = NTC_lines[i]
            ntc_clause = list(map(int, ntc_clause.split()))
            ntc_clause.pop()
            clauses_compl.append(ntc_clause)
        j += comb(q, 2, exact=True)
    for line in NSC_lines:
        clause = list(map(int, line.split()))
        clause.pop()
        clauses_col.append(clause)
    
    Mcol = len(clauses_col) // q
    f.close()

    return Ncol, Mcol, clauses_compl, clauses_col

def read_solution(bitstring, numerate=True):
    bitstring = bitstring[1:-1]
    solution = list(map(float, bitstring.split(', ')))
    if numerate:
        tol = 1e-5
        sol = [(idx+1) if y >= 1-tol else -(idx+1) for idx, y in enumerate(solution)]
    else:
        sol = [int(y) for y in solution]

    return sol

def sanity_check_sat(problem, solution):
    pass

def eval_clause(clause, solution):
    for i in range(len(clause)):
        if clause[i] == solution[abs(clause[i]) - 1]:
            return True
    return False


def check_clauses(solution, clauses):
    for clause in clauses:
        if not eval_clause(clause, solution):
            return False
    return True


def energy(solution, clauses_compl, clauses_col):
    for clause in clauses_compl:
        if not eval_clause(clause, solution):
            sel = rnd.choice(clause)
            solution[abs(sel) - 1] = sel
    e = 0
    for clause in clauses_col:
        if not eval_clause(clause, solution):
            e += 1    
    return e


def main(args):
    filepath = os.path.normpath(os.path.join(args.csv_dir, args.csv_result_name))
    print('Computing coloring solution energy for corresponding SAT reductions in', filepath)
    results_df = pd.read_csv(filepath, sep=';')
    valid_data_dir = os.path.join(args.cnf_dir, f'{args.q}COL', 'test-sat')

    # rnd.seed(args.seed)

    sat2col_res_df = [['N', 'M', 'file_id', 'E']]
    counter = 0
    for _, x in tqdm(results_df.iterrows(), total=len(results_df), desc="Evaluating SAT to COL solutions..."):
        N, M, idx, _, assignment, _, Solved = x
        filname_info = idx.split('-')
        filename_colN, filename_colM, filename_idx = int(filname_info[0]), int(filname_info[1]), int(filname_info[2])
        col_N = N // args.q
        col_M = (M - col_N * (1 + (args.q * (args.q - 1)) // 2)) // args.q
        assert filename_colN == col_N, "N in filename and calculation must match!"
        assert filename_colM == col_M, "M in filename and calculation must match!"

        cnf_filename = f'COLSAT_N{filename_colN}_M{filename_colM}_id{filename_idx}.cnf'
        cnf_filepath = os.path.join(valid_data_dir, cnf_filename)

        # read dimacs
        Ncol, Mcol, clauses_compl, clauses_col = read_graph_dimacs(cnf_filepath, args.q)
        assert Ncol == col_N and Mcol == col_M, "Graph dimensions do not match!"

        # calculate energy in col solution space
        solution = read_solution(assignment)
        if len(solution) != N:
            counter += 1
            continue
        
        e = energy(solution, clauses_compl, clauses_col)
        sat2col_res_df.append([Ncol, Mcol, filename_idx, e])
    
    print(counter)
    df = pd.DataFrame(np.vstack(sat2col_res_df[1:]), columns=sat2col_res_df[0])
    df['c'] = (df['M'] / df['N']) * 2
    df['Solved'] = (df['E'] == 0).astype(int)
    print('Evaluation complete.')
    print('Avg. solving probability:', df["Solved"].mean())
    print()
    df.to_csv(os.path.join(args.csv_dir, 'SAT2COL_'+args.csv_result_name), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='results_csv/', help='Directory where the result files are stored')
    parser.add_argument('--cnf_dir', type=str, default='datasets/', help='Directory where the result files are stored')
    parser.add_argument('--csv_result_name', type=str, required=True, default=None, help='Filename containing the assingments of the model')
    parser.add_argument('--q', type=int, default=3, help='Number of colors (q-Col problem)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    main(args)