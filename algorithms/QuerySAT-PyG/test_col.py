"""
This script converts a candidate solution of the SAT problem into a candidate solution of the original q-coloring problem, and computes its energy.

Usage:
    python sat2col_random.py <file_graph> <file_solution> <q> [seed]

Arguments:
    file_graph (str): Path to the graph file in DIMACS format.
    file_solution (str): Path to the solution file.
    q (int): Number of colors.
    seed (int, optional): Seed for the random number generator. If not provided, the default seed is 1.

Returns:
    Ncol (int): Number of nodes in the original q-coloring problem.
    Mcol (int): Number of edges in the original q-coloring problem.
    e (int): Energy of the candidate solution of the original q-coloring problem.
"""

import os
import sys
import argparse
import pandas as pd
import numpy.random as rnd
from scipy.special import comb

def read_graph_dimacs(file, q):
    with open(file, 'r') as f:
        lines = f.readlines()
    clauses_compl = []
    clauses_col = []
    header = lines[1].split()
    Nsat = int(header[2])
    Ncol = Nsat // q
    Mcompl = Ncol * (1 + comb(q, 2, exact=True))
    for line in lines[2:Mcompl + 2]:
        clause = list(map(int, line.split()))
        clause.pop()
        clauses_compl.append(clause)
    for line in lines[Mcompl + 2:]:
        clause = list(map(int, line.split()))
        clause.pop()
        clauses_col.append(clause)
    Mcol = len(clauses_col) // q
    return Ncol, Mcol, clauses_compl, clauses_col


def read_solution(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    solution = []
    for line in lines:
        solution += list(map(int, line.split()))
    solution = sorted(solution, key=abs)
    return solution


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


if len(sys.argv) < 4:
    print(__doc__)
    sys.exit(1)

def main(args):
    filepath = os.path.normpath(os.path.join(args.csv_dir, args.csv_result_name))
    print('Computing coloring solution energy for corresponding SAT reductions in', filepath)
    df = pd.read_csv(filepath)

    file_graph = sys.argv[1]
    file_solution = sys.argv[2]

    rnd.seed(args.seed)

    for (N, M, idx, E, assignment) in df.iterrows():
        cnf_filename = f'N{N}_M{M}_id{idx}.cnf'
        cnf_filepath = os.path.join()
        Ncol, Mcol, clauses_compl, clauses_col = build_graph_from_dimacs(file_graph, args.q)
    solution = read_solution(file_solution)
    e = energy(solution, clauses_compl, clauses_col)
    print(Ncol, Mcol, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True, default='results_csv/', help='Directory where the result files are stored')
    parser.add_argument('--cnf_dir', type=str, required=True, default='datasets/', help='Directory where the result files are stored')
    parser.add_argument('--csv_result_name', type=str, default=None, help='Filename containing the assingments of the model')
    parser.add_argument('--q', type=int, default=3, help='Number of colors (q-Col problem)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    main(args)