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
import sys
from scipy.special import comb
import numpy.random as rnd

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

file_graph = sys.argv[1]
file_solution = sys.argv[2]
q = int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

rnd.seed(seed)

Ncol, Mcol, clauses_compl, clauses_col = read_graph_dimacs(file_graph, q)
solution = read_solution(file_solution)
e = energy(solution, clauses_compl, clauses_col)
print(Ncol, Mcol, e)
