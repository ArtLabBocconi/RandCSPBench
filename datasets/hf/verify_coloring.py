"""Check that regenerated q-coloring instances match those evaluated in the paper.

Coverage: the (N, M, id) keys of test + test_ood must equal those in
results_evaluation/analysis-results/{q}COL-best_per_sample.csv.

Fingerprint: small test instances are solved exactly. Any instance marked as
solved in the paper must be colorable; otherwise the graphs differ.
"""
import argparse
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

import pandas as pd
from pysat.formula import CNF
from pysat.solvers import Solver

NAME_RE = re.compile(r'ErdosRenyi_N_(\d+)_M_(\d+)_id_(\d+)\.txt$')


def keys(folder):
    return {tuple(map(int, NAME_RE.match(f).groups())) for f in os.listdir(folder) if NAME_RE.match(f)}


def colorable(cnf_file):
    with Solver(name='cadical195', bootstrap_with=CNF(from_file=cnf_file).clauses) as s:
        return s.solve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('q', type=int, choices=[3, 5])
    parser.add_argument('--root', default=os.path.join(os.path.dirname(__file__), '..'))
    parser.add_argument('--results', default=os.path.join(os.path.dirname(__file__), '..', '..', 'results_evaluation', 'analysis-results'))
    parser.add_argument('--nmax', type=int, default=64, help='exactly solve test instances with N <= nmax')
    args = parser.parse_args()

    base = os.path.join(args.root, f'{args.q}COL')
    ref = pd.read_csv(os.path.join(args.results, f'{args.q}COL-best_per_sample.csv'))
    ref_keys = set(zip(ref.N, ref.M, ref.id))

    ok = True
    train, test, ood = (keys(os.path.join(base, s)) for s in ('train', 'test', 'test_ood'))
    print(f'{args.q}COL: train={len(train)} test={len(test)} test_ood={len(ood)} reference={len(ref_keys)}')
    gen_keys = test | ood
    if gen_keys != ref_keys:
        ok = False
        print(f'  coverage MISMATCH: {len(gen_keys - ref_keys)} extra, {len(ref_keys - gen_keys)} missing')
    else:
        print('  coverage OK')

    small = ref[ref.N <= args.nmax]
    files = [os.path.join(base, 'test-sat', f'COLSAT_N{N}_M{M}_id{i}.cnf') for N, M, i in zip(small.N, small.M, small.id)]
    with Pool() as pool:
        sat = pool.map(colorable, files, chunksize=50)

    tally = Counter(zip(small.N, small.solved, sat))
    for N in sorted(small.N.unique()):
        print(f'  N={N:3d}: solved&SAT={tally[N, 1, True]} solved&UNSAT={tally[N, 1, False]} '
              f'unsolved&SAT={tally[N, 0, True]} unsolved&UNSAT={tally[N, 0, False]}')
    bad = sum(tally[N, 1, False] for N in small.N.unique())
    if bad:
        ok = False
        print(f'  fingerprint MISMATCH: {bad} instances solved in the paper are not colorable')
    else:
        print('  fingerprint OK')

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
