"""Stage the q-coloring datasets for the Hugging Face dataset repo.

Writes into --out (mirroring the HF repo layout):
  coloring/{q}COL.zip                       raw edge lists (+ CNF reductions for train/test)
  coloring/{q}COL/{split}-*.parquet         one row per graph, for `datasets.load_dataset`
"""
import argparse
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from datasets import Dataset, Features, List, Value

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
RESULTS = os.path.normpath(os.path.join(HERE, '..', '..', 'results_evaluation', 'analysis-results'))

SPLITS = ['train', 'test', 'test_ood']
CNF_SPLITS = ['train', 'test']
C_GRID = {3: np.arange(3.32, 4.95, 0.18), 5: np.arange(9.9, 13.51, 0.4)}
NAME_RE = re.compile(r'ErdosRenyi_N_(\d+)_M_(\d+)_id_(\d+)\.txt$')
MAX_SHARD_BYTES = 300 * 2**20
MAX_ROW_GROUP_BYTES = 64 * 2**20

FEATURES = Features({
    'name': Value('string'),
    'q': Value('int8'),
    'N': Value('int32'),
    'M': Value('int32'),
    'c': Value('float64'),
    'id': Value('int32'),
    'edges': List(List(Value('int32'), length=2)),
    'solved': Value('bool'),
})


def nominal_c(q, N, M):
    # the generator sets M = round(c N / 2); the grid is coarse enough for this to be invertible
    (c,) = [c for c in C_GRID[q] if int(round(c * N / 2)) == M]
    return round(float(c), 2)


def read_edges(path):
    e = pd.read_csv(path, sep='\t', skiprows=2, header=None, usecols=[1, 2], dtype=np.int32).to_numpy()
    return e - 1


def gen_rows(files, q, folder, solved):
    for f in files:
        N, M, i = map(int, NAME_RE.match(f).groups())
        edges = read_edges(os.path.join(folder, f))
        assert edges.shape == (M, 2), f
        yield {'name': f, 'q': q, 'N': N, 'M': M, 'c': nominal_c(q, N, M), 'id': i,
               'edges': edges.tolist(), 'solved': solved.get((N, M, i))}


def shard_bounds(sizes, cap):
    bounds, start, acc = [], 0, 0
    for i, s in enumerate(sizes):
        if acc and acc + s > cap:
            bounds.append((start, i))
            start, acc = i, 0
        acc += s
    bounds.append((start, len(sizes)))
    return bounds


def build_parquet(q, out, num_proc):
    ref = pd.read_csv(os.path.join(RESULTS, f'{q}COL-best_per_sample.csv'))
    solved = {(N, M, i): bool(s) for N, M, i, s in zip(ref.N, ref.M, ref.id, ref.solved)}
    outdir = os.path.join(out, 'coloring', f'{q}COL')
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    for split in SPLITS:
        folder = os.path.join(ROOT, f'{q}COL', split)
        files = sorted((f for f in os.listdir(folder) if NAME_RE.match(f)),
                       key=lambda f: tuple(map(int, NAME_RE.match(f).groups())))
        ds = Dataset.from_generator(
            gen_rows, features=FEATURES, num_proc=num_proc,
            gen_kwargs={'files': files, 'q': q, 'folder': folder,
                        'solved': solved if split != 'train' else {}},
        )
        order = list(zip(ds['N'], ds['M'], ds['id']))
        assert order == sorted(order), 'shards were concatenated out of order'
        sizes = 8 * np.asarray(ds['M'], dtype=np.int64) + 64  # row sizes are dominated by the edges
        bounds = shard_bounds(sizes, MAX_SHARD_BYTES)
        for k, (a, b) in enumerate(bounds):
            path = os.path.join(outdir, f'{split}-{k:05d}-of-{len(bounds):05d}.parquet')
            row_group = max(1, int(MAX_ROW_GROUP_BYTES // sizes[a:b].max()))
            ds.select(range(a, b)).to_parquet(path, batch_size=row_group)
        print(f'{q}COL/{split}: {len(ds)} rows, {sizes.sum() / 2**20:.0f} MiB, {len(bounds)} shards', flush=True)


def build_zip(q, out):
    os.makedirs(os.path.join(out, 'coloring'), exist_ok=True)
    zpath = os.path.abspath(os.path.join(out, 'coloring', f'{q}COL.zip'))
    if os.path.exists(zpath):
        os.remove(zpath)
    members = [f'{q}COL/{s}' for s in SPLITS] + [f'{q}COL/{s}-sat' for s in CNF_SPLITS]
    subprocess.run(['zip', '-q', '-r', '-X', zpath, *members], cwd=ROOT, check=True)
    print(f'{zpath}: {os.path.getsize(zpath) / 2**30:.2f} GiB', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('what', choices=['parquet', 'zip'])
    parser.add_argument('--q', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--out', default=os.path.join(HERE, 'staging'))
    parser.add_argument('--num-proc', type=int, default=16)
    args = parser.parse_args()
    for q in args.q:
        if args.what == 'parquet':
            build_parquet(q, args.out, args.num_proc)
        else:
            build_zip(q, args.out)


if __name__ == '__main__':
    main()
