import argparse
import os
import gc
import sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch_scatter import scatter_sum

sys.path.append(os.path.abspath('./satbench'))
from models.gnn_pl import GNNPL
from data.dataloader import get_dataloader
from satbench.utils.utils import safe_log

def _append_rows(csv_path, df_batch):
    temp = Path(csv_path)
    write_header = not temp.exists() or temp.stat().st_size == 0
    df_batch.to_csv(csv_path, mode='a', index=False, header=write_header)

def main(args):
    # Save current settings locally to avoid overwrite
    supervised_eval = args.supervised_eval
    ood_largeN_eval = args.ood_largeN_eval
    n_iterations = args.n_iterations
    scaling_factor = args.scaling_factor
    batch_size = args.batch_size
    num_workers = args.num_workers
    full_test = args.full_test
    global_gpu = args.gpu
    K = args.K
    task = args.task.upper()

    # Dataset paths
    if task == 'SAT':
        train_dir = os.path.join(args.dataset_root, f'{K}SAT', 'sc', 'train-final')
        valid_dir = os.path.join(args.dataset_root, f'{K}SAT', 'sc', 'test-final')
        valid_label_file = os.path.join(args.dataset_root, f'{K}SAT', 'sc', 'test_labels-final.csv')
    else:
        train_dir = os.path.join(args.dataset_root, f'{K}COL', 'train-sat')
        valid_dir = os.path.join(args.dataset_root, f'{K}COL', 'test-sat')
        valid_label_file = os.path.join(args.dataset_root, f'{K}COL', f'{K}COL-test-labels.csv')

    # Checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_file)
    K_ckpt = int(args.ckpt_file.split('_')[0][0])

    # Load model
    assert os.path.exists(ckpt_path), f'Checkpoint file {ckpt_path} does not exist!'
    model = GNNPL.load_from_checkpoint(ckpt_path)

    # Assertions
    assert task in ['SAT', 'COL'], f'Task must be SAT or COL, got {task}'
    assert K == K_ckpt, f'Checkpoint file {ckpt_path} does not match the specified K value {K}!'
    if args.K == 4:
        assert args.K == 4 and not supervised_eval, 'Only unsupervised models are available for 4-SAT evaluation'

    # Scaling on/off
    scale_model = scaling_factor > 0
    fixed_iter_flag = n_iterations > 0
    assert scale_model != fixed_iter_flag, 'Either set scaling_factor > 0 to enable scaling, or n_iterations > 0 for fixed iterations'
    
    # Resuming evaluation
    if args.resume_from:
        assert args.save_every > 0, 'Resuming from a CSV file requires save_every > 0 to periodically flush results to disk.'

    print('Running evaluation procedure with the following settings:')
    print(f' - K: {K}')
    print(f' - Task: {task}')
    print(f' - Supervised evaluation: {supervised_eval}')
    print(f' - OOD large N evaluation: {ood_largeN_eval}')
    if scale_model:
        print(f' - Scaling factor: {scaling_factor}')
    else:
        print(f' - Number of iterations: {n_iterations}')
    print(f' - Batch size: {batch_size}')
    print(f' - Number of workers: {num_workers}')
    print(f' - Full test: {full_test}')
    print(f' - GPU: {global_gpu}')
    print(f' - Resume from: {args.resume_from}')
    print(f' - Save every: {args.save_every} batches')

    # Print gpu information if it is used
    if global_gpu >= 0:
        print('Using GPU:', global_gpu)
        print('GPU name:', torch.cuda.get_device_name(global_gpu))
    else:
        print('Using CPU for evaluation.')


    # Redefine parameters; use opts for rest of code
    opts = model.hparams['args']
    opts.batch_size = batch_size
    opts.num_workers = num_workers
    opts.full_test = full_test
    opts.ood_largeN_eval = ood_largeN_eval
    opts.gpu = global_gpu
    opts.train_dir = train_dir
    opts.valid_dir = valid_dir
    opts.valid_label_file = valid_label_file

    if ood_largeN_eval:
        valid_dir = '/'.join(opts.valid_dir.split('/')[:-1] + ['test-ood'])
        opts.valid_dir = valid_dir

    # Dataloader (Do not shuffle to correctly resume by index!)
    val_loader = get_dataloader(
        opts.valid_dir,
        opts.valid_splits,
        opts.valid_sample_size,
        opts,
        'valid',
        ns=None,
    )

    print('Testloader created', len(val_loader))
    print(f'Starting evaluation procedure for {ckpt_path}...')

    # Device
    if global_gpu < 0:
        model = model.to(torch.device('cpu'))
    else:
        model = model.to(torch.device(f'cuda:{global_gpu}'))
    model.eval()

    # Build problem ID list
    # make ids a concat between N, M, and sample id: this is a unique identifier
    ids = [filename.split('N')[-1].split('_')[0] + '-' + 
           filename.split('M')[-1].split('_')[0] +  '-' + 
           filename.split('id')[-1].split('.')[0]
           for filename in val_loader.dataset.all_files['unknown']]

    print(f'Evaluation dataset size before filtering: {len(val_loader.dataset)} samples.')
    print(f'Evaluation dataloader batches before filtering: {len(val_loader)} batches.')
    c_test = 3.68 if K==3 else 11.1
    N_test = 1024
    M_test = int(round(c_test * N_test / 2))
    ids = [id_str for id_str in ids if int(id_str.split('-')[0]) == N_test]
    ids = [id_str for id_str in ids if int(id_str.split('-')[1]) == M_test]
    ids = np.array(ids)
    val_loader.dataset.filter_ids(ids)
    print(f'Evaluation dataset size after filtering: {len(val_loader.dataset)} samples.')
    print(f'Evaluation dataloader batches after filtering: {len(val_loader)} batches.')

    processed_ids = None
    if args.resume_from:
        results_csv = os.path.normpath(args.resume_from)
        results_df = pd.read_csv(results_csv)
        processed_ids = results_df['id'].values
        print(f'[resuming...] Loaded {len(processed_ids)} processed ids from {results_csv}')
    else:
        if not ood_largeN_eval:
            save_name = f'{K}{task}_results_{args.ckpt_file}_niters{n_iterations}_Nscale{scaling_factor}'
        else:
            save_name = f'{K}{task}_results_{args.ckpt_file}_niters{n_iterations}_Nscale{scaling_factor}_OODN'
        results_csv = os.path.normpath(os.path.join(args.save_dir, 'FINAL_SPEEDTEST' + save_name + '.csv'))
        print(f'Results will be saved to: {results_csv}...')

    # Safeguard for save directory 
    os.makedirs(args.save_dir, exist_ok=True)

    # Column layout for streamed writes
    base_cols = ['N', 'M', 'id', 'E']
    if task == 'COL':
        base_cols.append('assignment')
    base_cols += ['alpha', 'Solved']

    batch_buffer = []
    save_every = max(1, int(args.save_every))


    start_time = time.time()

    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        # Determine the ids for this batch BEFORE moving to device
        bsz_tmp = batch.num_graphs
        ids_batch = ids[i * bsz_tmp : (i + 1) * bsz_tmp]

        # When resuming from saved results, skip the batch if it was already processed
        if args.resume_from:
            ids_check = processed_ids[i * bsz_tmp : (i + 1) * bsz_tmp]
            if len(ids_batch) == len(ids_check):
                if np.all(ids_batch == ids_check):
                    continue
                if np.any(ids_batch != ids_check):
                    print('check if this happens')
        
        # Move to device
        batch = batch.to(model.device)
        batch_size = batch.num_graphs
        c_size = batch.c_size.sum().item()
        c_batch = batch.c_batch
        l_edge_index = batch.l_edge_index
        c_edge_index = batch.c_edge_index

        # Get N and M
        Ns, Ms = batch.l_size // 2, batch.c_size

        # Set message-passing iterations (fixed or scaled)
        if not scale_model:
            model.model.opts.n_iterations = n_iterations
        else:
            model.model.opts.n_iterations = int(scaling_factor * Ns.min().item())

        # Forward
        v_pred = model(batch)

        # Compute per-mode losses to select best assignment
        losses = []
        for j in range(v_pred.shape[1]):
            mode_pred = v_pred[:, j]
            l_pred = torch.stack([mode_pred, 1 - mode_pred], dim=1).reshape(-1)
            l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
            c_loss = -safe_log(1 - l_pred_aggr.exp())
            loss = scatter_sum(c_loss, c_batch, dim=0)
            assert batch_size == loss.shape[-1], 'the loss must be calculated for each element of the batch'
            losses.append(loss)

        losses = torch.stack(losses, dim=0)
        best_pred_idx = torch.argmin(losses, dim=0)

        # Select best assignments
        batch_sliced = v_pred.reshape(batch_size, -1, v_pred.size(1))
        broadcasted_idx = best_pred_idx.long().view(batch_size, 1, 1).expand(-1, batch_sliced.size(1), 1)
        final_v_pred = batch_sliced.gather(dim=2, index=broadcasted_idx).squeeze(2).view(-1)

        # Compute energy
        v_assign = (final_v_pred >= 0.5).float()
        l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
        c_unsat = 1 - c_sat
        energies = scatter_sum(c_unsat, c_batch, dim=0, dim_size=batch_size)
        
        if ((i + 1) % save_every == 0) and batch_buffer:
            _append_rows(results_csv, pd.concat(batch_buffer, ignore_index=True))
            batch_buffer.clear()

        if ((i + 1) % 5) == 0:
            gc.collect()
            if global_gpu >= 0:
                torch.cuda.empty_cache()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str, required=True, help='PL checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, required=True, default='ckpt', help='Directory where the checkpoints are stored')
    parser.add_argument('--dataset_root', type=str, default='../../datasets', help='Directory where the benchmarks are stored')
    parser.add_argument('--save_dir', type=str, default='results_csv/', help='Directory where to save the evaluation results')
    parser.add_argument('--n_iterations', type=int, default=-1, help='Number of iterations for message passing (ignored if scaling_factor > 0)')
    parser.add_argument('--scaling_factor', type=float, default=1.0, help='Factor that scales iterations based on number of variables; set <=0 to disable scaling')
    parser.add_argument('--supervised_eval', action='store_true', help='Use supervised models; if False, uses unsupervised ones (default False)')
    parser.add_argument('--ood_largeN_eval', action='store_true', help='Evaluate on larger, OOD problems (default False)')
    parser.add_argument('--K', type=int, default=3, help='Number of variables in a clause of the SAT problem')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use; set <0 for CPU')
    parser.add_argument('--full_test', action='store_false', help='Use all available testing samples (SAT and UNSAT) (True by default)')
    parser.add_argument('--task', type=str, default='sat', choices=['sat', 'col'], help='Task to evaluate: SAT solving or coloring (default: sat)')
    
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from existing CSV file by skipping already evaluated problems (default None)')
    parser.add_argument('--save_every', type=int, default=5, help='Flush results to disk every N batches (default 5)')

    args = parser.parse_args()
    main(args)
