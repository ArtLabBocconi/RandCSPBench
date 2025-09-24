import argparse
import os
import gc
import sys
import torch 
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_scatter import scatter_sum

sys.path.append(os.path.abspath("./satbench"))
from models.gnn_pl import GNNPL
from data.dataloader import get_dataloader


def main(args):
    # Save current settings to avoid args being overwritten
    supervised_eval = args.supervised_eval
    ood_largeN_eval = args.ood_largeN_eval
    n_iterations = args.n_iterations
    scaling_factor = args.scaling_factor
    ood_largeN_eval = args.ood_largeN_eval
    batch_size = args.batch_size
    num_workers = args.num_workers
    full_test =  args.full_test
    global_gpu = args.gpu
    K = args.K
    train_dir = os.path.join(args.dataset_root, f'{K}SAT', 'sc', 'train-final')
    train_label_file = os.path.join(args.dataset_root, f'{K}SAT', 'sc', 'train_labels-final.csv')
    valid_dir = os.path.join(args.dataset_root, f'{K}SAT', 'sc', 'test-final')
    valid_label_file = os.path.join(args.dataset_root, f'{K}SAT', 'sc', 'test_labels-final.csv')
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_file)
    K_ckpt = int(args.ckpt_file.split('_')[0][0])

    # Assertions 
    assert os.path.exists(ckpt_path), f'Checkpoint file {ckpt_path} does not exist!'
    assert K == K_ckpt, f'Checkpoint file {ckpt_path} does not match the specified K value {K}!'
    if args.K == 4:
        assert args.K == 4 and not supervised_eval, 'Only unsupervised models are available for 4-SAT evaluation'

    # Set up scaling factor for the number of iterations
    if scaling_factor > 0:
        scale_model = True
    else:
        scale_model = False
    
    print('Running evaluation procedure with the following settings:')
    print(f' - K: {K}')
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

    # Load model
    model = GNNPL.load_from_checkpoint(ckpt_path)

    # Redefine parameters, name them opts and use them for the rest of the code. Need to understand dataset issues
    opts = model.hparams['args']
    opts.batch_size = batch_size
    opts.num_workers = num_workers
    opts.full_test = full_test
    opts.ood_largeN_eval = ood_largeN_eval
    opts.gpu = global_gpu
    opts.valid_dir = valid_dir
    opts.train_dir = train_dir
    opts.valid_label_file = valid_label_file
    opts.train_label_file = train_label_file    

    if ood_largeN_eval:
        valid_dir = opts.valid_dir.split('/')[:-1] + ['test-ood']
        valid_dir = '/'.join(valid_dir)

    val_loader = get_dataloader(
        valid_dir, 
        opts.valid_splits, 
        opts.valid_sample_size, 
        opts, 
        'valid',
        ns=None
    )        

    if not ood_largeN_eval:
        test_labels = pd.read_csv(opts.valid_label_file)
        ids = [int(fn.split('id')[-1].split('.')[0]) for fn in test_labels['cnf_file'].values]
    else:
        ids = [int(filename.split('id')[-1].split('.')[0]) for filename in val_loader.dataset.all_files['unknown']]
    
    print('Testloader created', len(val_loader))
    print(f'Starting evaluation procedure for {ckpt_path}...')
    model = model.to(torch.device(f'cuda:{global_gpu}'))
    model.eval()
    with torch.no_grad():
        # Create results df
        df = [['N', 'M', 'id', 'Efinale']]
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            batch = batch.to(model.device)
            batch_size = batch.num_graphs

            # Get N and M
            Ns, Ms = batch.l_size // 2, batch.c_size

            # If scaling factor is set, scale the number of iterations based on the number of variables
            if not scale_model:
                model.model.opts.n_iterations = n_iterations
            else:
                model.model.opts.n_iterations = int(scaling_factor * Ns.min().item())

            # Get assignment
            v_pred = model(batch)
            v_pred = v_pred.detach() # detach to make sure it's not tracked by autograd

            # check if each clause is satisfied for the current problem given the predicted assignment
            total_clauses = Ms.sum().item()
            c_batch_indicator = batch.c_batch
            v_assign = (v_pred > 0.5).float()
            l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
            c_sat = torch.clamp(scatter_sum(l_assign[batch.l_edge_index], batch.c_edge_index, dim=0, dim_size=total_clauses), max=1) 

            # calculate solution energies (total number of unsatisfied clauses)
            c_unsat = 1 - c_sat 
            energies = scatter_sum(c_unsat, c_batch_indicator, dim=0, dim_size=batch_size)

            # Append results
            ids_batch = ids[i*batch_size:(i+1)*batch_size]
            append_material = np.stack([Ns.detach().cpu().numpy(), Ms.detach().cpu().numpy(), \
                                        ids_batch, energies.detach().cpu().numpy()]).astype(int).T
            df.append(append_material)

            # Empty cache and manually perform gc for any unexpected memory clogging
            if i % 10:
                gc.collect()
                torch.cuda.empty_cache()

        df = pd.DataFrame(np.vstack(df[1:]), columns=df[0])
        df['alpha'] = df['M'] / df['N']
        df['Solved'] = (df['Efinale'] == 0).astype(int)

        if not ood_largeN_eval:
            save_name = f'{K}SAT_results_{args.ckpt_file}_niters{model.model.opts.n_iterations}_Nscale{scaling_factor}.csv'
        else:
            save_name = f'{K}SAT_results_{args.ckpt_file}_niters{model.model.opts.n_iterations}_Nscale{scaling_factor}_OODN.csv'

    df.to_csv(os.path.join(args.save_dir, save_name), index=False)
    print('Evaluation complete.')
    print('Avg. solving probability:', df["Solved"].mean())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str, required=True, help='PL checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, required=True, default='ckpt', help='Directory where the checkpoints are stored')
    parser.add_argument('--dataset_root', type=str, default='../../datasets', help='Directory where the benchmarks are stored')
    parser.add_argument('--save_dir', type=str, default='results_csv/', help='Directory where to save the evaluation results')
    parser.add_argument('--n_iterations', type=int, default=32, help='Number of iterations for message passing (not considered if scaling_factor is specified)')
    parser.add_argument('--scaling_factor', type=float, default=2., help='Factor that scales the number of msg passing iterations based on the number of variables')
    parser.add_argument('--supervised_eval', action='store_true', help='perform the evaluation procedure using the supervised models. If false, uses the unsupervised ones (False by default)')
    parser.add_argument('--ood_largeN_eval', action='store_true', help='perform the evaluation procedure on larger problems on which the model has never been trained on (False by default)')
    parser.add_argument('--K', type=int, default=3, help='Number of variables in a clause of the SAT problem')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for the evaluation procedure')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the dataloader')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use for the evaluation procedure')
    parser.add_argument('--full_test', action='store_false', help='use all of the available testing samples (SAT and UNSAT) (True by default)')

    args = parser.parse_args()
    main(args)