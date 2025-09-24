import os
import sys
import argparse
import lightning as L

from datetime import datetime
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

sys.path.append(os.path.abspath('./satbench'))
from models.gnn_pl import GNNPL
from utils.options import add_model_options
from data.dataloader import get_dataloader

def main(args):
    # set seed
    L.seed_everything(args.seed)
    args.full_test = False # To evaluate accuracy during training we can just use SAT instances in the validation set

    # create dataloaders
    Ns = [16, 32, 64, 128, 256]
    if args.use_all_Ns:
        train_Ns = Ns
        test_Ns = Ns
        n_savename = 'all'
    else:
        train_Ns = args.train_Ns
        test_Ns = list(set(Ns) - set(train_Ns))
        n_savename = ';'.join(map(str, train_Ns))

    train_loader = get_dataloader(
        args.train_dir, 
        args.train_splits, 
        args.train_sample_size, 
        args,
        'train',
        ns=train_Ns
    )  
    val_loader = None
    if args.valid_dir is not None:
        val_loader = get_dataloader(
            args.valid_dir, 
            args.valid_splits, 
            args.valid_sample_size, 
            args, 
            'valid',
            ns=test_Ns
    )        
    print('Data loaders created. Len train:', len(train_loader), 'Len valid:', len(val_loader))
    
    # Define model (lightning version)
    model = GNNPL(args)

    # define callbacks and logger
    logger = None
    dt_string = datetime.now().strftime('%d-%m-%Y-%H-%M')
    K = int(args.train_dir.split('/')[3][0])
    run_name = f'{K}SAT_{args.task}_{args.model}_{args.loss}_seed={args.seed}_trainS={args.train_sample_size}_validS={args.valid_sample_size}_perN={args.sample_per_N}_trainNs={n_savename}_{dt_string}'
    
    if args.resume_from:
        run_name += '_continuation_from_' + args.resume_from.split('_')[-2] # gets the datetime string from the checkpoint saved with the suffix '_last'

    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            reinit=True,
            entity=args.wandb_entity
        )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='acc_test', 
        mode='max',
        save_top_k=2,
        filename=run_name + '_{epoch}',
        dirpath='ckpt/',
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f'{run_name}_last'

    # Define lightning trainer object
    if args.resume_from:
        num_epochs = args.epochs * 10
    else:
        num_epochs = args.epochs

    trainer = Trainer(
        max_epochs=num_epochs, 
        devices=args.gpu,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps, 
        callbacks=[checkpoint_callback],
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=args.clip_norm
    )

    # Fit (and evaluate) model
    if args.resume_from:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument('task', type=str, choices=['assignment'], help='Solving task')
    parser.add_argument('train_dir', type=str, help='Directory with training data')
    parser.add_argument('--train_splits', type=str, nargs='+', choices=['unknown'], default=None, help='Category of the training data')
    parser.add_argument('--train_sample_size', type=int, default=None, help='The number of instance in each training splits')
    parser.add_argument('--valid_dir', type=str, default=None, help='Directory with validating data')
    parser.add_argument('--valid_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat','unknown'], default=None, help='Category of the validating data')
    parser.add_argument('--valid_sample_size', type=int, default=None, help='The number of instance in each validating splits')
    parser.add_argument('--label', type=str, choices=[None, 'satisfiability', 'assignment', 'core_variable'], default=None, help='Label')
    parser.add_argument('--data_fetching', type=str, choices=['parallel', 'sequential'], default='parallel', help='Fetch data in sequential order or in parallel')
    parser.add_argument('--loss', type=str, choices=['supervised', 'unsupervised_2'], default=None, help='Loss type for assignment prediction')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='L2 regularization weight')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler')
    parser.add_argument('--lr_step_size', type=int, default=50, help='Learning rate step size')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Learning rate factor')
    parser.add_argument('--lr_patience', type=int, default=10, help='Learning rate patience')
    parser.add_argument('--clip_norm', type=float, default=0.8, help='Clipping norm')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', type=int, nargs='+', default=0, help='GPU index(s)')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run only a few steps for testing.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint from which to resume training from.')
    parser.add_argument('--train_label_file', type=str, default=None, help='csv file with assignments for the training data')
    parser.add_argument('--valid_label_file', type=str, default=None, help='csv file with assignments for the validating data')
    parser.add_argument('--sample_per_N', action='store_false', help='sample a sample_size per each N (#variables) or not. (true by default)')
    parser.add_argument('--train_Ns', type=int, nargs='+', default=[16,32,64,128], help='List of Ns to train on (useful for e.g., OOD testing') 
    parser.add_argument('--use_all_Ns', action='store_false', help='Overrides the above and uses all availables Ns (True by default)')
    parser.add_argument('--ood_largeN_eval', action='store_true', help='perform the evaluation procedure on larger problems on which the model has never been trained on (False by default)')

    # Model-specific args
    add_model_options(parser)

    # Wandb (logging) args
    parser.add_argument('--wandb', action='store_true', help='Log to wandb. (true by default)')
    parser.add_argument('--wandb_project', type=str, default='kSAT-bench', help='Wandb project name.')
    parser.add_argument('--wandb_entity', type=str, default='gskenderi', help='Wandb entity name.')
    parser.add_argument('--log_every_n_steps', type=int, default=1, help='Log training stats every N steps.')

    args = parser.parse_args()
    main(args)
