import os
import itertools
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from satbench.data.dataset import SATDataset

def collate_fn(batch):
    return Batch.from_data_list([s for s in list(itertools.chain(*batch))])


def get_dataloader(data_dir, splits, sample_size, opts, mode, use_contrastive_learning=False, ns=None):
    assert opts.valid_label_file is not None, 'Missing valid_label_file in opts, must exists in order to validate the model.'
    label_file = opts.train_label_file if mode == 'train' else opts.valid_label_file
    data_dir = os.path.normpath(data_dir)
    label_file = os.path.normpath(label_file)
    dataset = SATDataset(data_dir, splits, sample_size, ns, use_contrastive_learning, opts, label_file=label_file)
    batch_size = opts.batch_size // len(splits) if opts.data_fetching == 'parallel' else opts.batch_size

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode=='train'),
        collate_fn=collate_fn,
        num_workers=opts.num_workers,
    )

def get_ood_testloader(data_dir, splits, sample_size, opts, mode, use_contrastive_learning=False, ns=None):
    assert opts.valid_label_file is not None, 'Missing valid_label_file in opts, must exists in order to validate the model.'
    label_file = opts.valid_label_file
    data_dir = os.path.normpath(data_dir)
    label_file = os.path.normpath(label_file)
    dataset = SATDataset(data_dir, splits, sample_size, ns, use_contrastive_learning, opts, label_file=label_file)
    batch_size = opts.batch_size // len(splits) if opts.data_fetching == 'parallel' else opts.batch_size

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode=='train'),
        collate_fn=collate_fn,
        num_workers=opts.num_workers,
    )
