import os
import glob
import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Dataset

from satbench.utils.utils import parse_cnf_file, clean_clauses
from satbench.data.data import construct_lcg, construct_vcg


class SATDataset(Dataset):
    def __init__(self, data_dir, splits, sample_size, ns, use_contrastive_learning, opts, files=None, label_file = None):
        self.opts = opts
        self.splits = splits
        self.sample_size = sample_size
        self.data_partition = data_dir.split('/')[-1]
        self.all_files = self._get_files(data_dir, files)

        if not opts.ood_largeN_eval:

            if label_file is None:
                if 'train' in self.data_partition:
                    self.split_len = self._get_split_len()
                    
                self.all_labels = self._get_labels(data_dir)
            else:
                self.all_labels = self._get_labels_from_file(label_file)

        # Remove instances with no assignment (labeled unsat by the solver) if the training is supervised 
        # or if this is the testing set (we can only evaluate accuracy on a fully SAT testing set)
        # Otherwise if full_test is true, then get all the labels
        if ((self.opts.loss == 'supervised' and label_file) or ('test' in self.data_partition)) and self.opts.full_test == False:
            for split in self.splits:
                self.all_files[split] = [cnf_filepath for cnf_filepath, label in zip(self.all_files[split], self.all_labels[split]) if torch.isinf(label).sum().item() == 0]
                self.all_labels[split] = [label for label in self.all_labels[split] if torch.isinf(label).sum().item() == 0]

        # Select only the files with the given ns, to train/test the model on a subset of the Ns 
        if ns is not None:
            for split in self.splits:
                # order of operations is quite important here!
                self.all_labels[split] = [label for cnf_filepath, label in zip(self.all_files[split], self.all_labels[split]) if int(cnf_filepath.split('_')[0].split('N')[1]) in ns] 
                self.all_files[split] = [cnf_filepath for cnf_filepath in self.all_files[split] if int(cnf_filepath.split('_')[0].split('N')[1]) in ns]
               
        self.use_contrastive_learning = use_contrastive_learning
        if self.use_contrastive_learning:
            self.positive_indices = self._get_positive_indices()

        self.split_len = self._get_split_len()

        super().__init__(data_dir)
    
    def _get_files(self, data_dir, files=None):
        if files is None:
            files = {}
            for split in self.splits:
                if split == "unknown":
                    split_files = list(sorted(glob.glob(data_dir + f'/*.cnf', recursive=True)))
                else:
                    split_files = list(sorted(glob.glob(data_dir + f'/{split}/*.cnf', recursive=True)))

                # (optional) data sampling
                if self.sample_size is not None and len(split_files) > self.sample_size:
                    if self.opts.sample_per_N:
                        Ns = [f.split('_')[0].split('N')[1] for f in split_files]
                        split_files = random.sample(split_files, self.sample_size*np.unique(Ns).shape[0])
                    else:
                        split_files = random.sample(split_files, self.sample_size)
                    
                files[split] = split_files
        else:
            assert isinstance(files, dict)
            for split in self.splits:
                split_files = files[split]
                if self.sample_size is not None and len(split_files) > self.sample_size:
                    split_files = random.sample(split_files, self.sample_size)
                split_files = [data_dir + f'/{split}/' + file for file in split_files]
                for cnf_filepath in split_files:
                    assert os.path.exists(cnf_filepath)
                files[split] = split_files
        return files
    
    def _get_labels(self, data_dir):
        labels = {}
        if self.opts.label == 'satisfiability':
            for split in self.splits:
                if split == 'sat' or split == 'augmented_sat':
                    labels[split] = [torch.tensor(1., dtype=torch.float)] * self.split_len
                else:
                    # split == 'unsat' or split == 'augmented_unsat'
                    labels[split] = [torch.tensor(0., dtype=torch.float)] * self.split_len
        elif self.opts.label == 'assignment':
            for split in self.splits:
                # assert split == 'sat' or split == 'augmented_sat'
                labels[split] = []
                for cnf_filepath in self.all_files[split]:
                    filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
                    assignment_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_assignment.pkl')
                    with open(assignment_file, 'rb') as f:
                        assignment = pickle.load(f)
                    labels[split].append(torch.tensor(assignment, dtype=torch.float))
        elif self.opts.label == 'core_variable':
            for split in self.splits:
                assert split == 'unsat' or split == 'augmented_unsat'
                labels[split] = []
                for cnf_filepath in self.all_files[split]:
                    filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
                    assignment_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_core_variable.pkl')
                    with open(assignment_file, 'rb') as f:
                        core_variable = pickle.load(f)
                    labels[split].append(torch.tensor(core_variable, dtype=torch.float))
        else:
            assert self.opts.label == None
            for split in self.splits:
                labels[split] = [None] * self.split_len
        
        return labels
    

    def _get_labels_from_file(self, label_file):
        labels = {}
        df_labels = pd.read_csv(label_file)
        assert self.splits == ['unknown']
        for split in self.splits:
            labels[split] = []
            for cnf_filepath in self.all_files[split]:
                filename = '/'.join(cnf_filepath.split('/')[-2:])
                instance_labels = df_labels[df_labels['cnf_file'] == filename]['assignment'].values[0]
                if len(instance_labels) == 1:
                    lbl = [torch.inf]
                else:
                    lbl = [0 if int(x) < 0 else 1 for x in instance_labels.split()[:-1]]
                    N = int(filename.split('_')[0].split('N')[1])
                    assert len(lbl) == N, f'Length of lbl is {len(lbl)} but N is {N}!'

                labels[split].append(torch.tensor(lbl, dtype=torch.float))
        
        return labels

    def _get_split_len(self):
        lens = [len(self.all_files[split]) for split in self.splits]
        assert len(set(lens)) == 1
        return lens[0]
    
    def _get_file_name(self, split, cnf_filepath):
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        return f'{split}/{filename}_{self.opts.graph}.pt'
    
    def _get_positive_indices(self):
        # calculate the index to map the original instance to its augmented one, and vice versa.
        positive_indices = []
        for offset, split in enumerate(self.splits):
            if split == 'sat':
                positive_indices.append(torch.tensor(self.splits.index('augmented_sat')-offset, dtype=torch.long))
            elif split == 'augmented_sat':
                positive_indices.append(torch.tensor(self.splits.index('sat')-offset, dtype=torch.long))
            elif split == 'unsat':
                positive_indices.append(torch.tensor(self.splits.index('augmented_unsat')-offset, dtype=torch.long))
            elif split == 'augmented_unsat':
                positive_indices.append(torch.tensor(self.splits.index('unsat')-offset, dtype=torch.long))
        return positive_indices
    
    @property
    def processed_file_names(self):       
        names = []
        for split in self.splits:
            for cnf_filepath in self.all_files[split]:
                names.append(self._get_file_name(split, cnf_filepath))
        return names
    
    @property
    def processed_dir(self) -> str:
        path = None
        if self.opts.loss == 'supervised' or 'test' in self.data_partition:
            path = os.path.join(self.root, f'ss{self.sample_size}_processed')
        else:
            path = os.path.join(self.root, f'ss{self.sample_size}_processed_unsupervised')
        return path

    def _save_data(self, split, cnf_filepath):
        file_name = self._get_file_name(split, cnf_filepath)
        saved_path = os.path.join(self.processed_dir, file_name)
        if os.path.exists(saved_path):
            return
        
        n_vars, clauses, learned_clauses = parse_cnf_file(cnf_filepath, split_clauses=True)
        N = int(cnf_filepath.split('_')[0].split('N')[1])
        assert n_vars == N, f'N from parsing method is {n_vars} but N on filename is {N}!'
        
        
        # limit the size of the learned clauses to 1000 if learned clauses are enabled
        if len(learned_clauses) > 1000:
            clauses = clauses + learned_clauses[:1000]
        else:
            clauses = clauses + learned_clauses
        
        # clauses = clean_clauses(clauses) # This is a function that removes duplicate or empty clauses 
                    
        if self.opts.graph == 'lcg':
            data = construct_lcg(n_vars, clauses)
        elif self.opts.graph == 'vcg':
            data = construct_vcg(n_vars, clauses)

        torch.save(data, saved_path)
    
    def process(self):
        for split in self.splits:
            os.makedirs(os.path.join(self.processed_dir, split), exist_ok=True)
        
        for split in self.splits:
            for cnf_filepath in tqdm(self.all_files[split]):
                self._save_data(split, cnf_filepath)
    
    def len(self):
        if self.opts.data_fetching == 'parallel':
            return self.split_len
        else:
            return self.split_len * len(self.splits)

    def get(self, idx):
        if self.opts.data_fetching == 'parallel':
            data_list = []
            for split_idx, split in enumerate(self.splits):
                # Add check if we're doing OOD eval in large N (no available labels here)
                if self.opts.ood_largeN_eval:
                    cnf_filepath = self.all_files[split][idx]
                    file_name = self._get_file_name(split, cnf_filepath)
                    saved_path = os.path.join(self.processed_dir, file_name)
                    data = torch.load(saved_path)
                    data.y = torch.tensor([-1 for i in range(data.l_size // 2)], dtype=torch.float)
                    data.sat_problem = None
                    data_list.append(data)
                else:                
                    cnf_filepath = self.all_files[split][idx]
                    label = self.all_labels[split][idx]
                    file_name = self._get_file_name(split, cnf_filepath)
                    saved_path = os.path.join(self.processed_dir, file_name)
                    data = torch.load(saved_path)
                    
                    if torch.isinf(label).all(): 
                        data.y = torch.tensor([-1 for i in range(data.l_size // 2)], dtype=torch.float)
                        data.sat_problem = False
                    else:
                        data.y = label
                        data.sat_problem = True

                    if self.use_contrastive_learning:
                        data.positive_index = self.positive_indices[split_idx]
                    data_list.append(data)
            return data_list
        else:
            for split in self.splits:
                if idx >= self.split_len:
                    idx -= self.split_len
                else:
                    cnf_filepath = self.all_files[split][idx]
                    label = self.all_labels[split][idx]
                    file_name = self._get_file_name(split, cnf_filepath)
                    saved_path = os.path.join(self.processed_dir, file_name)
                    data = torch.load(saved_path)
                    data.y = label
                    return [data]


