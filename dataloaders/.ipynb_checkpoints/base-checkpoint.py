import pickle
import pandas as pd
import numpy as np
from rdkit import Chem 
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
import sys
import torch
from torch.utils.data import Dataset 
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split


###############################################
#                                             #
#              Dataset Base Class             #
#                                             #
###############################################


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def onek_encoding(x, allowable_set):
    if x not in allowable_set:                                                                                                                                               
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def featurization(x):

    return x

def check_exists(path):    
    return True if os.path.isfile(path) and os.path.getsize(path) > 0 else False

def add_index(input_array, ebd_size):
    add_idx, temp_arrays = 0, []
    for i in range(input_array.shape[0]):
        temp_array = input_array[i,:,:]
        masking_indices = temp_array.sum(1).nonzero()
        temp_array += add_idx
        temp_arrays.append(temp_array)
        add_idx = masking_indices[0].max()+1
    new_array = np.concatenate(temp_arrays, 0)

    return new_array.reshape(-1)


class DtiDatasetBase(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_instances, self.meta_instances = [], []

        self.analysis_mode = False

        # Gathering All Meta-data from DTI Datasets
        self.data_path = os.path.join(args.root_path, f'dataset_{args.dataset_version}/')
        complex_dataframe, protein_dataframe, ligand_dataframe = [], [], []
        
        for dataset in args.dataset_subsets.split('+'):
            complex_path = f'{self.data_path}complex_metadata_{dataset}.csv'
            protein_path = f'{self.data_path}protein_metadata_{dataset}.csv'
            ligand_path = f'{self.data_path}ligand_metadata_{dataset}.csv'
            complex_dataframe.append(pd.read_csv(complex_path, index_col='complex_id'))
            protein_dataframe.append(pd.read_csv(protein_path, index_col='protein_id'))
            ligand_dataframe.append(pd.read_csv(ligand_path, index_col='ligand_id'))
        
        self.complex_dataframe = pd.concat(complex_dataframe)
        self.protein_dataframe = pd.concat(protein_dataframe)
        self.ligand_dataframe = pd.concat(ligand_dataframe)

        self.complex_dataframe = self.complex_dataframe[self.complex_dataframe['ba_measure']==args.ba_measure]
        if not args.inference_mode:
            self.complex_dataframe.dropna(subset=['ba_value'], axis=0, inplace=True)

        self.complex_indices = self.complex_dataframe.index
        if args.debug_mode or args.toy_test:
            self.complex_indices = self.complex_dataframe.index[:args.debug_index]

        self.kfold_splits = []

        # Which Features to Include?
        self.protein_features = args.protein_features # 'esm+blosum+onehot'
        self.ligand_features = args.ligand_features 

    def check_ligand(self, ligand_idx):
        return

    def check_protein(self, protein_idx):
        if self.pdf.loc[protein_idx, 'fasta_length'] >= 1000:
            raise FastaLengthException(self.pdf.loc[protein_idx, 'fasta_length'])

    def check_complex(self, complex_idx):
        return 

    def __len__(self):
        return len(self.data_instances)

    def __getitem__(self, idx):
        if self.analysis_mode:
        # if self.args.analysis_model:
            return self.data_instances[idx], self.meta_instances[idx]
        else:
            return self.data_instances[idx]

    def make_random_splits(self):
        print("Making Random Splits")
        kf = KFold(n_splits=5, shuffle=True)
        for a, b in kf.split(self.indices):
            train_indices, test_indices = a, b
            train_indices, valid_indices = train_test_split(train_indices, test_size=0.05)
            self.kfold_splits.append((train_indices, valid_indices, test_indices))

class FastaLengthException(Exception):
    def __init__(self, fasta_length, message="fasta length should not exceed 1000"):
        self.fasta_length = fasta_length
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.fasta_length} -> {self.message}'

class NoProteinGraphException(Exception):
    def __init__(self, protein_idx, message="protein graph structure file not available"):
        self.protein_idx = protein_idx
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.protein_idx} -> {self.message}'

class NoProteinFeaturesException(Exception):
    def __init__(self, protein_idx, message="protein advanced features file not available"):
        self.protein_idx = protein_idx
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.protein_idx} -> {self.message}'

class NoComplexGraphException(Exception):
    def __init__(self, complex_idx, message="complex advanced features file not available"):
        self.complex_idx = complex_idx
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.complex_idx} -> {self.message}'


###############################################
#                                             #
#              Collate Functions              #
#                                             #
###############################################

def stack_and_pad(arr_list, max_length=None):
    M = max([x.shape[0] for x in arr_list]) if not max_length else max_length
    N = max([x.shape[1] for x in arr_list])
    T = np.zeros((len(arr_list), M, N))
    t = np.zeros((len(arr_list), M))
    s = np.zeros((len(arr_list), M, N))

    for i, arr in enumerate(arr_list):
        # sum of 16 interaction type, one is enough
        if len(arr.shape) > 2:
            arr = (arr.sum(axis=2) > 0.0).astype(float)
        T[i, 0:arr.shape[0], 0:arr.shape[1]] = arr
        t[i, 0:arr.shape[0]] = 1 if arr.sum() != 0.0 else 0
        s[i, 0:arr.shape[0], 0:arr.shape[1]] = 1 if arr.sum() != 0.0 else 0
    return T, t, s

def stack_and_pad_2d(arr_list, block='lower_left', max_length=None):
    max0 = max([a.shape[0] for a in arr_list]) if not max_length else max_length
    max1 = max([a.shape[1] for a in arr_list])
    list_shapes = [a.shape for a in arr_list]

    final_result = np.zeros((len(arr_list), max0, max1))
    final_masks_2d = np.zeros((len(arr_list), max0))
    final_masks_3d = np.zeros((len(arr_list), max0, max1))

    if block == 'upper_left':
        for i, shape in enumerate(list_shapes):
            # sum of 16 interaction type, one is enough
            if len(arr_list[i].shape) > 2:
                arr_list[i] = (arr_list[i].sum(axis=2) == True).astype(float)
            final_result[i, :shape[0], :shape[1]] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], :shape[1]] = 1
    elif block == 'lower_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, max1-shape[1]:] = 1
    elif block == 'lower_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, :shape[1]] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, :shape[1]] = 1
    elif block == 'upper_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], max1-shape[1]:] = 1
    else:
        raise

    return final_result, final_masks_2d, final_masks_3d

def stack_and_pad_3d(arr_list, block='lower_left'):
    max0 = max([a.shape[0] for a in arr_list])
    max1 = max([a.shape[1] for a in arr_list])
    max2 = max([a.shape[2] for a in arr_list])
    list_shapes = [a.shape for a in arr_list]

    final_result = np.zeros((len(arr_list), max0, max1, max2))
    final_masks_2d = np.zeros((len(arr_list), max0))
    final_masks_3d = np.zeros((len(arr_list), max0, max1))
    final_masks_4d = np.zeros((len(arr_list), max0, max1, max2))

    if block == 'upper_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], :shape[1], :shape[2]] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], :shape[1]] = 1
            final_masks_4d[i, :shape[0], :shape[1], :] = 1
    elif block == 'lower_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, max1-shape[1]:] = 1
            final_masks_4d[i, max0-shape[0]:, max1-shape[1]:, :] = 1
    elif block == 'lower_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, :shape[1]] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, :shape[1]] = 1
            final_masks_4d[i, max0-shape[0]:, :shape[1], :] = 1
    elif block == 'upper_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], max1-shape[1]:] = 1
            final_masks_4d[i, :shape[0], max1-shape[1]:, :] = 1
    else:
        raise

    return final_result, final_masks_2d, final_masks_3d, final_masks_4d

def ds_normalize(input_array):
    # Doubly Stochastic Normalization of Edges from CVPR 2019 Paper
    assert len(input_array.shape) == 3
    input_array = input_array / np.expand_dims(input_array.sum(1)+1e-8, axis=1)
    output_array = np.einsum('ijb,jkb->ikb', input_array,
                             input_array.transpose(1, 0, 2))
    output_array = output_array / (output_array.sum(0)+1e-8)

    return output_array