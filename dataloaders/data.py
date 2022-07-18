from MONN import *
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import argparse
import pdb
import os
import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', default = '/home/hajung/study/monn/monn_practice/data', type =str)
parser.add_argument('--dataset_version', default=220622, type=int)
parser.add_argument('--dataset_subsets', default='pdb_2020_general', type=str)
parser.add_argument('--ba_measure', default='KIKD', type=str)
parser.add_argument('--inference_mode', default = False)
parser.add_argument('--debug_mode', default=False)
parser.add_argument('--toy_test', default=False)
parser.add_argument('--debug_index', default = False)
parser.add_argument('--protein_features', type=str)
parser.add_argument('--ligand_features', type = str)


parser.add_argument('--fold', default = 0, type=int)
parser.add_argument('--batch_size', default = 16, type=int)
#'KIKD' 'IC50'
args = parser.parse_args()


random.seed(12345)

dataset = DtiDataset(args)
dataset.make_random_splits()
indices = dataset.kfold_splits

fold_indices = indices[args.fold]
train_idx, valid_idx, test_idx = fold_indices[0], fold_indices[1], fold_indices[2]
train = SubsetRandomSampler(train_idx)
valid = SubsetRandomSampler(valid_idx)
test = SubsetRandomSampler(test_idx)

train_loader = DataLoader(dataset, batch_size = len(train),  collate_fn = collate_fn, sampler = train)
valid_loader = DataLoader(dataset, batch_size =len(valid), collate_fn = collate_fn, sampler = valid)
test_loader = DataLoader(dataset, batch_size = 8, collate_fn = collate_fn, sampler = test)


data = next(iter(test_loader))
with open('./data/data_sample.pkl' , 'wb') as writer:
    pickle.dump(data, writer)