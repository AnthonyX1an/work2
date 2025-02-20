import argparse
import copy
import os
import random
import sys
import warnings
import time, subprocess

import numpy as np
import torch 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F

from parse import parser_add_default_args, parser_add_main_args
from dataset import load_dataset

# data_dir = "/home/dell/sx/DoubleGum/data/ADHD/ADHD"
# data_dir = '/home/dell/sx/DoubleGum/data/EEG/FACED_dataset_2_labels.mat'
data_dir = '/home/dell/sx/MNIST'
data_name = 'MNIST'
n_blocks = 16
knn = 3

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
parser_add_default_args(args)
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")
    

if data_name == 'MNIST':
    train_dataset, test_dataset = load_dataset(data_dir, data_name, n_blocks, knn)
    #batch
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
elif data_name == 'EEG':
    loader, train_dataset, test_dataset = load_dataset(data_dir, data_name, n_blocks, knn)
elif data_name == 'ADHD':
    train_dataset, test_dataset = load_dataset(data_dir, data_name, n_blocks, knn)