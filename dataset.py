from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy
import scipy.io
import time
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.spatial.distance import cdist
import torch_geometric.transforms as T

from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url, class_rand_splits, adj_mul

from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import degree, to_undirected, remove_self_loops, add_self_loops
import os

from google_drive_downloader import GoogleDriveDownloader as gdd

import networkx as nx
import scipy.sparse as sp

from ogb.nodeproppred import NodePropPredDataset

from PIL import Image

from data.EEG.ge.protocols import *
from data.EEG.ge.models import *

from nilearn.connectome import ConnectivityMeasure

class NCDataset(Dataset):  # 继承自 PyTorch 的 Dataset 类
    def __init__(self, name):
        self.name = name
        self.graphs = []
        self.labels = []

    def add_graph_and_label(self, graph, label):
        self.graphs.append(graph)
        self.labels.append(label)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        return graph, label

    def __len__(self):
        return len(self.graphs)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def MNIST_to_blocks(img_np, n_blocks):
    img_array = np.array(img_np).astype(np.float32)
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # 转换成NCHW格式
    batch_size, channels, height, width = img_tensor.shape
    block_size = int(height / np.sqrt(n_blocks))  # 根据n_blocks计算每个block的尺寸
    blocks = []
    
    # 分块处理
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = img_tensor[:, :, i:i+block_size, j:j+block_size]
            block = block.flatten()
            blocks.append(block)
    
    # 将所有块的特征合并为一个张量
    return torch.stack(blocks, dim=0)
 
def load_MNIST_dataset(data_dir, name, n_blocks, knn):
    data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

    train_data = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_data = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)

    indices_train = [i for i, (img, label) in enumerate(train_data)][:800]
    indices_test = [i for i, (img, label) in enumerate(test_data)][:200]
    
    # indices_train = [i for i, (img, label) in enumerate(train_data) if label in [6, 9]][:800]
    # indices_test = [i for i, (img, label) in enumerate(test_data) if label in [6, 9]][:200]

    train_data_subset = Subset(train_data, indices_train)
    test_data_subset = Subset(test_data, indices_test)

    train_dataset = NCDataset('MNIST_Dataset')
    test_dataset = NCDataset('MNIST_Dataset')

    salt_prob = 0.003
    pepper_prob = 0.003

    for img, label in DataLoader(train_data_subset, batch_size=1):
        img_np = img.numpy().squeeze()  # 转换为numpy数组
        # img_np = add_salt_and_pepper_noise(img_np, salt_prob, pepper_prob)
        features = MNIST_to_blocks(img_np, n_blocks)  # 应用image_to_blocks处理图像
        
        # mapped_label = 0 if label.item() == 6 else 1
        
        graph = {
            'edge_index': None,
            'edge_feat': None,
            'node_feat': features,
            'num_nodes': n_blocks
        }
        edge_index = find_k_nearest(graph, knn)
        graph['edge_index'] = edge_index

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_blocks)
        adjs = [edge_index]
        for _ in range(2 - 1):
                edge_index = adj_mul(edge_index, edge_index, n_blocks)
                adjs.append(edge_index)
        graph['adjs'] = adjs
        # print("edge_index: ", adjs[0].shape)
        train_dataset.add_graph_and_label(graph, label)
    
    for img, label in DataLoader(test_data_subset, batch_size=1):
        img_np = img.squeeze().numpy()  # Converting tensor to numpy array
        # img_np = add_salt_and_pepper_noise(img_np, salt_prob, pepper_prob)
        # visualize_blocks_with_numbers(img_np, label)
        features = MNIST_to_blocks(img_np, n_blocks)  # Applying image_to_blocks
        
        # mapped_label = 0 if label.item() == 6 else 1
        
        graph = {
            'edge_index': None,
            'edge_feat': None,
            'node_feat': features,
            'num_nodes': n_blocks
        }
        edge_index = find_k_nearest(graph, knn)
        graph['edge_index'] = edge_index

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_blocks)
        adjs = [edge_index]
        for _ in range(2 - 1):
                edge_index = adj_mul(edge_index, edge_index, n_blocks)
                adjs.append(edge_index)
        graph['adjs'] = adjs
        # print("edge_index: ", graph['adjs'].shape)
        test_dataset.add_graph_and_label(graph, label)

    print("load dataset success")
    return train_dataset, test_dataset

def load_dataset(data_dir, dataname, n_blocks, knn):
    if dataname in ('Image'):
        dataset = load_Image_dataset(data_dir, dataname, n_blocks, knn)
    elif dataname in ('MNIST'):
        train_dataset, test_dataset = load_MNIST_dataset(data_dir, dataname, n_blocks, knn)
        return train_dataset, test_dataset
    elif dataname in ('EEG'):
        dataset = load_EEG_dataset(data_dir, dataname, n_blocks, knn)
    elif dataname in ('ADHD'):
        train_dataset, test_dataset = load_ADHD_as_NCDataset(data_dir, dataname, n_blocks, knn)
        return train_dataset, test_dataset
    else:
        raise ValueError('Invalid dataname')
    return dataset