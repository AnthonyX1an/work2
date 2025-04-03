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
from eval import evaluate, eval_acc, eval_rocauc, eval_f1
from parse import parse_method, parser_add_main_args, parser_add_default_args 
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
elif data_name == 'EEG':
    loader, train_dataset, test_dataset = load_dataset(data_dir, data_name, n_blocks, knn)
elif data_name == 'ADHD':
    train_dataset, test_dataset = load_dataset(data_dir, data_name, n_blocks, knn)

if data_name == 'MNIST':
    n = train_dataset.graphs[0]['num_nodes']
    e = train_dataset.graphs[0]['edge_index'].shape[1]
    c = 10
    d = train_dataset.graphs[0]['node_feat'].shape[1]
elif data_name == 'EEG':
    n = train_dataset.graphs[0]['num_nodes']
    e = train_dataset.graphs[0]['edge_index'].shape[1]
    c = 2
    d = train_dataset.graphs[0]['node_feat'].shape[1]
elif data_name == 'ADHD':
    n = train_dataset.graphs[0]['num_nodes']
    e = train_dataset.graphs[0]['edge_index_in'].shape[1]
    c = 2
    d = train_dataset.graphs[0]['node_feat'].shape[1]

print(f"dataset {data_name} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

model = parse_method(args.method, args, c, d, device)

criterion1 = nn.CrossEntropyLoss()
eval_func = eval_acc
print("Model:", model)

optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

train_losses = []
test_losses = []
accuracies = []
f1_scores = []
precisions = []
recalls = []
best_loss = float('inf')  # 初始化为正无穷
best_acc = 0
best_weights = {}  # 用于保存最低损失时的权重

model.reset_parameters()
train_list = [i for i in range(len(train_dataset.graphs))]

def save_best_weights(node_weight, save_path="best_weights"):
    # Save node weights as separate files for label 0 and label 1
    torch.save(node_weight[0], f"{save_path}_label_0_node_weight.pt")
    torch.save(node_weight[1], f"{save_path}_label_1_node_weight.pt")
    print(f"Best node weights saved to {save_path}_label_0_node_weight.pt and {save_path}_label_1_node_weight.pt")

for epoch in range(args.epochs):
    #这里需要修改
    #train
    model.train()
    total_loss = 0
    best_val = float('-inf')
    random.shuffle(train_list)
    flag = 0
    for i in train_list:
        flag += 1
        optimizer.zero_grad()
        # center_optimizer.zero_grad()
        graph = train_dataset.graphs[train_list[i]]
        label =  train_dataset.labels[train_list[i]]
        
        graph['node_feat'] = graph['node_feat'].to(device)  # This is fine for a single tensor
        # adjs_list = graph['adjs']
        # graph['edge_index'] = torch.stack(adjs_list, dim=0)
        graph['adjs'] = [adj.to(device) for adj in graph['adjs']]
        graph['edge_index'] = graph['adjs'][0]
        # out = model(graph['node_feat'], graph['adjs'])
        out = model(graph)
        
        if label == 0:
            label = torch.tensor(0).to(device)
        elif label == 1:
             label = torch.tensor(1).to(device)
        elif label == 2:
            label = torch.tensor(2).to(device)
        elif label == 3:
             label = torch.tensor(3).to(device)
        elif label == 4:
            label = torch.tensor(4).to(device)
        elif label == 5:
            label = torch.tensor(5).to(device)
        elif label == 6:
            label = torch.tensor(6).to(device)
        elif label == 7:
            label = torch.tensor(7).to(device)
        elif label == 8:
            label = torch.tensor(8).to(device)
        elif label == 9:
            label = torch.tensor(9).to(device)

        classification_loss = criterion1(out, label)
        loss = classification_loss
        loss.backward()
        optimizer.step()
        total_loss = classification_loss.detach().cpu().item() + total_loss
    total_loss = total_loss / len(train_dataset.graphs)
    train_losses.append(total_loss)
    print(f'Epoch: {epoch}',
          f'Loss:{total_loss}')

    #eval
    model.eval()
    #用于保存weight
    epoch_node_weights = {0: [], 1: []}
    
    with torch.no_grad():
        ground_truth = []
        y_pred =[] 
        t_loss = 0.0
        flag_ = 0
        for graph, label in zip(test_dataset.graphs, test_dataset.labels):
            flag_ += 1
            graph['node_feat'] = graph['node_feat'].to(device)  # This is fine for a single tensor
            # adjs_list = graph['adjs']
            # graph['edge_index'] = torch.stack(adjs_list, dim=0)
            graph['adjs'] = [adj.to(device) for adj in graph['adjs']]
            graph['edge_index'] = graph['adjs'][0]
            # out = model(graph['node_feat'], graph['adjs'])
            out = model(graph)
            if label == 0:
                label = torch.tensor(0).to(device)
            elif label == 1:
                label = torch.tensor(1).to(device)
            elif label == 2:
                label = torch.tensor(2).to(device)
            elif label == 3:
                label = torch.tensor(3).to(device)
            elif label == 4:
                label = torch.tensor(4).to(device)
            elif label == 5:
                label = torch.tensor(5).to(device)
            elif label == 6:
                label = torch.tensor(6).to(device)
            elif label == 7:
                label = torch.tensor(7).to(device)
            elif label == 8:
                label = torch.tensor(8).to(device)
            elif label == 9:
                label = torch.tensor(9).to(device)
            ground_truth.append(label.cpu().numpy().item())  
            classification_loss = criterion1(out, label)
            loss = classification_loss
            
            t_loss += loss.detach().cpu().item()
            y_pred.append(torch.argmax(out,dim=-1).detach().cpu().item())  
            
            # if label == 0:
            #     epoch_node_weights[0].append(node_weight.cpu().numpy())
            # else:
            #     epoch_node_weights[1].append(node_weight.cpu().numpy())
            
        accuracy = accuracy_score(ground_truth, y_pred)   
        f1 = f1_score(ground_truth, y_pred, average='weighted')  # 对于二分类问题  
        precision = precision_score(ground_truth, y_pred, average='weighted')  # 对于二分类问题  
        recall = recall_score(ground_truth, y_pred, average='weighted')  # 对于二分类问题 
        test_loss = t_loss/len(test_dataset.graphs)
        
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
         
        print("Test Loss: ",round(t_loss/len(test_dataset.graphs),4)\
                ," Test Accuracy:",round(accuracy ,4)\
                ," Test F1:",round(f1,4)\
                ," Test Precision:",round(precision,4)\
                ," Test Recall:",round(recall,4) )    
        if accuracy > best_acc:
            print(f"New best accuracy found: {accuracy}, updating best node weights")
            best_acc = accuracy
            best_node_weight = epoch_node_weights.copy()
        else:
            print(f"Accuracy {accuracy} is not better than best_acc {best_acc}, no update")

        # top4_values_9, top4_indices_9 = torch.topk(torch.tensor(best_weights[0]), 4, dim=0)
        # top4_values_6, top4_indices_6 = torch.topk(torch.tensor(best_weights[2]), 4, dim=0)
        
        # print(f"best_acc:{best_acc}")
        # print(f"best_weight9:{best_weights[0]}, best_weight6:{best_weights[2]}")
        # print(f"top4_9:{top4_indices_9}, top4_6:{top4_indices_6}")
        
        scheduler.step(t_loss / len(test_dataset.graphs))

if best_node_weight[0] is not None and best_node_weight[1] is not None:
    save_best_weights(best_node_weight)
else:
    print("No best weights to save. Ensure that the model improved during training.")

epochs = list(range(1, args.epochs + 1))