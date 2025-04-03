import math
import os
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GCN
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch import Tensor

def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention=torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1) #[N, N]
        normalizer=attention_normalizer.squeeze(dim=-1).mean(dim=-1,keepdims=True) #[N,1]
        attention=attention/normalizer


    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1,
                                             self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1,
                                            self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1,
                                                  self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act=use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x = data['node_feat']
        edge_index = data['edge_index']
        edge_weight = data['edge_weight'] if 'edge_weight' in data else None
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            if self.use_act:
                x = self.activation(x) 
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]

class Masker(nn.Module):
    def __init__(self, n_heads: int, n_nodes: int) -> None:
        super().__init__()
        self.mask = nn.Parameter(torch.Tensor(n_heads, n_nodes, 2))
        nn.init.xavier_normal_(self.mask)

    def forward(self) -> tuple[Tensor, Tensor]:
        """
        Outputs:
            mask - [n_heads, n_nodes], binary selection mask per head
            mask_logits - [n_heads, n_nodes], selection logits
        """
        mask_logits = torch.log_softmax(self.mask, dim=-1)
        mask = F.gumbel_softmax(mask_logits, tau=1, hard=True)[..., 1]  # Select 1 (include)
        return mask, mask_logits[..., 1]


class HyBRiDConstructor(nn.Module):
    def __init__(self, n_hypers: int, n_nodes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.masker = Masker(n_hypers, n_nodes)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Inputs:
            x - [n_nodes, feature_dim]
        Outputs:
            h - [n_hypers, feature_dim]  # aggregated features per hyperedge
            mask - [n_hypers, n_nodes]
            mask_logits - [n_hypers, n_nodes]
        """
        x = x["node_feat"]
        n_nodes, feat_dim = x.size()
        mask, mask_logits = self.masker()  # mask: [n_hypers, n_nodes]

        x = x.unsqueeze(0)                     # [1, n_nodes, feature_dim]
        mask = mask.unsqueeze(-1)              # [n_hypers, n_nodes, 1]
        x_masked = mask * x                    # [n_hypers, n_nodes, feature_dim]

        h = x_masked.sum(1) / (1e-7 + mask.sum(1))  # [n_hypers, feature_dim]
        h = self.dropout(h)

        return h, (mask.squeeze(-1), mask_logits)

def edge_mapping(mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    n_hypers, n_nodes = mask.shape
    node_to_hyper = [[] for _ in range(n_nodes)]
    
    for hyper_id in range(n_hypers):
        for node_id in range(n_nodes):
            if mask[hyper_id, node_id] > 0:
                node_to_hyper[node_id].append(hyper_id)

    old_src, old_dst = edge_index
    new_edges = set()

    for i in range(edge_index.size(1)):
        src = old_src[i].item()
        dst = old_dst[i].item()
        hypers_src = node_to_hyper[src]
        hypers_dst = node_to_hyper[dst]

        for h_src in hypers_src:
            for h_dst in hypers_dst:
                new_edges.add((h_src, h_dst))  # 自动去重

    if len(new_edges) > 0:
        edge_index_new = torch.tensor(list(new_edges), dtype=torch.long).t().contiguous()
    else:
        edge_index_new = torch.empty((2, 0), dtype=torch.long)

    return edge_index_new

class SGFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, 
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, use_act=False, graph_weight=0.8, gnn=None, aggregate='add'):
        super().__init__()
        self.trans_conv=TransConv(in_channels,hidden_channels,num_layers,num_heads,alpha,dropout,use_bn,use_residual,use_weight)
        self.gnn=gnn
        self.use_graph=use_graph
        self.graph_weight=graph_weight
        self.use_act=use_act
        self.constructor= HyBRiDConstructor(n_hypers=8, n_nodes=16)
        self.aggregate=aggregate

        if aggregate=='add':
            # self.fc=nn.Linear(hidden_channels,out_channels)
            self.fc=nn.Linear(512, out_channels)
        elif aggregate=='cat':
            # self.fc=nn.Linear(2*hidden_channels,out_channels)
            self.fc=nn.Linear(512, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')
        
        self.params1=list(self.trans_conv.parameters())
        self.params2=list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()) )

    def forward(self,data):
        x, mask= self.constructor(data)
        # data["node_feat"] = x
        edge_index_old = data["edge_index"]
        edge_index_new = edge_mapping(mask[0], edge_index_old).to(x.device)
        data_new = {
            'node_feat': x,
            'edge_index': edge_index_new
        }
        x1=self.trans_conv(data_new)
        if self.use_graph:
            x2=self.gnn(data_new)
            if self.aggregate=='add':
                x=self.graph_weight*x2+(1-self.graph_weight)*x1
            else:
                x=torch.cat((x1,x2),dim=1)
        else:
            x=x1
        x = x.flatten()
        x=self.fc(x)
        return x
    
    def get_attentions(self, x):
        attns=self.trans_conv.get_attentions(x) # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()