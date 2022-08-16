import json
import numpy as np
from rdkit import Chem
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_add_pool

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
    
class GCN(torch.nn.Module):
    def __init__(
        self,
        n_encodings,
        n_channels=32,
        preprocess_layers=2,
        convolutional_layer=GCNConv,
        convolutional_layers=6,
        convolutional_layer_args = {
            'aggr': 'add'
        },
        postprocess_layers=2,
        postprocess_channels=32,
        global_pooling=global_add_pool,
        dropout=0.25
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_encodings, n_channels)
        self.preprocess = nn.ModuleList([nn.Linear(n_channels, n_channels) for _ in range(preprocess_layers)])
        self.convs = nn.ModuleList([convolutional_layer(n_channels, n_channels, **convolutional_layer_args) for _ in range(convolutional_layers)])
        self.global_pooling = global_pooling
        self.dropout = dropout
        self.postprocess = nn.ModuleList([nn.Linear(postprocess_channels if i == 0 else n_channels, n_channels) for i in range(postprocess_layers)])
        self.properties = nn.Linear(n_channels, 16)

    def forward(self, data, mode='logp'):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.embedding(x)

        for linear in self.preprocess:
            x = linear(x)
            x = F.relu(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.global_pooling(x, batch)

        for linear in self.postprocess:
            x = linear(x)
            x = F.relu(x)

        if mode == 'logp':
            return torch.reshape(self.properties(x)[:, 0], (-1, 1))
        elif mode == 'both':
            return self.properties(x)
        elif mode == 'properties':
            return torch.reshape(self.properties(x)[:, 1:], (-1, 15))