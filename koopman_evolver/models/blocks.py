import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from abc import ABC, abstractmethod
import networkx as nx
import os
import urllib.request
import zipfile
import math
import warnings

# PyTorch Geometric
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import to_dense_batch
except ImportError:
    pass


class EdgeConditionedConv(MessagePassing):
    """
    Message passing layer incorporating edge features (bond distances).
    Computes message: m_ij = MLP([h_i, h_j, e_ij])
    """
    def __init__(self, in_channels: int, edge_channels: int, out_channels: int):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(tmp)


class GraphEncoder(nn.Module):
    """
    Encoder that outputs a flattened graph-state vector of node embeddings
    without any global pooling: (B, T, n_atoms, node_dim) → (B, T, n_atoms * hidden_dim).
    """
    def __init__(self, node_dim: int = 6, edge_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.node_project = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.conv1 = EdgeConditionedConv(in_channels=hidden_dim, edge_channels=edge_dim, out_channels=hidden_dim)
        self.conv2 = EdgeConditionedConv(in_channels=hidden_dim, edge_channels=edge_dim, out_channels=hidden_dim)
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        B, T, n_atoms, _ = node_features.shape
        n_edges = edge_index.shape[1]

        # Flatten time and batch dimensions for GNN processing
        x = node_features.reshape(B * T * n_atoms, -1)
        edge_attr = edge_features.reshape(B * T * n_edges, -1)
        device = node_features.device
        # Compute batch offset indexes for the graph structure
        offsets = torch.arange(B * T, device=device).view(B * T, 1, 1) * n_atoms
        edge_index_batched = edge_index.unsqueeze(0).to(device) + offsets
        edge_index_batched = edge_index_batched.transpose(0, 1).reshape(2, -1)
        # Message passing
        h = self.node_project(x)
        h = self.conv1(h, edge_index_batched, edge_attr)
        h = F.gelu(h)
        h = self.conv2(h, edge_index_batched, edge_attr) # Shape: (B * T * n_atoms, hidden_dim)
        # Flatten node dimensions to produce the graph-state vector
        h_seq = h.view(B, T, n_atoms, h.shape[-1]) # Shape: (B, T, 576)
        return h_seq


class GraphDecoder(nn.Module):
    """
    Decoder mapping the flattened graph state vector back to coordinates: R^576 → R^{n_atoms x 3}.
    """
    def __init__(self, state_dim: int = 576, hidden_dim: int = 128, n_atoms: int = 9):
        super().__init__()
        self.n_atoms = n_atoms
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_atoms * 3)
        )
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 2:
            B, D = s.shape
            out = self.mlp(s)
            return out.view(B, self.n_atoms, 3)
        else:
            B, T, D = s.shape
            out = self.mlp(s)
            return out.view(B, T, self.n_atoms, 3)


