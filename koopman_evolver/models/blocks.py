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

class EGNN_Layer(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0):
        super(EGNN_Layer, self).__init__()
        self.hidden_nf = hidden_nf

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_node_nf * 2 + 1 + in_edge_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU()
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1, bias=False)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_nf + hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, out_node_nf)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        row, col = edge_index

        # Relative distance squared
        coord_diff = x[row] - x[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        h_row, h_col = h[row], h[col]
        if edge_attr is None:
            edge_input = torch.cat([h_row, h_col, radial], dim=1)
        else:
            edge_input = torch.cat([h_row, h_col, radial, edge_attr], dim=1)

        m_ij = self.edge_mlp(edge_input)

        # Coordinate update (sum over neighbors)
        coord_trans = coord_diff * self.coord_mlp(m_ij)
        x_agg = torch.zeros_like(x)
        x_agg.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(1)), coord_trans)

        # Update coordinates
        x_new = x + x_agg

        # Message aggregation
        m_i = torch.zeros(h.size(0), self.hidden_nf, device=h.device)
        m_i.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.hidden_nf), m_ij)

        # Node feature update
        node_input = torch.cat([h, m_i], dim=-1)
        h_new = h + self.node_mlp(node_input)

        return h_new, x_new

class EGNNDynamicsNet(nn.Module):
    """
    EGNN wrapper complying with the GraphAwareTrainer and Evaluator API.
    """
    def __init__(self, edge_index, node_dim=6, edge_dim=1, hidden_dim=64, latent_dim=576, n_atoms=9):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_atoms = n_atoms
        self.register_buffer("edge_index", edge_index.clone())

        self.encoder_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.decoder = DummyDecoder(n_atoms=n_atoms, hidden_dim=hidden_dim)

        self.egnn_layers = nn.ModuleList([
            EGNN_Layer(hidden_dim - 3, hidden_dim, hidden_dim - 3)
            for _ in range(3)
        ])

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, node_features, edge_index, edge_features, lengths):
        # Force the first 3 dims of the hidden state to be the spatial coordinates
        x = node_features[:, :, :, :3]
        h_feat = self.encoder_mlp(node_features)
        h_state = torch.cat([x, h_feat[:, :, :, 3:]], dim=-1)
        return h_state

    def transition_step(self, h):
        if h.dim() == 2:
            h = h.unsqueeze(0)
            is_2d = True
        else:
            is_2d = False

        B_steps, n_atoms, hidden_dim = h.shape
        x = h[:, :, :3]
        feat = h[:, :, 3:]

        x_flat = x.reshape(B_steps * n_atoms, 3)
        feat_flat = feat.reshape(B_steps * n_atoms, hidden_dim - 3)

        E = self.edge_index.size(1)
        batch_offsets = torch.arange(B_steps, device=h.device).repeat_interleave(E) * n_atoms
        edge_idx_batched = self.edge_index.repeat(1, B_steps) + batch_offsets

        for layer in self.egnn_layers:
            feat_flat, x_flat = layer(feat_flat, x_flat, edge_idx_batched)

        x_new = x_flat.view(B_steps, n_atoms, 3)
        feat_new = feat_flat.view(B_steps, n_atoms, hidden_dim - 3)

        out = torch.cat([x_new, feat_new], dim=-1)
        if is_2d: out = out.squeeze(0)
        return out

    def forward_rollout(self, h0, steps=100, latent_seed=True):
        if latent_seed:
            h = h0[:, 0]
        else:
            raise NotImplementedError()
        rollout = [h]
        for _ in range(1, steps):
            h = self.transition_step(h)
            rollout.append(h)
        return torch.stack(rollout, dim=1)

    def compute_loss(self, outputs, targets, lengths, epoch, node_features=None):
        h_seq = outputs
        B, T, n_atoms, hidden_dim = h_seq.shape

        h_t_list, h_tgt_list = [], []
        for b, L in enumerate(lengths):
            if L < 2: continue
            h_t_list.append(h_seq[b, :L-1])
            h_tgt_list.append(h_seq[b, 1:L])

        h_t = torch.cat(h_t_list, dim=0)
        h_tgt = torch.cat(h_tgt_list, dim=0)

        h_pred = self.transition_step(h_t)
        l_dyn = F.mse_loss(h_pred, h_tgt)

        step_diff = torch.norm(h_tgt - h_t, dim=-1)
        step_norm = torch.norm(h_t, dim=-1) + 1e-8
        relative_change = (step_diff / step_norm).mean()
        l_collapse = torch.relu(0.05 - relative_change)

        if node_features is not None:
            coords_true = node_features[:, :, :, :3]
            s_seq = h_seq.reshape(B, T, n_atoms * hidden_dim)
            coords_pred = self.decoder(s_seq)
            l_recon = F.mse_loss(coords_pred, coords_true)
            src_idx, dst_idx = self.edge_index[0], self.edge_index[1]
            d_true = torch.norm(coords_true[:, :, src_idx] - coords_true[:, :, dst_idx], dim=-1)
            d_pred = torch.norm(coords_pred[:, :, src_idx] - coords_pred[:, :, dst_idx], dim=-1)
            l_iso = F.mse_loss(d_pred, d_true)
        else:
            l_recon = 0.0
            l_iso = 0.0

        total_loss = l_dyn + 2.0 * l_collapse + 10.0 * l_recon + 5.0 * l_iso
        return total_loss, {
            'loss': total_loss.item(),
            'l_dyn': l_dyn.item(),
            'l_collapse': l_collapse.item(),
            'l_recon': l_recon.item() if node_features is not None else 0.0,
            'l_iso': l_iso.item() if node_features is not None else 0.0,
            'alpha': float(self.alpha.item())
        }

    def post_step_hook(self):
        pass


class EquivariantGraphEncoder(nn.Module):
    def __init__(self, node_dim=6, hidden_dim=64):
        super().__init__()
        self.feat_dim = hidden_dim - 3
        self.encoder_mlp = nn.Sequential(
            nn.Linear(node_dim - 3, self.feat_dim),
            nn.SiLU(),
            nn.Linear(self.feat_dim, self.feat_dim)
        )
        self.egnn1 = EGNN_Layer(in_node_nf=self.feat_dim, hidden_nf=hidden_dim, out_node_nf=self.feat_dim)
        self.egnn2 = EGNN_Layer(in_node_nf=self.feat_dim, hidden_nf=hidden_dim, out_node_nf=self.feat_dim)

    def forward(self, node_features, edge_index, edge_features=None):
        B, T, V, C = node_features.shape
        x = node_features[:, :, :, :3].reshape(B * T * V, 3)
        h_feat = node_features[:, :, :, 3:].reshape(B * T * V, C - 3)

        h = self.encoder_mlp(h_feat)

        E = edge_index.size(1)
        batch_offsets = torch.arange(B * T, device=node_features.device).repeat_interleave(E) * V
        edge_idx_batched = edge_index.repeat(1, B * T) + batch_offsets

        h, x = self.egnn1(h, x, edge_idx_batched)
        h, x = self.egnn2(h, x, edge_idx_batched)

        z = torch.cat([x, h], dim=-1)
        return z.view(B, T, V, 64)

class EquivariantGraphDecoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.feat_dim = hidden_dim - 3
        self.egnn = EGNN_Layer(in_node_nf=self.feat_dim, hidden_nf=hidden_dim, out_node_nf=self.feat_dim)

    def forward(self, h_inv, x_prev, edge_index):
        orig_shape = h_inv.shape[:-1]
        V = h_inv.size(-2)

        h_flat = h_inv.reshape(-1, self.feat_dim)
        x_flat = x_prev.reshape(-1, 3)

        num_graphs = h_flat.size(0) // V
        E = edge_index.size(1)
        batch_offsets = torch.arange(num_graphs, device=h_inv.device).repeat_interleave(E) * V
        edge_idx_batched = edge_index.repeat(1, num_graphs) + batch_offsets

        _, x_new = self.egnn(h_flat, x_flat, edge_idx_batched)
        return x_new.view(*orig_shape, 3)

class DummyDecoder(nn.Module):
    def __init__(self, n_atoms, hidden_dim=64):
        super().__init__()
        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
    def forward(self, z_flat):
        is_3d = z_flat.dim() == 3
        if is_3d:
            B, T, dim = z_flat.shape
            z_flat = z_flat.reshape(B * T, dim)

        B_T = z_flat.shape[0]
        z = z_flat.view(B_T, self.n_atoms, self.hidden_dim)
        x = z[:, :, :3]

        if is_3d:
            x = x.view(B, T, self.n_atoms, 3)
        return x

