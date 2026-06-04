from .blocks import GraphEncoder, GraphDecoder
from torch_geometric.utils import to_dense_batch
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


class GraphKoopmanNet(nn.Module):
    """
    OBSOLETE: Legacy Koopman dynamics model from Experiment 3.
    This class operates directly on GNN node embeddings without explicit physical constraints.
    It is superseded by `GraphAwareKoopmanNet` which strictly enforces pairwise distances
    and graph energy conservation during the latent rollout.
    """
    def __init__(self, node_dim: int = 6, edge_dim: int = 1, hidden_dim: int = 64, latent_dim: int = 576, n_atoms: int = 9):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_atoms = n_atoms
        self.encoder = GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.decoder = GraphDecoder(state_dim=latent_dim, hidden_dim=128, n_atoms=n_atoms)
        self.A_raw = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)

    @property
    def K(self):
        A_skew = self.A_raw - self.A_raw.T
        return torch.matrix_exp(A_skew)

    def forward(self, node_features, edge_index, edge_features, lengths):
        return self.encoder(node_features, edge_index, edge_features)

    def forward_rollout(self, h0, steps=100, latent_seed=True):
        if latent_seed:
            h = h0[:, 0]  # (B, n_atoms, hidden_dim)
        else:
            raise NotImplementedError()
        K = self.K
        rollout = []
        h_init = h
        for t in range(steps):
            K_pow = torch.matrix_power(K, t)
            rollout.append(torch.matmul(h_init, K_pow.t()))
        return torch.stack(rollout, dim=1)

    def post_step_hook(self):
        pass


class GraphAwareKoopmanNet(nn.Module):
    """
    Graph-Aware Koopman dynamics model with learnable coupling parameter alpha.
    """
    def __init__(self, edge_index, node_dim: int = 6, edge_dim: int = 1, hidden_dim: int = 64, latent_dim: int = 576, n_atoms: int = 9):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_atoms = n_atoms
        self.encoder = GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.decoder = GraphDecoder(state_dim=latent_dim, hidden_dim=128, n_atoms=n_atoms)
        self.A_self = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.A_edge = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.register_buffer("edge_index", edge_index.clone())
        # Precompute P transition matrix
        P = torch.zeros(n_atoms, n_atoms)
        P[edge_index[1], edge_index[0]] = 1.0
        row_sums = P.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        P = P / row_sums
        self.register_buffer("P", P)



    @property
    def K_global(self):
        A_self_skew = self.A_self - self.A_self.T
        A_edge_skew = self.A_edge - self.A_edge.T
        I_N = torch.eye(self.n_atoms, device=self.A_self.device)
        P_sym = 0.5 * (self.P + self.P.T)
        term_self = torch.kron(I_N, A_self_skew)
        term_edge = torch.kron(P_sym, A_edge_skew)
        A_glob = term_self + self.alpha * term_edge
        return torch.matrix_exp(A_glob)

    def forward(self, node_features, edge_index, edge_features, lengths):
        return self.encoder(node_features, edge_index, edge_features)

    def transition_step(self, h, K_glob=None):
        if h.dim() == 2:
            h = h.unsqueeze(0)
            is_2d = True
        else:
            is_2d = False

        B_steps, n_atoms, hidden_dim = h.shape
        h_flat = h.reshape(B_steps, n_atoms * hidden_dim)
        if K_glob is None:
            K_glob = self.K_global

        out_flat = torch.matmul(h_flat, K_glob.t())
        out = out_flat.reshape(B_steps, n_atoms, hidden_dim)

        if is_2d:
            out = out.squeeze(0)
        return out
    def forward_rollout(self, h0, steps=100, latent_seed=True):
        if latent_seed:
            h = h0[:, 0]  # (B, n_atoms, hidden_dim)
        else:
            raise NotImplementedError()
        rollout = [h]
        K_glob = self.K_global
        for _ in range(1, steps):
            h = self.transition_step(h, K_glob=K_glob)
            rollout.append(h)
        return torch.stack(rollout, dim=1)
    def get_global_K(self):
        return self.K_global.detach().cpu().numpy()
    def compute_loss(self, outputs, targets, lengths, epoch, node_features=None):
        h_seq = outputs  # (B, T, n_atoms, hidden_dim)
        B, T, n_atoms, hidden_dim = h_seq.shape

        h_t_list, h_tgt_list = [], []
        for b, L in enumerate(lengths):
            if L < 2: continue
            h_t_list.append(h_seq[b, :L-1])
            h_tgt_list.append(h_seq[b, 1:L])

        h_t = torch.cat(h_t_list, dim=0) # (Total_steps, n_atoms, hidden_dim)
        h_tgt = torch.cat(h_tgt_list, dim=0) # (Total_steps, n_atoms, hidden_dim)

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
            # Isometric regularization: penalize bonded distance distortion
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


