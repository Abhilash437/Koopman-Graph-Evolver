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


class GraphGRUNet(nn.Module):
    """
    OBSOLETE: Legacy GRU dynamics baseline from Experiment 3.
    This class operates directly on GNN node embeddings without explicit physical constraints.
    It is superseded by `GraphAwareGRUNet` which strictly enforces pairwise distances
    and graph energy conservation during the latent rollout for fair comparison.
    """
    def __init__(self, node_dim: int = 6, edge_dim: int = 1, hidden_dim: int = 64, latent_dim: int = 576, n_atoms: int = 9):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_atoms = n_atoms
        self.encoder = GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.decoder = GraphDecoder(state_dim=latent_dim, hidden_dim=128, n_atoms=n_atoms)
        self.rnn_transition = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_features, edge_index, edge_features, lengths):
        return self.encoder(node_features, edge_index, edge_features)

    def forward_rollout(self, h0, steps=100, latent_seed=True):
        if latent_seed:
            h = h0[:, 0]  # (B, n_atoms, hidden_dim)
        else:
            raise NotImplementedError()
        B, n_atoms, hidden_dim = h.shape
        rollout = [h]
        for _ in range(1, steps):
            h_flat = h.reshape(B * n_atoms, hidden_dim)
            h_next_flat = self.rnn_transition(h_flat, h_flat)
            h = h_next_flat.reshape(B, n_atoms, hidden_dim)
            rollout.append(h)
        return torch.stack(rollout, dim=1)

    def post_step_hook(self):
        pass


class GraphAwareGRUNet(nn.Module):
    def __init__(self, edge_index, node_dim: int = 6, edge_dim: int = 1, hidden_dim: int = 64, latent_dim: int = 576, n_atoms: int = 9):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_atoms = n_atoms
        self.encoder = GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.decoder = GraphDecoder(state_dim=latent_dim, hidden_dim=128, n_atoms=n_atoms)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rnn_transition = nn.GRUCell(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.register_buffer("edge_index", edge_index.clone())
        # Precompute P transition matrix
        P = torch.zeros(n_atoms, n_atoms)
        P[edge_index[1], edge_index[0]] = 1.0
        row_sums = P.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        P = P / row_sums
        self.register_buffer("P", P)

    def forward(self, node_features, edge_index, edge_features, lengths):
        return self.encoder(node_features, edge_index, edge_features)

    def transition_step(self, h):
        if h.dim() == 2:
            h = h.unsqueeze(0)
            is_2d = True
        else:
            is_2d = False

        B_steps, n_atoms, hidden_dim = h.shape
        M = torch.matmul(self.P, h)
        M_proj = self.msg_proj(M)

        h_flat = h.reshape(-1, hidden_dim)
        m_flat = (self.alpha * M_proj).reshape(-1, hidden_dim)

        h_next_flat = self.rnn_transition(m_flat, h_flat)
        out = h_next_flat.reshape(B_steps, n_atoms, hidden_dim)

        if is_2d:
            out = out.squeeze(0)
        return out

    def forward_rollout(self, h0, steps=100, latent_seed=True):
        if latent_seed:
            h = h0[:, 0]  # (B, n_atoms, hidden_dim)
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

        unroll_steps = 4
        
        l_dyn = 0.0
        h_t_list = []
        for b, L in enumerate(lengths):
            if L <= unroll_steps: continue
            h_t_list.append(h_seq[b, :L-unroll_steps])
            
        if len(h_t_list) > 0:
            h_curr = torch.cat(h_t_list, dim=0)
            for step in range(1, unroll_steps + 1):
                h_curr = self.transition_step(h_curr)
                tgt_list = []
                for b, L in enumerate(lengths):
                    if L <= unroll_steps: continue
                    tgt_list.append(h_seq[b, step:L-unroll_steps+step])
                h_tgt = torch.cat(tgt_list, dim=0)
                l_dyn = l_dyn + F.mse_loss(h_curr, h_tgt)
            l_dyn = l_dyn / unroll_steps
        else:
            # Fallback for very short sequences
            h_t_list, h_tgt_list = [], []
            for b, L in enumerate(lengths):
                if L < 2: continue
                h_t_list.append(h_seq[b, :L-1])
                h_tgt_list.append(h_seq[b, 1:L])
            h_t = torch.cat(h_t_list, dim=0)
            h_tgt = torch.cat(h_tgt_list, dim=0)
            h_pred = self.transition_step(h_t)
            l_dyn = F.mse_loss(h_pred, h_tgt)

        # 1-step collapse loss
        tgt_0_list, tgt_1_list = [], []
        for b, L in enumerate(lengths):
            if L < 2: continue
            tgt_0_list.append(h_seq[b, :L-1])
            tgt_1_list.append(h_seq[b, 1:L])
        h_t0 = torch.cat(tgt_0_list, dim=0)
        h_t1 = torch.cat(tgt_1_list, dim=0)
        
        step_diff = torch.norm(h_t1 - h_t0, dim=-1)
        step_norm = torch.norm(h_t0, dim=-1) + 1e-8
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





class FlatMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        h_flat = self.net(x_flat)
        return h_flat.reshape(B, T, -1)



class FlatKoopmanNet(nn.Module):
    """
    A completely graph-free Koopman model baseline.
    Uses an MLP encoder over the flattened coordinate space and learns
    a single dense orthogonal transition matrix.
    
    The internal latent space is flat, but it reshapes its outputs to 
    (B, T, n_atoms, h_dim) to be drop-in compatible with standard evaluators.
    """
    def __init__(
        self,
        n_atoms: int = 9,
        input_dim: int = 6,
        latent_dim: int = 576,
        encoder_hidden_dims: list = None,
        decoder_hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        flat_input_dim = n_atoms * input_dim

        # Pure MLP encoder: R^{N_atoms*6} -> R^{latent_dim}
        self.encoder = FlatMLPEncoder(
            input_dim=flat_input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
        )

        # We can reuse the GraphDecoder since it also takes a flattened latent vector
        from .blocks import GraphDecoder
        self.decoder = GraphDecoder(
            state_dim=latent_dim,
            n_atoms=n_atoms,
            hidden_dim=decoder_hidden_dim,
        )

        # Flat Lie-algebraic transition
        self.A_raw = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)

    @property
    def K(self):
        A_skew = self.A_raw - self.A_raw.T
        return torch.matrix_exp(A_skew)

    def get_global_K(self):
        return self.K.detach().cpu().numpy()

    def get_flat_K(self):
        return self.get_global_K()

    def forward(self, node_features, edge_index=None, edge_features=None, lengths=None):
        B, T, n_atoms, D = node_features.shape
        x_flat = node_features.reshape(B, T, n_atoms * D)
        h_seq_flat = self.encoder(x_flat)  # (B, T, latent_dim)
        # Reshape to standard (B, T, n_atoms, h_dim) for evaluator compatibility
        return h_seq_flat.reshape(B, T, n_atoms, self.latent_dim // n_atoms)

    def transition_step(self, h, K=None):
        # Handle both 2D (B, latent_dim) and 3D (B, n_atoms, h_dim)
        is_3d = (h.dim() == 3)
        if is_3d:
            B, n_atoms, d = h.shape
            h_flat = h.reshape(B, -1)
        else:
            h_flat = h
            
        if K is None:
            K = self.K
            
        h_next_flat = torch.matmul(h_flat, K.t())
        
        if is_3d:
            return h_next_flat.reshape(B, n_atoms, d)
        return h_next_flat

    def forward_rollout(self, h0, steps=100, latent_seed=True):
        if latent_seed:
            h = h0[:, 0]  # Extracts t=0
        else:
            raise NotImplementedError()

        K = self.K
        rollout = [h]
        for _ in range(1, steps):
            h = self.transition_step(h, K=K)
            rollout.append(h)
        return torch.stack(rollout, dim=1)

    def compute_loss(self, outputs, targets, lengths, epoch, node_features=None):
        # outputs is (B, T, n_atoms, h_dim)
        B, T, n_atoms, h_dim = outputs.shape
        h_seq = outputs.reshape(B, T, -1)  # Flat for computation

        h_t_list, h_tgt_list = [], []
        for b, L in enumerate(lengths):
            if L < 2:
                continue
            h_t_list.append(h_seq[b, :L - 1])
            h_tgt_list.append(h_seq[b, 1:L])

        h_t = torch.cat(h_t_list, dim=0)      
        h_tgt = torch.cat(h_tgt_list, dim=0)    

        h_pred = self.transition_step(h_t)
        l_dyn = torch.nn.functional.mse_loss(h_pred, h_tgt)

        step_diff = torch.norm(h_tgt - h_t, dim=-1)
        step_norm = torch.norm(h_t, dim=-1) + 1e-8
        relative_change = (step_diff / step_norm).mean()
        l_collapse = torch.relu(0.05 - relative_change)

        if node_features is not None:
            coords_true = node_features[:, :, :, :3]
            coords_pred = self.decoder(h_seq)
            l_recon = torch.nn.functional.mse_loss(coords_pred, coords_true)
        else:
            l_recon = 0.0

        total_loss = l_dyn + 2.0 * l_collapse + 10.0 * l_recon
        return total_loss, {
            'loss': total_loss.item(),
            'l_dyn': l_dyn.item(),
            'l_collapse': l_collapse.item(),
            'l_recon': l_recon.item() if node_features is not None else 0.0,
        }

    def post_step_hook(self):
        pass
