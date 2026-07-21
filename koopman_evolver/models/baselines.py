from .blocks import GraphEncoder, GraphDecoder, EquivariantGraphEncoder, EquivariantGraphDecoder, DummyDecoder
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


# --- class EquivariantKoopmanNet(nn.Module): ---
# ── Phase 7: Cell 3 - Equivariant Graph Koopman Network (E-GKN) ───────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class EquivariantKoopmanNet(nn.Module):
    def __init__(self, edge_index, node_dim=6, edge_dim=1, hidden_dim=64, latent_dim=576, n_atoms=9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_atoms = n_atoms
        self.encoder = EquivariantGraphEncoder(node_dim=node_dim, hidden_dim=hidden_dim)
        self.decoder = DummyDecoder(n_atoms=n_atoms, hidden_dim=hidden_dim)
        self.eq_updater = EquivariantGraphDecoder(hidden_dim=hidden_dim)

        self.A_self = nn.Parameter(torch.zeros(hidden_dim - 3, hidden_dim - 3))
        self.A_edge = nn.Parameter(torch.zeros(hidden_dim - 3, hidden_dim - 3))
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.register_buffer("edge_index", edge_index.clone())
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
        return self.encoder(node_features, edge_index)

    def transition_step(self, z, K_glob=None):
        if z.dim() == 2:
            z = z.unsqueeze(0)
            is_2d = True
        else:
            is_2d = False

        B_steps, n_atoms, hidden_dim = z.shape
        x_t = z[:, :, :3]
        h_t = z[:, :, 3:]

        h_flat = h_t.reshape(B_steps, n_atoms * (hidden_dim - 3))
        if K_glob is None: K_glob = self.K_global

        h_pred_flat = torch.matmul(h_flat, K_glob.t())
        h_pred = h_pred_flat.reshape(B_steps, n_atoms, hidden_dim - 3)

        x_pred = self.eq_updater(h_pred, x_t, self.edge_index)

        z_pred = torch.cat([x_pred, h_pred], dim=-1)
        if is_2d: z_pred = z_pred.squeeze(0)
        return z_pred

    def forward_rollout(self, z0, steps=100, latent_seed=True):
        if latent_seed:
            z = z0[:, 0]
        else:
            raise NotImplementedError()
        rollout = [z]
        for _ in range(1, steps):
            z = self.transition_step(z)
            rollout.append(z)
        return torch.stack(rollout, dim=1)

    def compute_loss(self, outputs, targets, lengths, epoch, node_features=None):
        z_seq = outputs
        B, T, n_atoms, hidden_dim = z_seq.shape

        z_t_list, z_tgt_list = [], []
        for b, L in enumerate(lengths):
            if L < 2: continue
            z_t_list.append(z_seq[b, :L-1])
            z_tgt_list.append(z_seq[b, 1:L])

        z_t = torch.cat(z_t_list, dim=0)
        z_tgt = torch.cat(z_tgt_list, dim=0)

        z_pred = self.transition_step(z_t)
        l_dyn = F.mse_loss(z_pred, z_tgt)

        step_diff = torch.norm(z_tgt - z_t, dim=-1)
        step_norm = torch.norm(z_t, dim=-1) + 1e-8
        relative_change = (step_diff / step_norm).mean()
        l_collapse = torch.relu(0.05 - relative_change)

        if node_features is not None:
            coords_true = node_features[:, :, :, :3]
            z_flat = z_seq.reshape(B, T, n_atoms * hidden_dim)
            coords_pred = self.decoder(z_flat)
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

class EquivariantGRUNet(nn.Module):
    def __init__(self, edge_index, node_dim=6, edge_dim=1, hidden_dim=64, latent_dim=576, n_atoms=9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_atoms = n_atoms
        self.encoder = EquivariantGraphEncoder(node_dim=node_dim, hidden_dim=hidden_dim)
        self.decoder = DummyDecoder(n_atoms=n_atoms, hidden_dim=hidden_dim)
        self.eq_updater = EquivariantGraphDecoder(hidden_dim=hidden_dim)

        self.msg_proj = nn.Linear(hidden_dim - 3, hidden_dim - 3, bias=False)
        self.rnn_transition = nn.GRUCell(hidden_dim - 3, hidden_dim - 3)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.register_buffer("edge_index", edge_index.clone())
        P = torch.zeros(n_atoms, n_atoms)
        P[edge_index[1], edge_index[0]] = 1.0
        row_sums = P.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        P = P / row_sums
        self.register_buffer("P", P)

    def forward(self, node_features, edge_index, edge_features, lengths):
        return self.encoder(node_features, edge_index)

    def transition_step(self, z):
        if z.dim() == 2:
            z = z.unsqueeze(0)
            is_2d = True
        else:
            is_2d = False

        B_steps, n_atoms, hidden_dim = z.shape
        x_t = z[:, :, :3]
        h_t = z[:, :, 3:]

        M = torch.matmul(self.P, h_t)
        M_proj = self.msg_proj(M)

        h_flat = h_t.reshape(-1, hidden_dim - 3)
        m_flat = (self.alpha * M_proj).reshape(-1, hidden_dim - 3)

        h_next_flat = self.rnn_transition(m_flat, h_flat)
        h_pred = h_next_flat.reshape(B_steps, n_atoms, hidden_dim - 3)

        x_pred = self.eq_updater(h_pred, x_t, self.edge_index)

        z_pred = torch.cat([x_pred, h_pred], dim=-1)
        if is_2d: z_pred = z_pred.squeeze(0)
        return z_pred

    def forward_rollout(self, z0, steps=100, latent_seed=True):
        if latent_seed:
            z = z0[:, 0]
        else:
            raise NotImplementedError()
        rollout = [z]
        for _ in range(1, steps):
            z = self.transition_step(z)
            rollout.append(z)
        return torch.stack(rollout, dim=1)

    def compute_loss(self, outputs, targets, lengths, epoch, node_features=None):
        z_seq = outputs
        B, T, n_atoms, hidden_dim = z_seq.shape

        z_t_list, z_tgt_list = [], []
        for b, L in enumerate(lengths):
            if L < 2: continue
            z_t_list.append(z_seq[b, :L-1])
            z_tgt_list.append(z_seq[b, 1:L])

        z_t = torch.cat(z_t_list, dim=0)
        z_tgt = torch.cat(z_tgt_list, dim=0)

        z_pred = self.transition_step(z_t)
        l_dyn = F.mse_loss(z_pred, z_tgt)

        step_diff = torch.norm(z_tgt - z_t, dim=-1)
        step_norm = torch.norm(z_t, dim=-1) + 1e-8
        relative_change = (step_diff / step_norm).mean()
        l_collapse = torch.relu(0.05 - relative_change)

        if node_features is not None:
            coords_true = node_features[:, :, :, :3]
            z_flat = z_seq.reshape(B, T, n_atoms * hidden_dim)
            coords_pred = self.decoder(z_flat)
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



import sys
import os

# Ensure SEGNO is on the path so we can import its modules
segno_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'SEGNO', 'MD22', 'spatial_graph'))
if segno_path not in sys.path:
    sys.path.append(segno_path)

try:
    from n_body_system.model import SEGNO  # type: ignore
except ImportError:
    SEGNO = None

class SEGNODynamicsNet(nn.Module):
    """
    SEGNO wrapper complying with the GraphAwareTrainer and Evaluator API.
    """
    def __init__(self, edge_index, node_dim=6, edge_dim=1, hidden_dim=64, latent_dim=576, n_atoms=9):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_atoms = n_atoms
        self.register_buffer("edge_index", edge_index.clone())

        if SEGNO is None:
            raise ImportError("SEGNO could not be imported. Ensure SEGNO repository is accessible.")

        self.model = SEGNO(
            in_node_nf=node_dim,
            in_edge_nf=edge_dim, # pass bond distances as edge_attr to match E-GKN
            hidden_nf=hidden_dim,
            device='cpu',
            n_layers=3,
            coords_weight=1.0,
            recurrent=False,
            norm_diff=True, # crucial stabilization for unrolled ODE trajectories
            tanh=False
        )

        # DummyDecoder extracts the first 3 dimensions (coordinates)
        self.decoder = DummyDecoder(n_atoms=n_atoms, hidden_dim=node_dim)

    def forward(self, node_features, edge_index, edge_features, lengths):
        # We just pass the 6D (pos+vel) node features as the rolling state
        return node_features

    def transition_step(self, z):
        if z.dim() == 2:
            z = z.unsqueeze(0)
            is_2d = True
        else:
            is_2d = False

        B_steps, n_atoms, node_dim = z.shape
        x_t = z[:, :, :3]
        v_t = z[:, :, 3:6]

        # Flatten for SEGNO
        x_flat = x_t.reshape(-1, 3)
        v_flat = v_t.reshape(-1, 3)
        his_flat = z.reshape(-1, node_dim)

        # Batch edges
        E = self.edge_index.size(1)
        batch_offsets = torch.arange(B_steps, device=z.device).repeat_interleave(E) * n_atoms
        edge_idx_batched = self.edge_index.repeat(1, B_steps) + batch_offsets

        # Compute dynamic bond distances as edge_attr
        src, dst = self.edge_index[0], self.edge_index[1]
        distances = torch.norm(x_t[:, src] - x_t[:, dst], dim=-1, keepdim=True) # [B_steps, E, 1]
        edge_attr_batched = distances.reshape(-1, 1)

        x_next_flat = self.model(his_flat, x_flat, edge_idx_batched, v_flat, edge_attr=edge_attr_batched)

        x_next = x_next_flat.reshape(B_steps, n_atoms, 3)
        v_next = x_next - x_t  # Approximate next velocity

        z_next = torch.cat([x_next, v_next], dim=-1)
        if is_2d: z_next = z_next.squeeze(0)
        return z_next

    def forward_rollout(self, z0, steps=100, latent_seed=True):
        if latent_seed:
            z = z0[:, 0]
        else:
            raise NotImplementedError()
        rollout = [z]
        for _ in range(1, steps):
            z = self.transition_step(z)
            rollout.append(z)
        return torch.stack(rollout, dim=1)

    def compute_loss(self, outputs, targets, lengths, epoch, node_features=None):
        z_seq = outputs
        B, T, n_atoms, node_dim = z_seq.shape

        z_t_list, z_tgt_list = [], []
        for b, L in enumerate(lengths):
            if L < 2: continue
            z_t_list.append(z_seq[b, :L-1])
            z_tgt_list.append(z_seq[b, 1:L])

        z_t = torch.cat(z_t_list, dim=0)
        z_tgt = torch.cat(z_tgt_list, dim=0)

        z_pred = self.transition_step(z_t)
        l_dyn = F.mse_loss(z_pred, z_tgt)

        if node_features is not None:
            coords_true = node_features[:, :, :, :3]
            z_flat = z_seq.reshape(B, T, n_atoms * node_dim)
            coords_pred = self.decoder(z_flat)
            l_recon = F.mse_loss(coords_pred, coords_true)

            src_idx, dst_idx = self.edge_index[0], self.edge_index[1]
            d_true = torch.norm(coords_true[:, :, src_idx] - coords_true[:, :, dst_idx], dim=-1)
            d_pred = torch.norm(coords_pred[:, :, src_idx] - coords_pred[:, :, dst_idx], dim=-1)
            l_iso = F.mse_loss(d_pred, d_true)
        else:
            l_recon = 0.0
            l_iso = 0.0

        total_loss = l_dyn + 10.0 * l_recon + 5.0 * l_iso
        return total_loss, {
            'loss': total_loss.item(),
            'l_dyn': l_dyn.item(),
            'l_recon': l_recon.item() if node_features is not None else 0.0,
            'l_iso': l_iso.item() if node_features is not None else 0.0,
        }

    def post_step_hook(self):
        pass
