import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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

from koopman_evolver.data.dataset_split import GraphDatasetSplit


class GraphTrainer:
    """
    OBSOLETE: Legacy Trainer for Experiment 3 models (GraphKoopmanNet and GraphGRUNet).
    Superseded by `GraphAwareTrainer` which actively monitors and minimizes the structural
    `alpha` tradeoff parameter during training to preserve physical constraints.
    """
    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dir,
        checkpoint_name="best.pt",
        epochs=100,
        batch_size=16,
        device="cpu",
        log_every=10,
    ):
        self.model           = model.to(device)
        self.optimizer       = optimizer
        self.checkpoint_dir  = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.device          = device
        self.log_every       = log_every
        os.makedirs(checkpoint_dir, exist_ok=True)
    def fit(self, train_split: GraphDatasetSplit, val_split: GraphDatasetSplit):
        node_features_train = torch.tensor(train_split.node_features, dtype=torch.float32, device=self.device)
        edge_features_train = torch.tensor(train_split.edge_features, dtype=torch.float32, device=self.device)
        edge_index_train    = train_split.edge_index.to(self.device)
        train_lengths       = train_split.lengths
        node_features_val = torch.tensor(val_split.node_features, dtype=torch.float32, device=self.device)
        edge_features_val = torch.tensor(val_split.edge_features, dtype=torch.float32, device=self.device)
        edge_index_val    = val_split.edge_index.to(self.device)
        val_lengths       = val_split.lengths
        N = node_features_train.shape[0]
        best_r2   = -1e9
        best_info = {'epoch': -1, 'val_r2': -1e9}
        for epoch in range(1, self.epochs + 1):
            avg_log = self._train_epoch(
                node_features_train, edge_index_train, edge_features_train, train_lengths, N, epoch
            )
            val_r2 = self._compute_val_r2(
                node_features_val, edge_index_val, edge_features_val, val_lengths
            )
            # Save best checkpoint (ignoring collapsed epochs returning -1.0)
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_info = {
                    'epoch':  epoch,
                    'val_r2': val_r2,
                }
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_r2':           val_r2,
                }, os.path.join(self.checkpoint_dir, self.checkpoint_name))
            if epoch % self.log_every == 0 or epoch == 1:
                best_str = '\u2190 BEST' if best_info.get('epoch') == epoch else ''
                print(
                    f"Epoch {epoch:>3d}/{self.epochs} | "
                    f"Loss {avg_log.get('loss', 0):.4f} | "
                    f"l_dyn {avg_log.get('l_dyn', 0):.4f} | "
                    f"l_recon {avg_log.get('l_recon', 0):.4f} | "
                    f"Val R\u00b2 {val_r2:.4f}  {best_str}"
                )
        print(f"\nBest \u2192 epoch {best_info['epoch']}, Val R\u00b2 = {best_info['val_r2']:.4f}")
        return best_info
    def _train_epoch(self, node_feats, edge_idx, edge_feats, lengths, N, epoch):
        self.model.train()
        perm = torch.randperm(N)
        total_log   = {}
        num_batches = 0
        for start in range(0, N, self.batch_size):
            idx = perm[start : start + self.batch_size]
            batch_nodes = node_feats[idx]
            batch_edges = edge_feats[idx]
            batch_lens  = [lengths[i] for i in idx.tolist()]
            self.optimizer.zero_grad()
            h_seq = self.model(batch_nodes, edge_idx, batch_edges, batch_lens)
            loss, log = self.model.compute_loss(
                h_seq, targets=None, lengths=batch_lens, epoch=epoch, node_features=batch_nodes
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.model.post_step_hook()
            for k, v in log.items():
                total_log[k] = total_log.get(k, 0.0) + v
            num_batches += 1
        return {k: v / num_batches for k, v in total_log.items()}
    @torch.no_grad()
    def _compute_val_r2(self, node_feats, edge_idx, edge_feats, val_lengths):
        self.model.eval()
        h_seq = self.model(node_feats, edge_idx, edge_feats, val_lengths)
        B, T, n_atoms, hidden_dim = h_seq.shape
        # Reject checkpoints if validation latent space collapsed
        all_ratios = []
        for b, L in enumerate(val_lengths):
            if L < 2: continue
            h_seq_b = h_seq[b, :L]
            h_seq_b_flat = h_seq_b.reshape(L, n_atoms * hidden_dim)
            diffs = (h_seq_b_flat[1:] - h_seq_b_flat[:-1]).norm(dim=1)
            norms = h_seq_b_flat[:-1].norm(dim=1) + 1e-8
            ratios = (diffs / norms).tolist()
            all_ratios.extend(ratios)

        mean_relative_change = np.mean(all_ratios) if all_ratios else 0.0
        if mean_relative_change < 0.05:
            return -1.0
        h_t_list, h_tgt_list = [], []
        for b, L in enumerate(val_lengths):
            if L < 2: continue
            h_t_list.append(h_seq[b, :L-1])
            h_tgt_list.append(h_seq[b, 1:L])
        h_t   = torch.cat(h_t_list,   dim=0).reshape(-1, hidden_dim)
        h_tgt = torch.cat(h_tgt_list, dim=0).reshape(-1, hidden_dim)
        if hasattr(self.model, 'K'):
            K_node = self.model.K
            h_pred = torch.matmul(h_t, K_node.t())
        else:
            h_pred = self.model.rnn_transition(h_t, h_t)
        y_true = h_tgt.cpu().numpy()
        y_pred = h_pred.cpu().numpy()
        r2 = r2_score(y_true.ravel(), y_pred.ravel())
        self.model.train()
        return r2


class GraphAwareTrainer:
    """
    Trainer for Graph dynamics models (GraphAwareKoopmanNet and GraphAwareGRUNet).
    Guards checkpoints against trivial latent collapse.
    """
    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dir,
        checkpoint_name="best.pt",
        epochs=100,
        batch_size=16,
        device="cpu",
        log_every=10,
    ):
        self.model           = model.to(device)
        self.optimizer       = optimizer
        self.checkpoint_dir  = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.device          = device
        self.log_every       = log_every
        os.makedirs(checkpoint_dir, exist_ok=True)

    def fit(self, train_split: GraphDatasetSplit, val_split: GraphDatasetSplit):
        node_features_train = torch.tensor(train_split.node_features, dtype=torch.float32, device=self.device)
        edge_features_train = torch.tensor(train_split.edge_features, dtype=torch.float32, device=self.device)
        edge_index_train    = train_split.edge_index.to(self.device)
        train_lengths       = train_split.lengths

        node_features_val = torch.tensor(val_split.node_features, dtype=torch.float32, device=self.device)
        edge_features_val = torch.tensor(val_split.edge_features, dtype=torch.float32, device=self.device)
        edge_index_val    = val_split.edge_index.to(self.device)
        val_lengths       = val_split.lengths

        N = node_features_train.shape[0]
        best_r2   = -1e9
        best_info = {'epoch': -1, 'val_r2': -1e9}

        for epoch in range(1, self.epochs + 1):
            avg_log = self._train_epoch(
                node_features_train, edge_index_train, edge_features_train, train_lengths, N, epoch
            )
            val_r2 = self._compute_val_r2(
                node_features_val, edge_index_val, edge_features_val, val_lengths
            )
            # Save best checkpoint (ignoring collapsed epochs returning -1.0)
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_info = {
                    'epoch':  epoch,
                    'val_r2': val_r2,
                }
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_r2':           val_r2,
                }, os.path.join(self.checkpoint_dir, self.checkpoint_name))

            if epoch % self.log_every == 0 or epoch == 1:
                alpha_str = f" | alpha {avg_log.get('alpha', 0):.4f}" if 'alpha' in avg_log else ""
                best_str = '\u2190 BEST' if best_info.get('epoch') == epoch else ''
                print(
                    f"Epoch {epoch:>3d}/{self.epochs} | "
                    f"Loss {avg_log.get('loss', 0):.4f} | "
                    f"l_dyn {avg_log.get('l_dyn', 0):.4f} | "
                    f"l_recon {avg_log.get('l_recon', 0):.4f}{alpha_str} | "
                    f"Val R\u00b2 {val_r2:.4f}  {best_str}"
                )
        print(f"\nBest \u2192 epoch {best_info['epoch']}, Val R\u00b2 = {best_info['val_r2']:.4f}")
        return best_info

    def _train_epoch(self, node_feats, edge_idx, edge_feats, lengths, N, epoch):
        self.model.train()
        perm = torch.randperm(N)
        total_log   = {}
        num_batches = 0
        for start in range(0, N, self.batch_size):
            idx = perm[start : start + self.batch_size]
            batch_nodes = node_feats[idx]
            batch_edges = edge_feats[idx]
            batch_lens  = [lengths[i] for i in idx.tolist()]
            self.optimizer.zero_grad()
            h_seq = self.model(batch_nodes, edge_idx, batch_edges, batch_lens)
            loss, log = self.model.compute_loss(
                h_seq, targets=None, lengths=batch_lens, epoch=epoch, node_features=batch_nodes
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.model.post_step_hook()
            for k, v in log.items():
                total_log[k] = total_log.get(k, 0.0) + v
            num_batches += 1
        return {k: v / num_batches for k, v in total_log.items()}

    @torch.no_grad()
    def _compute_val_r2(self, node_feats, edge_idx, edge_feats, val_lengths):
        self.model.eval()
        
        all_h_seq = []
        B_total = node_feats.shape[0]
        batch_size = self.batch_size
        
        for i in range(0, B_total, batch_size):
            end_idx = min(i + batch_size, B_total)
            batch_nodes = node_feats[i:end_idx]
            batch_edges = edge_feats[i:end_idx]
            batch_lens = val_lengths[i:end_idx]
            h_seq_b = self.model(batch_nodes, edge_idx, batch_edges, batch_lens)
            all_h_seq.append(h_seq_b)
            
        h_seq = torch.cat(all_h_seq, dim=0)
        B, T, n_atoms, hidden_dim = h_seq.shape
        # Reject checkpoints if validation latent space collapsed
        all_ratios = []
        for b, L in enumerate(val_lengths):
            if L < 2: continue
            h_seq_b = h_seq[b, :L]
            h_seq_b_flat = h_seq_b.reshape(L, n_atoms * hidden_dim)
            diffs = (h_seq_b_flat[1:] - h_seq_b_flat[:-1]).norm(dim=1)
            norms = h_seq_b_flat[:-1].norm(dim=1) + 1e-8
            ratios = (diffs / norms).tolist()
            all_ratios.extend(ratios)

        mean_relative_change = np.mean(all_ratios) if all_ratios else 0.0
        if mean_relative_change < 0.05:
            return -1.0
        h_t_list, h_tgt_list = [], []
        for b, L in enumerate(val_lengths):
            if L < 2: continue
            h_t_list.append(h_seq[b, :L-1])
            h_tgt_list.append(h_seq[b, 1:L])
        h_t   = torch.cat(h_t_list,   dim=0)
        h_tgt = torch.cat(h_tgt_list, dim=0)
        
        h_pred_list = []
        # Use a larger batch size here since it's just the transition step, not the full graph conv
        step_batch = self.batch_size * 4 
        for i in range(0, h_t.shape[0], step_batch):
            end_idx = min(i + step_batch, h_t.shape[0])
            h_pred_list.append(self.model.transition_step(h_t[i:end_idx]))
            
        h_pred = torch.cat(h_pred_list, dim=0)
        
        y_true = h_tgt.cpu().numpy()
        y_pred = h_pred.cpu().numpy()
        r2 = r2_score(y_true.ravel(), y_pred.ravel())
        self.model.train()
        return r2


