from koopman_evolver.data.dataset_split import GraphDatasetSplit
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
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


@dataclass
class GraphEvalResults:
    """
    Standard evaluation results container for graph state models.
    """
    koopman_mse_mean : np.ndarray
    koopman_mse_std  : np.ndarray
    baseline_mse_mean: np.ndarray
    baseline_mse_std : np.ndarray

    rho_koopman      : float

    koopman_geom_mean : np.ndarray
    koopman_geom_std  : np.ndarray
    baseline_geom_mean: np.ndarray
    baseline_geom_std : np.ndarray

    koopman_node_geom_mean : np.ndarray
    koopman_node_geom_std  : np.ndarray
    baseline_node_geom_mean: np.ndarray
    baseline_node_geom_std : np.ndarray

    relative_change_koopman : float
    relative_change_baseline: float

    meta: dict = field(default_factory=dict)


class GraphKoopmanEvaluator:
    """
    Evaluator for Graph dynamics models. Computes rollout MSE, spectral radius,
    and inter-atom geometry retention for both coordinate and node embedding spaces.
    """
    def __init__(
        self,
        koopman_model  : nn.Module,
        baseline_model : nn.Module,
        device         : str = 'cpu',
        rollout_steps  : int = 29,
        batch_size     : int = 64,
        n_atoms        : int = 9,
        hidden_dim     : int = 64
    ):
        self.koopman_model  = koopman_model.to(device).eval()
        self.baseline_model = baseline_model.to(device).eval()
        self.device         = device
        self.rollout_steps  = rollout_steps
        self.batch_size     = batch_size
        self.n_atoms        = n_atoms
        self.hidden_dim     = hidden_dim

    def run(self, split: GraphDatasetSplit) -> GraphEvalResults:
        node_feats = torch.tensor(split.node_features, dtype=torch.float32, device=self.device)
        edge_feats = torch.tensor(split.edge_features, dtype=torch.float32, device=self.device)
        edge_idx   = split.edge_index.to(self.device)
        lengths    = split.lengths

        with torch.no_grad():
            z_koop = self._encode_all(self.koopman_model, node_feats, edge_idx, edge_feats, lengths)
            z_base = self._encode_all(self.baseline_model, node_feats, edge_idx, edge_feats, lengths)

            koop_mse_mean, koop_mse_std = self._rollout_mse(self.koopman_model, z_koop, lengths)
            base_mse_mean, base_mse_std = self._rollout_mse(self.baseline_model, z_base, lengths)

            rho_koopman = self._spectral_radii(self.koopman_model)

            # Coordinate geometry retention (physical geometry ratio)
            koop_geom_mean, koop_geom_std = self._coordinate_geometry_retention(self.koopman_model, z_koop, lengths)
            base_geom_mean, base_geom_std = self._coordinate_geometry_retention(self.baseline_model, z_base, lengths)

            # Node embedding geometry retention (embedding geometry ratio)
            k_node_geom_mean, k_node_geom_std = self._node_geometry_retention(self.koopman_model, z_koop, lengths)
            b_node_geom_mean, b_node_geom_std = self._node_geometry_retention(self.baseline_model, z_base, lengths)

            rc_koop = self._relative_change(z_koop, lengths)
            rc_base = self._relative_change(z_base, lengths)

        return GraphEvalResults(
            koopman_mse_mean        = koop_mse_mean,
            koopman_mse_std         = koop_mse_std,
            baseline_mse_mean       = base_mse_mean,
            baseline_mse_std        = base_mse_std,
            rho_koopman             = rho_koopman,
            koopman_geom_mean       = koop_geom_mean,
            koopman_geom_std        = koop_geom_std,
            baseline_geom_mean      = base_geom_mean,
            baseline_geom_std       = base_geom_std,
            koopman_node_geom_mean  = k_node_geom_mean,
            koopman_node_geom_std   = k_node_geom_std,
            baseline_node_geom_mean = b_node_geom_mean,
            baseline_node_geom_std  = b_node_geom_std,
            relative_change_koopman = rc_koop,
            relative_change_baseline= rc_base,
            meta                    = split.meta,
        )

    def _encode_all(self, model, node_feats, edge_idx, edge_feats, lengths):
        N = node_feats.shape[0]
        z_list = []
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch_nodes = node_feats[start:end]
            batch_edges = edge_feats[start:end]
            batch_lens  = lengths[start:end]
            z_batch     = model(batch_nodes, edge_idx, batch_edges, batch_lens)
            z_list.append(z_batch)
        return torch.cat(z_list, dim=0)

    def _rollout_mse(self, model, z, lengths):
        z_cpu = z.cpu()
        mse_per_step_mean = np.zeros(self.rollout_steps)
        mse_per_step_std  = np.zeros(self.rollout_steps)

        if not hasattr(model, 'K'):
            for s in range(1, self.rollout_steps + 1):
                errors = []
                for b, T in enumerate(lengths):
                    if T <= s + 1:
                        continue
                    seed = z_cpu[b:b+1, :1, :, :]
                    preds = model.forward_rollout(
                        seed.to(self.device), steps=s + 1, latent_seed=True
                    ).cpu()
                    z_pred   = preds[0, s]
                    z_target = z_cpu[b, s]
                    errors.append(F.mse_loss(z_pred, z_target).item())
                mse_per_step_mean[s - 1] = np.mean(errors) if errors else 0.0
                mse_per_step_std[s - 1]  = np.std(errors)  if errors else 0.0
        else:
            K = model.K.detach().cpu()
            for s in range(1, self.rollout_steps + 1):
                K_pow = torch.matrix_power(K, s)
                errors = []
                for b, T in enumerate(lengths):
                    if T <= s + 1:
                        continue
                    z_init   = z_cpu[b, :T - s]  # (T-s, 9, 64)
                    z_target = z_cpu[b, s:T]     # (T-s, 9, 64)
                    z_pred   = torch.matmul(z_init, K_pow.t())
                    errors.append(F.mse_loss(z_pred, z_target).item())
                mse_per_step_mean[s - 1] = np.mean(errors) if errors else 0.0
                mse_per_step_std[s - 1]  = np.std(errors)  if errors else 0.0

        return mse_per_step_mean, mse_per_step_std

    def _spectral_radii(self, model):
        if not hasattr(model, 'K'):
            return 0.0
        K = model.K.detach().cpu().numpy()
        eigvals = np.linalg.eigvals(K)
        return float(np.max(np.abs(eigvals)))

    def _coordinate_geometry_retention(self, model, z, lengths):
        N = len(lengths)
        geom_means = np.zeros(self.rollout_steps + 1)
        geom_stds  = np.zeros(self.rollout_steps + 1)

        with torch.no_grad():
            z_all_0 = z.to(self.device)[:, :1, :, :]
            rollout_all = model.forward_rollout(z_all_0, steps=self.rollout_steps + 1, latent_seed=True)

            B, steps, n_atoms, h_dim = rollout_all.shape
            rollout_all_flat = rollout_all.reshape(B * steps, n_atoms * h_dim)
            coords_all_flat = model.decoder(rollout_all_flat)
            coords_all = coords_all_flat.reshape(B, steps, n_atoms, 3).cpu()

        for s in range(self.rollout_steps + 1):
            ratios = []
            for b in range(N):
                if lengths[b] <= s:
                    continue
                X0 = coords_all[b, 0]
                Xs = coords_all[b, s]
                D0 = torch.norm(X0.unsqueeze(1) - X0.unsqueeze(0), dim=-1)
                Ds = torch.norm(Xs.unsqueeze(1) - Xs.unsqueeze(0), dim=-1)
                norm_D0 = torch.norm(D0).item()
                if norm_D0 < 1e-8:
                    continue
                ratio = torch.norm(Ds).item() / norm_D0
                ratios.append(ratio)
            geom_means[s] = np.mean(ratios) if ratios else 0.0
            geom_stds[s]  = np.std(ratios)  if ratios else 0.0

        return geom_means, geom_stds

    def _node_geometry_retention(self, model, z, lengths):
        N = len(lengths)
        geom_means = np.zeros(self.rollout_steps + 1)
        geom_stds  = np.zeros(self.rollout_steps + 1)

        with torch.no_grad():
            z_all_0 = z.to(self.device)[:, :1, :, :]
            rollout_all = model.forward_rollout(z_all_0, steps=self.rollout_steps + 1, latent_seed=True).cpu()

        for s in range(self.rollout_steps + 1):
            ratios = []
            for b in range(N):
                if lengths[b] <= s:
                    continue
                H0 = rollout_all[b, 0]
                Hs = rollout_all[b, s]
                D0 = torch.norm(H0.unsqueeze(1) - H0.unsqueeze(0), dim=-1)
                Ds = torch.norm(Hs.unsqueeze(1) - Hs.unsqueeze(0), dim=-1)
                norm_D0 = torch.norm(D0).item()
                if norm_D0 < 1e-8:
                    continue
                ratio = torch.norm(Ds).item() / norm_D0
                ratios.append(ratio)
            geom_means[s] = np.mean(ratios) if ratios else 0.0
            geom_stds[s]  = np.std(ratios)  if ratios else 0.0

        return geom_means, geom_stds

    def _relative_change(self, z, lengths):
        z_cpu = z.cpu()
        all_ratios = []
        for b, T in enumerate(lengths):
            if T < 2:
                continue
            z_seq = z_cpu[b, :T]
            z_seq_flat = z_seq.reshape(T, -1)
            diffs = (z_seq_flat[1:] - z_seq_flat[:-1]).norm(dim=1)
            norms = z_seq_flat[:-1].norm(dim=1)
            ratios = (diffs / (norms + 1e-6)).tolist()
            all_ratios.extend(ratios)
        return float(np.mean(all_ratios)) if all_ratios else 0.0

    def print_summary(self, results: GraphEvalResults):
        sep = "=" * 70
        print(sep)
        print(f"EVALUATION SUMMARY — {results.meta.get('dataset','?')} {results.meta.get('molecule','')}")
        print(sep)
        print("\n[1] ROLLOUT MSE")
        print(f"  {'Step':>4}  {'Koopman':>12}  {'Baseline':>12}")
        for s in range(len(results.koopman_mse_mean)):
            print(f"  {s+1:>4}  {results.koopman_mse_mean[s]:>12.4e}  {results.baseline_mse_mean[s]:>12.4e}")

        print("\n[2] SPECTRAL RADIUS")
        print(f"  rho(K)      : {results.rho_koopman:.6f}  "
              f"{'PASS (conservative)' if abs(results.rho_koopman - 1.0) < 1e-3 else 'WARN'}")

        print("\n[3] GEOMETRY RETENTION @ final step")
        print(f"  Koopman Coordinate Retention : {results.koopman_geom_mean[-1]:.4f}")
        print(f"  Baseline Coordinate Retention: {results.baseline_geom_mean[-1]:.4f}")
        print(f"  Koopman Node Emb. Retention  : {results.koopman_node_geom_mean[-1]:.4f}")
        print(f"  Baseline Node Emb. Retention : {results.baseline_node_geom_mean[-1]:.4f}")

        print("\n[4] COLLAPSE DIAGNOSTIC")
        print(f"  Koopman  relative change: {results.relative_change_koopman:.4f}  "
              f"{'PASS' if results.relative_change_koopman > 0.05 else 'FAIL'}")
        print(f"  Baseline relative change: {results.relative_change_baseline:.4f}")
        print(sep)

    def plot(self, results: GraphEvalResults, title: Optional[str] = None, save_path: Optional[str] = None):
        KOOP_COLOR = "#2166ac"
        BASE_COLOR = "#d6604d"

        steps_err  = np.arange(1, self.rollout_steps + 1)
        steps_geom = np.arange(0, self.rollout_steps + 1)

        k0 = results.koopman_geom_mean[0] + 1e-8
        b0 = results.baseline_geom_mean[0] + 1e-8
        koop_geom_norm = results.koopman_geom_mean  / k0
        base_geom_norm = results.baseline_geom_mean / b0
        koop_gstd_norm = results.koopman_geom_std   / k0
        base_gstd_norm = results.baseline_geom_std  / b0

        k_node0 = results.koopman_node_geom_mean[0] + 1e-8
        b_node0 = results.baseline_node_geom_mean[0] + 1e-8
        koop_node_norm = results.koopman_node_geom_mean / k_node0
        base_node_norm = results.baseline_node_geom_mean / b_node0
        koop_nstd_norm = results.koopman_node_geom_std / k_node0
        base_nstd_norm = results.baseline_node_geom_std / b_node0

        dataset_label = (title or f"{results.meta.get('dataset','?')} {results.meta.get('molecule','')}")

        fig = plt.figure(figsize=(13, 11))
        gs  = gridspec.GridSpec(3, 1, hspace=0.35)

        # Top: MSE
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps_err, results.koopman_mse_mean, color=KOOP_COLOR, linewidth=2.5, marker='o', markersize=4, label='Koopman (Node-Level)')
        ax1.fill_between(steps_err, results.koopman_mse_mean - results.koopman_mse_std, results.koopman_mse_mean + results.koopman_mse_std, color=KOOP_COLOR, alpha=0.15)
        ax1.plot(steps_err, results.baseline_mse_mean, color=BASE_COLOR, linewidth=2.5, linestyle='--', marker='x', markersize=5, label='GRU Baseline')
        ax1.fill_between(steps_err, results.baseline_mse_mean - results.baseline_mse_std, results.baseline_mse_mean + results.baseline_mse_std, color=BASE_COLOR, alpha=0.15)
        ax1.set_ylabel("Rollout MSE", fontsize=12)
        ax1.set_title(f"Predictive Accuracy vs. Geometric Stability — {dataset_label}", fontsize=13, fontweight='bold', pad=10)
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.5)

        # Middle: Coordinate Geometry Retention
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(steps_geom, koop_geom_norm, color=KOOP_COLOR, linewidth=2.5, marker='o', markersize=4)
        ax2.fill_between(steps_geom, koop_geom_norm - koop_gstd_norm, koop_geom_norm + koop_gstd_norm, color=KOOP_COLOR, alpha=0.15)
        ax2.plot(steps_geom, base_geom_norm, color=BASE_COLOR, linewidth=2.5, linestyle='--', marker='x', markersize=5)
        ax2.fill_between(steps_geom, base_geom_norm - base_gstd_norm, base_geom_norm + base_gstd_norm, color=BASE_COLOR, alpha=0.15)
        ax2.axhline(1.0, color='gray', linewidth=1.0, linestyle=':', alpha=0.7, label='Perfect retention (ratio = 1.0)')
        ax2.set_ylabel("Pairwise Coordinate Distance Ratio\n(normalized to t=0)", fontsize=12)
        ax2.legend(fontsize=10, loc='lower left')
        ax2.grid(True, linestyle=':', alpha=0.5)

        # Bottom: Node Embedding Geometry Retention
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(steps_geom, koop_node_norm, color=KOOP_COLOR, linewidth=2.5, marker='o', markersize=4, label='Koopman (Node-Level)')
        ax3.fill_between(steps_geom, koop_node_norm - koop_nstd_norm, koop_node_norm + koop_nstd_norm, color=KOOP_COLOR, alpha=0.15)
        ax3.plot(steps_geom, base_node_norm, color=BASE_COLOR, linewidth=2.5, linestyle='--', marker='x', markersize=5, label='GRU Baseline')
        ax3.fill_between(steps_geom, base_node_norm - base_nstd_norm, base_node_norm + base_nstd_norm, color=BASE_COLOR, alpha=0.15)
        ax3.axhline(1.0, color='gray', linewidth=1.0, linestyle=':', alpha=0.7)
        ax3.set_xlabel("Prediction Horizon (steps)", fontsize=12)
        ax3.set_ylabel("Node Embedding Distance Ratio\n(normalized to t=0)", fontsize=12)
        ax3.legend(fontsize=10, loc='lower left')
        ax3.grid(True, linestyle=':', alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


@dataclass
class GraphAwareEvalResults:
    """
    Evaluation results container for graph-aware dynamics.
    """
    koopman_mse_mean : np.ndarray
    koopman_mse_std  : np.ndarray
    baseline_mse_mean: np.ndarray
    baseline_mse_std : np.ndarray

    rho_koopman      : float

    koopman_geom_mean : np.ndarray
    koopman_geom_std  : np.ndarray
    baseline_geom_mean: np.ndarray
    baseline_geom_std : np.ndarray

    koopman_node_geom_mean : np.ndarray
    koopman_node_geom_std  : np.ndarray
    baseline_node_geom_mean: np.ndarray
    baseline_node_geom_std : np.ndarray

    koopman_energy_mean : np.ndarray
    koopman_energy_std  : np.ndarray
    baseline_energy_mean: np.ndarray
    baseline_energy_std : np.ndarray

    relative_change_koopman : float
    relative_change_baseline: float

    meta: dict = field(default_factory=dict)


class GraphAwareKoopmanEvaluator:
    """
    Evaluator for Graph-Aware dynamics models. Computes rollout MSE, spectral radius of the Kronecker
    operator K_global, dual geometry retention ratios, and rollout Graph Energy.
    """
    def __init__(
        self,
        koopman_model  : nn.Module,
        baseline_model : nn.Module,
        device         : str = 'cpu',
        rollout_steps  : int = 29,
        batch_size     : int = 64,
        n_atoms        : int = 9,
        hidden_dim     : int = 64
    ):
        self.koopman_model  = koopman_model.to(device).eval()
        self.baseline_model = baseline_model.to(device).eval()
        self.device         = device
        self.rollout_steps  = rollout_steps
        self.batch_size     = batch_size
        self.n_atoms        = n_atoms
        self.hidden_dim     = hidden_dim

    def run(self, split: GraphDatasetSplit) -> GraphAwareEvalResults:
        node_feats = torch.tensor(split.node_features, dtype=torch.float32, device=self.device)
        edge_feats = torch.tensor(split.edge_features, dtype=torch.float32, device=self.device)
        edge_idx   = split.edge_index.to(self.device)
        lengths    = split.lengths

        with torch.no_grad():
            z_koop = self._encode_all(self.koopman_model, node_feats, edge_idx, edge_feats, lengths)
            z_base = self._encode_all(self.baseline_model, node_feats, edge_idx, edge_feats, lengths)

            koop_mse_mean, koop_mse_std = self._rollout_mse(self.koopman_model, z_koop, lengths)
            base_mse_mean, base_mse_std = self._rollout_mse(self.baseline_model, z_base, lengths)

            rho_koopman = self._spectral_radii(self.koopman_model)

            # Coordinate geometry retention (physical geometry ratio)
            koop_geom_mean, koop_geom_std = self._coordinate_geometry_retention(self.koopman_model, z_koop, lengths)
            base_geom_mean, base_geom_std = self._coordinate_geometry_retention(self.baseline_model, z_base, lengths)

            # Node embedding geometry retention (embedding geometry ratio)
            k_node_geom_mean, k_node_geom_std = self._node_geometry_retention(self.koopman_model, z_koop, lengths)
            b_node_geom_mean, b_node_geom_std = self._node_geometry_retention(self.baseline_model, z_base, lengths)

            # Graph energy ratio
            k_energy_mean, k_energy_std = self._graph_energy_retention(self.koopman_model, z_koop, lengths)
            b_energy_mean, b_energy_std = self._graph_energy_retention(self.baseline_model, z_base, lengths)

            rc_koop = self._relative_change(z_koop, lengths)
            rc_base = self._relative_change(z_base, lengths)

        return GraphAwareEvalResults(
            koopman_mse_mean        = koop_mse_mean,
            koopman_mse_std         = koop_mse_std,
            baseline_mse_mean       = base_mse_mean,
            baseline_mse_std        = base_mse_std,
            rho_koopman             = rho_koopman,
            koopman_geom_mean       = koop_geom_mean,
            koopman_geom_std        = koop_geom_std,
            baseline_geom_mean      = base_geom_mean,
            baseline_geom_std       = base_geom_std,
            koopman_node_geom_mean  = k_node_geom_mean,
            koopman_node_geom_std   = k_node_geom_std,
            baseline_node_geom_mean = b_node_geom_mean,
            baseline_node_geom_std  = b_node_geom_std,
            koopman_energy_mean     = k_energy_mean,
            koopman_energy_std      = k_energy_std,
            baseline_energy_mean    = b_energy_mean,
            baseline_energy_std     = b_energy_std,
            relative_change_koopman = rc_koop,
            relative_change_baseline= rc_base,
            meta                    = split.meta,
        )

    def _encode_all(self, model, node_feats, edge_idx, edge_feats, lengths):
        N = node_feats.shape[0]
        z_list = []
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch_nodes = node_feats[start:end]
            batch_edges = edge_feats[start:end]
            batch_lens  = lengths[start:end]
            z_batch     = model(batch_nodes, edge_idx, batch_edges, batch_lens)
            z_list.append(z_batch)
        return torch.cat(z_list, dim=0)

    def _rollout_mse(self, model, z, lengths):
        z_cpu = z.cpu()
        mse_per_step_mean = np.zeros(self.rollout_steps)
        mse_per_step_std  = np.zeros(self.rollout_steps)

        if not hasattr(model, 'get_global_K'):
            # GRU model (calls forward_rollout starting from frame 0)
            for s in range(1, self.rollout_steps + 1):
                errors = []
                for b, T in enumerate(lengths):
                    if T <= s + 1:
                        continue
                    seed = z_cpu[b:b+1, :1, :, :]
                    preds = model.forward_rollout(
                        seed.to(self.device), steps=s + 1, latent_seed=True
                    ).cpu()
                    z_pred   = preds[0, s]
                    z_target = z_cpu[b, s]
                    errors.append(F.mse_loss(z_pred, z_target).item())
                mse_per_step_mean[s - 1] = np.mean(errors) if errors else 0.0
                mse_per_step_std[s - 1]  = np.std(errors)  if errors else 0.0
        else:
            # Koopman model (calls transition_step across all sequence frames in parallel)
            for s in range(1, self.rollout_steps + 1):
                errors = []
                for b, T in enumerate(lengths):
                    if T <= s + 1:
                        continue
                    z_init   = z[b, :T - s].to(self.device)  # (T-s, 9, 64)
                    z_target = z[b, s:T].to(self.device)     # (T-s, 9, 64)

                    h = z_init
                    for _ in range(s):
                        h = model.transition_step(h)
                    errors.append(F.mse_loss(h, z_target).item())
                mse_per_step_mean[s - 1] = np.mean(errors) if errors else 0.0
                mse_per_step_std[s - 1]  = np.std(errors)  if errors else 0.0

        return mse_per_step_mean, mse_per_step_std

    def _spectral_radii(self, model):
        if not hasattr(model, 'get_global_K'):
            return 0.0
        K_global = model.get_global_K()
        eigvals = np.linalg.eigvals(K_global)
        return float(np.max(np.abs(eigvals)))

    def _coordinate_geometry_retention(self, model, z, lengths):
        N = len(lengths)
        geom_means = np.zeros(self.rollout_steps + 1)
        geom_stds  = np.zeros(self.rollout_steps + 1)

        with torch.no_grad():
            z_all_0 = z.to(self.device)[:, :1, :, :]
            rollout_all = model.forward_rollout(z_all_0, steps=self.rollout_steps + 1, latent_seed=True)

            B, steps, n_atoms, h_dim = rollout_all.shape
            rollout_all_flat = rollout_all.reshape(B * steps, n_atoms * h_dim)
            coords_all_flat = model.decoder(rollout_all_flat)
            coords_all = coords_all_flat.reshape(B, steps, n_atoms, 3).cpu()

        for s in range(self.rollout_steps + 1):
            ratios = []
            for b in range(N):
                if lengths[b] <= s:
                    continue
                X0 = coords_all[b, 0]
                Xs = coords_all[b, s]
                D0 = torch.norm(X0.unsqueeze(1) - X0.unsqueeze(0), dim=-1)
                Ds = torch.norm(Xs.unsqueeze(1) - Xs.unsqueeze(0), dim=-1)
                norm_D0 = torch.norm(D0).item()
                if norm_D0 < 1e-8:
                    continue
                ratio = torch.norm(Ds).item() / norm_D0
                ratios.append(ratio)
            geom_means[s] = np.mean(ratios) if ratios else 0.0
            geom_stds[s]  = np.std(ratios)  if ratios else 0.0

        return geom_means, geom_stds

    def _node_geometry_retention(self, model, z, lengths):
        N = len(lengths)
        geom_means = np.zeros(self.rollout_steps + 1)
        geom_stds  = np.zeros(self.rollout_steps + 1)

        with torch.no_grad():
            z_all_0 = z.to(self.device)[:, :1, :, :]
            rollout_all = model.forward_rollout(z_all_0, steps=self.rollout_steps + 1, latent_seed=True).cpu()

        for s in range(self.rollout_steps + 1):
            ratios = []
            for b in range(N):
                if lengths[b] <= s:
                    continue
                H0 = rollout_all[b, 0]
                Hs = rollout_all[b, s]
                D0 = torch.norm(H0.unsqueeze(1) - H0.unsqueeze(0), dim=-1)
                Ds = torch.norm(Hs.unsqueeze(1) - Hs.unsqueeze(0), dim=-1)
                norm_D0 = torch.norm(D0).item()
                if norm_D0 < 1e-8:
                    continue
                ratio = torch.norm(Ds).item() / norm_D0
                ratios.append(ratio)
            geom_means[s] = np.mean(ratios) if ratios else 0.0
            geom_stds[s]  = np.std(ratios)  if ratios else 0.0

        return geom_means, geom_stds

    def _graph_energy_retention(self, model, z, lengths):
        N = len(lengths)
        energy_means = np.zeros(self.rollout_steps + 1)
        energy_stds  = np.zeros(self.rollout_steps + 1)

        with torch.no_grad():
            z_all_0 = z.to(self.device)[:, :1, :, :]
            rollout_all = model.forward_rollout(z_all_0, steps=self.rollout_steps + 1, latent_seed=True).cpu()

        for s in range(self.rollout_steps + 1):
            ratios = []
            for b in range(N):
                if lengths[b] <= s:
                    continue
                H0 = rollout_all[b, 0]
                Hs = rollout_all[b, s]
                E0 = torch.mean(torch.norm(H0, dim=-1)**2).item()
                Es = torch.mean(torch.norm(Hs, dim=-1)**2).item()
                if E0 < 1e-8:
                    continue
                ratios.append(Es / E0)
            energy_means[s] = np.mean(ratios) if ratios else 0.0
            energy_stds[s]  = np.std(ratios)  if ratios else 0.0

        return energy_means, energy_stds

    def _relative_change(self, z, lengths):
        z_cpu = z.cpu()
        all_ratios = []
        for b, T in enumerate(lengths):
            if T < 2:
                continue
            z_seq = z_cpu[b, :T]
            z_seq_flat = z_seq.reshape(T, -1)
            diffs = (z_seq_flat[1:] - z_seq_flat[:-1]).norm(dim=1)
            norms = z_seq_flat[:-1].norm(dim=1)
            ratios = (diffs / (norms + 1e-6)).tolist()
            all_ratios.extend(ratios)
        return float(np.mean(all_ratios)) if all_ratios else 0.0

    def print_summary(self, results: GraphAwareEvalResults):
        sep = "=" * 70
        print(sep)
        print(f"EVALUATION SUMMARY — {results.meta.get('dataset','?')} {results.meta.get('molecule','')}")
        print(sep)
        print("\n[1] ROLLOUT MSE")
        print(f"  {'Step':>4}  {'Koopman':>12}  {'Baseline':>12}")
        for s in range(len(results.koopman_mse_mean)):
            print(f"  {s+1:>4}  {results.koopman_mse_mean[s]:>12.4e}  {results.baseline_mse_mean[s]:>12.4e}")

        print("\n[2] SPECTRAL RADIUS")
        print(f"  rho(K_global): {results.rho_koopman:.6f}  "
              f"{'PASS (conservative)' if abs(results.rho_koopman - 1.0) < 5e-2 else 'WARN'}")

        print("\n[3] GEOMETRY RETENTION @ final step")
        print(f"  Koopman Coordinate Retention : {results.koopman_geom_mean[-1]:.4f}")
        print(f"  Baseline Coordinate Retention: {results.baseline_geom_mean[-1]:.4f}")
        print(f"  Koopman Node Emb. Retention  : {results.koopman_node_geom_mean[-1]:.4f}")
        print(f"  Baseline Node Emb. Retention : {results.baseline_node_geom_mean[-1]:.4f}")

        print("\n[4] GRAPH ENERGY @ final step")
        print(f"  Koopman Graph Energy Ratio   : {results.koopman_energy_mean[-1]:.4f}")
        print(f"  Baseline Graph Energy Ratio  : {results.baseline_energy_mean[-1]:.4f}")

        print("\n[5] COLLAPSE DIAGNOSTIC")
        print(f"  Koopman  relative change: {results.relative_change_koopman:.4f}  "
              f"{'PASS' if results.relative_change_koopman > 0.05 else 'FAIL'}")
        print(f"  Baseline relative change: {results.relative_change_baseline:.4f}")
        print(sep)

    def plot(self, results: GraphAwareEvalResults, title: Optional[str] = None, save_path: Optional[str] = None):
        KOOP_COLOR = "#2166ac"
        BASE_COLOR = "#d6604d"

        steps_err  = np.arange(1, self.rollout_steps + 1)
        steps_geom = np.arange(0, self.rollout_steps + 1)

        k0 = results.koopman_geom_mean[0] + 1e-8
        b0 = results.baseline_geom_mean[0] + 1e-8
        koop_geom_norm = results.koopman_geom_mean  / k0
        base_geom_norm = results.baseline_geom_mean / b0
        koop_gstd_norm = results.koopman_geom_std   / k0
        base_gstd_norm = results.baseline_geom_std  / b0

        k_node0 = results.koopman_node_geom_mean[0] + 1e-8
        b_node0 = results.baseline_node_geom_mean[0] + 1e-8
        koop_node_norm = results.koopman_node_geom_mean / k_node0
        base_node_norm = results.baseline_node_geom_mean / b_node0
        koop_nstd_norm = results.koopman_node_geom_std / k_node0
        base_nstd_norm = results.baseline_node_geom_std / b_node0

        dataset_label = (title or f"{results.meta.get('dataset','?')} {results.meta.get('molecule','')}")

        fig = plt.figure(figsize=(13, 14))
        gs  = gridspec.GridSpec(4, 1, hspace=0.35)

        # 1. MSE
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps_err, results.koopman_mse_mean, color=KOOP_COLOR, linewidth=2.5, marker='o', markersize=4, label='Koopman (Graph-Aware)')
        ax1.fill_between(steps_err, results.koopman_mse_mean - results.koopman_mse_std, results.koopman_mse_mean + results.koopman_mse_std, color=KOOP_COLOR, alpha=0.15)
        ax1.plot(steps_err, results.baseline_mse_mean, color=BASE_COLOR, linewidth=2.5, linestyle='--', marker='x', markersize=5, label='GRU Baseline')
        ax1.fill_between(steps_err, results.baseline_mse_mean - results.baseline_mse_std, results.baseline_mse_mean + results.baseline_mse_std, color=BASE_COLOR, alpha=0.15)
        ax1.set_ylabel("Rollout MSE", fontsize=12)
        ax1.set_title(f"Predictive Accuracy vs. Geometric Stability — {dataset_label}", fontsize=13, fontweight='bold', pad=10)
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.5)

        # 2. Coordinate Geometry Retention
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(steps_geom, koop_geom_norm, color=KOOP_COLOR, linewidth=2.5, marker='o', markersize=4)
        ax2.fill_between(steps_geom, koop_geom_norm - koop_gstd_norm, koop_geom_norm + koop_gstd_norm, color=KOOP_COLOR, alpha=0.15)
        ax2.plot(steps_geom, base_geom_norm, color=BASE_COLOR, linewidth=2.5, linestyle='--', marker='x', markersize=5)
        ax2.fill_between(steps_geom, base_geom_norm - base_gstd_norm, base_geom_norm + base_gstd_norm, color=BASE_COLOR, alpha=0.15)
        ax2.axhline(1.0, color='gray', linewidth=1.0, linestyle=':', alpha=0.7, label='Perfect retention (ratio = 1.0)')
        ax2.set_ylabel("Pairwise Coordinate Distance Ratio\n(normalized to t=0)", fontsize=12)
        ax2.legend(fontsize=10, loc='lower left')
        ax2.grid(True, linestyle=':', alpha=0.5)

        # 3. Node Embedding Geometry Retention
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(steps_geom, koop_node_norm, color=KOOP_COLOR, linewidth=2.5, marker='o', markersize=4, label='Koopman (Graph-Aware)')
        ax3.fill_between(steps_geom, koop_node_norm - koop_nstd_norm, koop_node_norm + koop_nstd_norm, color=KOOP_COLOR, alpha=0.15)
        ax3.plot(steps_geom, base_node_norm, color=BASE_COLOR, linewidth=2.5, linestyle='--', marker='x', markersize=5, label='GRU Baseline')
        ax3.fill_between(steps_geom, base_node_norm - base_nstd_norm, base_node_norm + base_nstd_norm, color=BASE_COLOR, alpha=0.15)
        ax3.axhline(1.0, color='gray', linewidth=1.0, linestyle=':', alpha=0.7)
        ax3.set_ylabel("Node Embedding Distance Ratio\n(normalized to t=0)", fontsize=12)
        ax3.legend(fontsize=10, loc='lower left')
        ax3.grid(True, linestyle=':', alpha=0.5)

        # 4. Graph Energy Retention
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(steps_geom, results.koopman_energy_mean, color=KOOP_COLOR, linewidth=2.5, marker='o', markersize=4, label='Koopman (Graph-Aware)')
        ax4.fill_between(steps_geom, results.koopman_energy_mean - results.koopman_energy_std, results.koopman_energy_mean + results.koopman_energy_std, color=KOOP_COLOR, alpha=0.15)
        ax4.plot(steps_geom, results.baseline_energy_mean, color=BASE_COLOR, linewidth=2.5, linestyle='--', marker='x', markersize=5, label='GRU Baseline')
        ax4.fill_between(steps_geom, results.baseline_energy_mean - results.baseline_energy_std, results.baseline_energy_mean + results.baseline_energy_std, color=BASE_COLOR, alpha=0.15)
        ax4.axhline(1.0, color='gray', linewidth=1.0, linestyle=':', alpha=0.7)
        ax4.set_xlabel("Prediction Horizon (steps)", fontsize=12)
        ax4.set_ylabel("Graph Energy Ratio (Et / E0)\n(normalized to t=0)", fontsize=12)
        ax4.legend(fontsize=10, loc='lower left')
        ax4.grid(True, linestyle=':', alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class PhysicsEval:
    def __init__(self, koop_model, gru_model, test_split, n_atoms, molecule_name):
        self.koop_model = koop_model
        self.gru_model = gru_model
        self.test_split = test_split
        self.n_atoms = n_atoms
        self.molecule_name = molecule_name
        self.device = next(koop_model.parameters()).device

    def extract_topology(self, edge_index):
        G = nx.Graph()
        G.add_nodes_from(range(self.n_atoms))
        edges = edge_index.cpu().T.numpy().tolist()
        G.add_edges_from(edges)
        bonds = list(G.edges())
        angles = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    angles.append((neighbors[i], node, neighbors[j]))
        torsions = []
        for u, v in G.edges():
            for w in G.neighbors(u):
                if w == v: continue
                for x in G.neighbors(v):
                    if x == u or x == w: continue
                    torsions.append((w, u, v, x))
        return bonds, angles, torsions

    def compute_angles(self, coords, angles):
        if not angles: return None
        vals = []
        for i, j, k in angles:
            v1 = coords[..., i, :] - coords[..., j, :]
            v2 = coords[..., k, :] - coords[..., j, :]
            v1_norm = torch.norm(v1, dim=-1)
            v2_norm = torch.norm(v2, dim=-1)
            dot = torch.sum(v1 * v2, dim=-1)
            cos_theta = torch.clamp(dot / (v1_norm * v2_norm + 1e-8), -1.0, 1.0)
            theta = torch.acos(cos_theta) * (180.0 / np.pi)
            vals.append(theta)
        return torch.stack(vals, dim=-1)

    def compute_torsions(self, coords, torsions):
        if not torsions: return None
        vals = []
        for i, j, k, l in torsions:
            b1 = coords[..., j, :] - coords[..., i, :]
            b2 = coords[..., k, :] - coords[..., j, :]
            b3 = coords[..., l, :] - coords[..., k, :]
            n1 = torch.cross(b1, b2, dim=-1)
            n2 = torch.cross(b2, b3, dim=-1)
            n1_norm = torch.norm(n1, dim=-1)
            n2_norm = torch.norm(n2, dim=-1)
            dot = torch.sum(n1 * n2, dim=-1)
            cos_phi = torch.clamp(dot / (n1_norm * n2_norm + 1e-8), -1.0, 1.0)
            phi = torch.acos(cos_phi) * (180.0 / np.pi)
            vals.append(phi)
        return torch.stack(vals, dim=-1)

    def run(self, steps=29, out_dir="./results"):
        self.koop_model.eval()
        self.gru_model.eval()
        bonds, angles, torsions = self.extract_topology(self.test_split.edge_index)
        print(f"[{self.molecule_name}] Extracted {len(bonds)} bonds, {len(angles)} angles, {len(torsions)} torsions.")

        node_feats = torch.tensor(self.test_split.node_features, dtype=torch.float32, device=self.device)
        edge_feats = torch.tensor(self.test_split.edge_features, dtype=torch.float32, device=self.device)
        edge_idx = self.test_split.edge_index.to(self.device)
        lengths = self.test_split.lengths

        with torch.no_grad():
            z_koop = []
            z_gru = []
            for start in range(0, len(node_feats), 64):
                end = min(start + 64, len(node_feats))
                z_koop.append(self.koop_model(node_feats[start:end], edge_idx, edge_feats[start:end], lengths[start:end]))
                z_gru.append(self.gru_model(node_feats[start:end], edge_idx, edge_feats[start:end], lengths[start:end]))
            z_koop = torch.cat(z_koop, dim=0)
            z_gru = torch.cat(z_gru, dim=0)

            z_koop_0 = z_koop[:, :1]
            z_gru_0 = z_gru[:, :1]

            roll_koop = self.koop_model.forward_rollout(z_koop_0, steps=steps+1, latent_seed=True)
            roll_gru = self.gru_model.forward_rollout(z_gru_0, steps=steps+1, latent_seed=True)

            B = roll_koop.shape[0]
            h_dim = 64

            coords_koop = self.koop_model.decoder(roll_koop.reshape(B * (steps+1), self.n_atoms * h_dim)).reshape(B, steps+1, self.n_atoms, 3).cpu()
            coords_gru = self.gru_model.decoder(roll_gru.reshape(B * (steps+1), self.n_atoms * h_dim)).reshape(B, steps+1, self.n_atoms, 3).cpu()

        def calc_drifts(coords):
            b_vals = []
            for i, j in bonds:
                b_vals.append(torch.norm(coords[..., i, :] - coords[..., j, :], dim=-1))
            b_vals = torch.stack(b_vals, dim=-1)
            b_drift = torch.mean(torch.abs(b_vals - b_vals[:, 0:1, :]), dim=(0, 2)).numpy()

            a_vals = self.compute_angles(coords, angles)
            a_drift = torch.mean(torch.abs(a_vals - a_vals[:, 0:1, :]), dim=(0, 2)).numpy() if a_vals is not None else np.zeros(steps+1)

            t_vals = self.compute_torsions(coords, torsions)
            t_drift = torch.mean(torch.abs(t_vals - t_vals[:, 0:1, :]), dim=(0, 2)).numpy() if t_vals is not None else np.zeros(steps+1)

            return b_drift, a_drift, t_drift

        kb, ka, kt = calc_drifts(coords_koop)
        gb, ga, gt = calc_drifts(coords_gru)

        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
        steps_arr = np.arange(steps + 1)

        ax1 = fig.add_subplot(gs[0, 0])
        K_glob = self.koop_model.get_global_K()
        eigvals = np.linalg.eigvals(K_glob)
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.8, label='Unit Circle (rho=1)')
        ax1.scatter(eigvals.real, eigvals.imag, color='#2166ac', alpha=0.6, s=15, label='K_global Eigenvalues')
        ax1.set_aspect('equal')
        ax1.set_title(f"{self.molecule_name} Eigenvalue Spectrum", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Real Part")
        ax1.set_ylabel("Imaginary Part")
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.legend(loc='lower left')

        ax2 = fig.add_subplot(gs[0, 1])
        angles_eig = np.angle(eigvals)
        ax2.hist(angles_eig, bins=100, color='#2166ac', edgecolor='black', alpha=0.7)
        ax2.set_title(f"Histogram of Eigenvalue Angles", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Angle (Radians)")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, linestyle=':', alpha=0.5)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(angles_eig, bins=100, color='#d6604d', edgecolor='black', alpha=0.7)
        ax3.set_title(f"Histogram of Eigenvalue Angles (Log Scale)", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Angle (Radians)")
        ax3.set_ylabel("Frequency (Log)")
        ax3.set_yscale('log')
        ax3.grid(True, linestyle=':', alpha=0.5)

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(steps_arr, kb, color='#2166ac', linewidth=2.5, marker='o', markersize=4, label='Koopman')
        ax4.plot(steps_arr, gb, color='#d6604d', linewidth=2.5, linestyle='--', marker='x', markersize=5, label='GRU')
        ax4.set_title("Mean Bond Length Drift", fontsize=13, fontweight='bold')
        ax4.set_xlabel("Prediction Horizon (steps)")
        ax4.set_ylabel("Drift from t=0 (Angstroms)")
        ax4.grid(True, linestyle=':', alpha=0.5)
        ax4.legend()

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(steps_arr, ka, color='#2166ac', linewidth=2.5, marker='o', markersize=4)
        ax5.plot(steps_arr, ga, color='#d6604d', linewidth=2.5, linestyle='--', marker='x', markersize=5)
        ax5.set_title("Mean Bond Angle Drift", fontsize=13, fontweight='bold')
        ax5.set_xlabel("Prediction Horizon (steps)")
        ax5.set_ylabel("Drift from t=0 (Degrees)")
        ax5.grid(True, linestyle=':', alpha=0.5)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(steps_arr, kt, color='#2166ac', linewidth=2.5, marker='o', markersize=4)
        ax6.plot(steps_arr, gt, color='#d6604d', linewidth=2.5, linestyle='--', marker='x', markersize=5)
        ax6.set_title("Mean Torsion Angle Drift", fontsize=13, fontweight='bold')
        ax6.set_xlabel("Prediction Horizon (steps)")
        ax6.set_ylabel("Drift from t=0 (Degrees)")
        ax6.grid(True, linestyle=':', alpha=0.5)

        plt.savefig(os.path.join(out_dir, f'physics_eval_{self.molecule_name}.png'), dpi=150, bbox_inches='tight')
        plt.show()


