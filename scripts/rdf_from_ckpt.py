#!/usr/bin/env python3
"""
Q3 RDF pilot: compare predicted vs ground-truth radial distribution g(r)
from existing KGE / G-GRU checkpoints (eval-only, no retraining).

Example (GCP, aspirin full seed 42):
  python3 scripts/rdf_from_ckpt.py \\
    --md17 aspirin \\
    --koopman-ckpt checkpoints/rebuttal_1LAu/graph_aware_koopman_aspirin_seed42_full_best.pt \\
    --gru-ckpt checkpoints/rebuttal_1LAu/graph_aware_gru_aspirin_seed42_full_best.pt \\
    --device cuda --out-dir results/rdf_pilot/aspirin_full_seed42
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from koopman_evolver.cli import get_data_path, get_device
from koopman_evolver.data.md17_adapter import MD17AdapterV2
from koopman_evolver.data.md22_adapter import MD22Adapter
from koopman_evolver.models.baselines import GraphAwareGRUNet
from koopman_evolver.models.koopman_net import GraphAwareKoopmanNet


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """coords: (..., N, 3) -> flattened upper-triangle distances."""
    n = coords.shape[-2]
    i, j = np.triu_indices(n, k=1)
    diff = coords[..., i, :] - coords[..., j, :]
    return np.linalg.norm(diff, axis=-1).reshape(-1)


def rdf_histogram(
    distances: np.ndarray,
    r_max: float = 6.0,
    bin_width: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.0, r_max + bin_width, bin_width)
    counts, edges = np.histogram(distances, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Shell volume normalization (relative RDF shape; not absolute g(r))
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    shell = np.maximum(shell, 1e-12)
    g = counts.astype(np.float64) / shell
    if g.sum() > 0:
        g = g / g.sum()
    return centers, g


def l1_distance(g_a: np.ndarray, g_b: np.ndarray) -> float:
    return float(np.abs(g_a - g_b).sum())


@torch.no_grad()
def decode_rollout(model, node_feats, edge_idx, edge_feats, lengths, steps, device, batch_size=32):
    """Encode t=0, rollout, decode → (B, steps+1, N, 3) on CPU."""
    model.eval()
    chunks = []
    n = node_feats.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        nf = node_feats[start:end]
        ef = edge_feats[start:end]
        ln = lengths[start:end]
        h_seq = model(nf, edge_idx, ef, ln)
        h0 = h_seq[:, :1]
        roll = model.forward_rollout(h0, steps=steps + 1, latent_seed=True)
        b, s, n_atoms, h_dim = roll.shape
        coords = model.decoder(roll.reshape(b * s, n_atoms * h_dim)).reshape(b, s, n_atoms, 3)
        chunks.append(coords.cpu())
    return torch.cat(chunks, dim=0)


def load_models(edge_index, n_atoms, koop_path, gru_path, device):
    latent_dim = n_atoms * 64
    koop = GraphAwareKoopmanNet(
        edge_index=edge_index, node_dim=6, edge_dim=1,
        hidden_dim=64, latent_dim=latent_dim, n_atoms=n_atoms,
    ).to(device)
    gru = GraphAwareGRUNet(
        edge_index=edge_index, node_dim=6, edge_dim=1,
        hidden_dim=64, latent_dim=latent_dim, n_atoms=n_atoms,
    ).to(device)
    k_ckpt = torch.load(koop_path, map_location=device, weights_only=False)
    g_ckpt = torch.load(gru_path, map_location=device, weights_only=False)
    koop.load_state_dict(k_ckpt["model_state_dict"])
    gru.load_state_dict(g_ckpt["model_state_dict"])
    print(f"Loaded KGE epoch={k_ckpt.get('epoch')}  GRU epoch={g_ckpt.get('epoch')}")
    return koop, gru


def rdf_at_step(coords: np.ndarray, lengths, step: int, r_max: float, bin_width: float):
    """coords (B, T, N, 3); keep samples with length > step."""
    dists = []
    for b in range(coords.shape[0]):
        if lengths[b] <= step:
            continue
        dists.append(pairwise_distances(coords[b, step]))
    if not dists:
        raise RuntimeError(f"No samples with length > {step}")
    return rdf_histogram(np.concatenate(dists), r_max=r_max, bin_width=bin_width)


def main():
    p = argparse.ArgumentParser(description="RDF pilot from KGE/G-GRU checkpoints")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--md17", type=str)
    g.add_argument("--md22", type=str)
    p.add_argument("--koopman-ckpt", required=True)
    p.add_argument("--gru-ckpt", required=True)
    p.add_argument("--rollout-steps", type=int, default=29)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out-dir", type=str, default="./results/rdf_pilot")
    p.add_argument("--r-max", type=float, default=6.0)
    p.add_argument("--bin-width", type=float, default=0.05)
    p.add_argument("--max-traj", type=int, default=0, help="Optional cap on test trajectories (0=all)")
    args = p.parse_args()

    device = get_device(args.device)
    if args.md17:
        name, dataset = args.md17, "md17"
        adapter = MD17AdapterV2(path=get_data_path("md17", name), molecule=name)
    else:
        name, dataset = args.md22, "md22"
        adapter = MD22Adapter(path=get_data_path("md22", name), molecule=name)

    print(f"[{name}] Loading {dataset} ...")
    _, test_split = adapter.load()
    n_atoms = adapter._n_atoms
    edge_index = test_split.edge_index

    node_feats = torch.tensor(test_split.node_features, dtype=torch.float32, device=device)
    edge_feats = torch.tensor(test_split.edge_features, dtype=torch.float32, device=device)
    edge_idx = edge_index.to(device)
    lengths = list(test_split.lengths)

    if args.max_traj > 0:
        node_feats = node_feats[: args.max_traj]
        edge_feats = edge_feats[: args.max_traj]
        lengths = lengths[: args.max_traj]

    # GT coordinates live in the first 3 channels of node features
    coords_gt = node_feats[..., :3].detach().cpu().numpy()

    koop, gru = load_models(edge_index, n_atoms, args.koopman_ckpt, args.gru_ckpt, device)
    steps = args.rollout_steps
    print(f"Rolling out {steps} steps on {node_feats.shape[0]} trajectories...")
    coords_k = decode_rollout(koop, node_feats, edge_idx, edge_feats, lengths, steps, device).numpy()
    coords_g = decode_rollout(gru, node_feats, edge_idx, edge_feats, lengths, steps, device).numpy()

    metrics: Dict[str, float] = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, step, title in zip(axes, [0, steps], ["t = 0", f"t = {steps}"]):
        r, g_gt = rdf_at_step(coords_gt, lengths, step, args.r_max, args.bin_width)
        _, g_k = rdf_at_step(coords_k, lengths, step, args.r_max, args.bin_width)
        _, g_g = rdf_at_step(coords_g, lengths, step, args.r_max, args.bin_width)
        metrics[f"L1_KGE_vs_GT_t{step}"] = l1_distance(g_k, g_gt)
        metrics[f"L1_GRU_vs_GT_t{step}"] = l1_distance(g_g, g_gt)
        ax.plot(r, g_gt, color="black", lw=2, label="GT")
        ax.plot(r, g_k, color="#2166ac", lw=1.8, label="KGE")
        ax.plot(r, g_g, color="#d6604d", lw=1.8, label="G-GRU")
        ax.set_title(title)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("normalized shell density")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    metrics["delta_L1_t_final_minus_t0_KGE"] = (
        metrics[f"L1_KGE_vs_GT_t{steps}"] - metrics["L1_KGE_vs_GT_t0"]
    )
    metrics["delta_L1_t_final_minus_t0_GRU"] = (
        metrics[f"L1_GRU_vs_GT_t{steps}"] - metrics["L1_GRU_vs_GT_t0"]
    )

    os.makedirs(args.out_dir, exist_ok=True)
    plot_path = os.path.join(args.out_dir, f"rdf_{name}.png")
    json_path = os.path.join(args.out_dir, f"rdf_{name}_metrics.json")
    fig.suptitle(f"RDF pilot — {name} (KGE vs G-GRU vs GT)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nRDF L1(|g_pred - g_GT|)  [lower is closer to ground truth]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
