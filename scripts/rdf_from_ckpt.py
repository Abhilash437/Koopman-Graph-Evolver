#!/usr/bin/env python3
"""
Q3 RDF pilot (debugged): compare KGE / G-GRU decoded rollouts to GT.

Pairwise RDFs are rigid-transform invariant, so COM/Kabsch cannot fix short-r
spikes. Those spikes usually mean absolute scale collapse in the decoder output
(self-consistent bonds can still look fine in within-trajectory drift metrics).

This script reports:
  1) scale diagnostics (mean/min bonded distance, fraction of pairs < 0.8 A)
  2) absolute all-pair RDF (original metric)
  3) scale-matched RDF (per-frame mean bond -> GT mean bond)
  4) bonded-distance histogram only

Example:
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
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from koopman_evolver.cli import get_data_path, get_device
from koopman_evolver.data.md17_adapter import MD17AdapterV2
from koopman_evolver.data.md22_adapter import MD22Adapter
from koopman_evolver.models.baselines import GraphAwareGRUNet
from koopman_evolver.models.koopman_net import GraphAwareKoopmanNet


def undirected_bonds(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    edges = edge_index.cpu().numpy()
    bonds = set()
    for u, v in zip(edges[0], edges[1]):
        if u == v:
            continue
        bonds.add((int(min(u, v)), int(max(u, v))))
    return sorted(bonds)


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[-2]
    i, j = np.triu_indices(n, k=1)
    return np.linalg.norm(coords[..., i, :] - coords[..., j, :], axis=-1).reshape(-1)


def bonded_distances(coords: np.ndarray, bonds: List[Tuple[int, int]]) -> np.ndarray:
    if not bonds:
        return np.array([], dtype=np.float64)
    vals = [np.linalg.norm(coords[..., i, :] - coords[..., j, :], axis=-1) for i, j in bonds]
    return np.stack(vals, axis=-1).reshape(-1)


def mean_bonded_length(coords: np.ndarray, bonds: List[Tuple[int, int]]) -> float:
    d = bonded_distances(coords, bonds)
    return float(d.mean()) if d.size else float("nan")


def scale_match_frame(coords: np.ndarray, bonds: List[Tuple[int, int]], target_mean_bond: float) -> np.ndarray:
    """Isotropically rescale a frame so mean bonded length matches target."""
    m = mean_bonded_length(coords, bonds)
    if not np.isfinite(m) or m < 1e-8 or not np.isfinite(target_mean_bond):
        return coords
    return coords * (target_mean_bond / m)


def rdf_histogram(distances: np.ndarray, r_max: float, bin_width: float) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.0, r_max + bin_width, bin_width)
    counts, edges = np.histogram(distances, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    shell = np.maximum(shell, 1e-12)
    g = counts.astype(np.float64) / shell
    if g.sum() > 0:
        g = g / g.sum()
    return centers, g


def hist_normalized(distances: np.ndarray, r_max: float, bin_width: float) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.0, r_max + bin_width, bin_width)
    counts, edges = np.histogram(distances, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts.astype(np.float64)


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def collect_distances(
    coords: np.ndarray,
    lengths,
    step: int,
    bonds: List[Tuple[int, int]],
    mode: str,
    gt_coords: np.ndarray | None = None,
) -> np.ndarray:
    """mode: absolute | scale_matched | bonded"""
    chunks = []
    for b in range(coords.shape[0]):
        if lengths[b] <= step:
            continue
        frame = coords[b, step]
        if mode == "absolute":
            chunks.append(pairwise_distances(frame))
        elif mode == "scale_matched":
            if gt_coords is None:
                raise ValueError("gt_coords required for scale_matched")
            gt_frame = gt_coords[b, min(step, gt_coords.shape[1] - 1)]
            target = mean_bonded_length(gt_frame, bonds)
            frame_s = scale_match_frame(frame, bonds, target)
            chunks.append(pairwise_distances(frame_s))
        elif mode == "bonded":
            chunks.append(bonded_distances(frame, bonds))
        else:
            raise ValueError(mode)
    if not chunks:
        raise RuntimeError(f"No samples with length > {step}")
    return np.concatenate(chunks)


def scale_diagnostics(coords: np.ndarray, lengths, step: int, bonds: List[Tuple[int, int]], tag: str) -> Dict[str, float]:
    bonds_all = []
    pairs_all = []
    for b in range(coords.shape[0]):
        if lengths[b] <= step:
            continue
        frame = coords[b, step]
        bonds_all.append(bonded_distances(frame, bonds))
        pairs_all.append(pairwise_distances(frame))
    bd = np.concatenate(bonds_all)
    pd = np.concatenate(pairs_all)
    out = {
        f"{tag}_mean_bond_t{step}": float(bd.mean()),
        f"{tag}_min_bond_t{step}": float(bd.min()),
        f"{tag}_mean_pair_t{step}": float(pd.mean()),
        f"{tag}_min_pair_t{step}": float(pd.min()),
        f"{tag}_frac_pairs_lt_0p8_t{step}": float(np.mean(pd < 0.8)),
    }
    return out


@torch.no_grad()
def decode_rollout(model, node_feats, edge_idx, edge_feats, lengths, steps, device, batch_size=32):
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
        bsz, s, n_atoms, h_dim = roll.shape
        coords = model.decoder(roll.reshape(bsz * s, n_atoms * h_dim)).reshape(bsz, s, n_atoms, 3)
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


def main():
    p = argparse.ArgumentParser(description="RDF pilot from KGE/G-GRU checkpoints (with scale debug)")
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
    p.add_argument("--max-traj", type=int, default=0)
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
    bonds = undirected_bonds(edge_index)
    print(f"Topology: {len(bonds)} undirected bonds, n_atoms={n_atoms}")

    node_feats = torch.tensor(test_split.node_features, dtype=torch.float32, device=device)
    edge_feats = torch.tensor(test_split.edge_features, dtype=torch.float32, device=device)
    edge_idx = edge_index.to(device)
    lengths = list(test_split.lengths)

    if args.max_traj > 0:
        node_feats = node_feats[: args.max_traj]
        edge_feats = edge_feats[: args.max_traj]
        lengths = lengths[: args.max_traj]

    coords_gt = node_feats[..., :3].detach().cpu().numpy()
    koop, gru = load_models(edge_index, n_atoms, args.koopman_ckpt, args.gru_ckpt, device)
    steps = args.rollout_steps
    print(f"Rolling out {steps} steps on {node_feats.shape[0]} trajectories...")
    coords_k = decode_rollout(koop, node_feats, edge_idx, edge_feats, lengths, steps, device).numpy()
    coords_g = decode_rollout(gru, node_feats, edge_idx, edge_feats, lengths, steps, device).numpy()

    metrics: Dict[str, float] = {"n_bonds": float(len(bonds)), "n_traj": float(node_feats.shape[0])}
    for step in (0, steps):
        metrics.update(scale_diagnostics(coords_gt, lengths, step, bonds, "GT"))
        metrics.update(scale_diagnostics(coords_k, lengths, step, bonds, "KGE"))
        metrics.update(scale_diagnostics(coords_g, lengths, step, bonds, "GRU"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    panels = [
        ("absolute", "Absolute all-pair RDF", args.r_max, True),
        ("scale_matched", "Scale-matched all-pair RDF", args.r_max, True),
        ("bonded", "Bonded-distance histogram", 3.0, False),
    ]

    for col, (mode, title, r_max, use_rdf) in enumerate(panels):
        for row, step in enumerate((0, steps)):
            ax = axes[row, col]
            d_gt = collect_distances(coords_gt, lengths, step, bonds, mode, coords_gt)
            d_k = collect_distances(coords_k, lengths, step, bonds, mode, coords_gt)
            d_g = collect_distances(coords_g, lengths, step, bonds, mode, coords_gt)
            if use_rdf:
                r, g_gt = rdf_histogram(d_gt, r_max, args.bin_width)
                _, g_k = rdf_histogram(d_k, r_max, args.bin_width)
                _, g_g = rdf_histogram(d_g, r_max, args.bin_width)
                ylab = "norm. shell dens."
            else:
                r, g_gt = hist_normalized(d_gt, r_max, args.bin_width)
                _, g_k = hist_normalized(d_k, r_max, args.bin_width)
                _, g_g = hist_normalized(d_g, r_max, args.bin_width)
                ylab = "density"

            key = f"L1_{mode}"
            metrics[f"{key}_KGE_vs_GT_t{step}"] = l1_distance(g_k, g_gt)
            metrics[f"{key}_GRU_vs_GT_t{step}"] = l1_distance(g_g, g_gt)

            ax.plot(r, g_gt, color="black", lw=2, label="GT")
            ax.plot(r, g_k, color="#2166ac", lw=1.6, label="KGE")
            ax.plot(r, g_g, color="#d6604d", lw=1.6, label="G-GRU")
            ax.set_title(f"{title} | t={step}")
            ax.set_xlabel("r (Å)")
            ax.set_ylabel(ylab)
            ax.grid(True, linestyle=":", alpha=0.5)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    for mode in ("absolute", "scale_matched", "bonded"):
        metrics[f"delta_L1_{mode}_KGE"] = (
            metrics[f"L1_{mode}_KGE_vs_GT_t{steps}"] - metrics[f"L1_{mode}_KGE_vs_GT_t0"]
        )
        metrics[f"delta_L1_{mode}_GRU"] = (
            metrics[f"L1_{mode}_GRU_vs_GT_t{steps}"] - metrics[f"L1_{mode}_GRU_vs_GT_t0"]
        )

    os.makedirs(args.out_dir, exist_ok=True)
    plot_path = os.path.join(args.out_dir, f"rdf_{name}.png")
    json_path = os.path.join(args.out_dir, f"rdf_{name}_metrics.json")
    fig.suptitle(f"RDF debug — {name} (absolute / scale-matched / bonded)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Scale diagnostics (mean bonded length should be ~1.0–1.5 Å for organics) ===")
    for step in (0, steps):
        print(
            f" t={step}: GT={metrics[f'GT_mean_bond_t{step}']:.4f}  "
            f"KGE={metrics[f'KGE_mean_bond_t{step}']:.4f}  "
            f"GRU={metrics[f'GRU_mean_bond_t{step}']:.4f}  |  "
            f"frac(pair<0.8) KGE={metrics[f'KGE_frac_pairs_lt_0p8_t{step}']:.3f} "
            f"GRU={metrics[f'GRU_frac_pairs_lt_0p8_t{step}']:.3f}"
        )
    print("\n=== L1 errors (lower = closer to GT) ===")
    for mode in ("absolute", "scale_matched", "bonded"):
        print(
            f" {mode}: t0 KGE={metrics[f'L1_{mode}_KGE_vs_GT_t0']:.4f} "
            f"GRU={metrics[f'L1_{mode}_GRU_vs_GT_t0']:.4f} | "
            f"t{steps} KGE={metrics[f'L1_{mode}_KGE_vs_GT_t{steps}']:.4f} "
            f"GRU={metrics[f'L1_{mode}_GRU_vs_GT_t{steps}']:.4f} | "
            f"ΔKGE={metrics[f'delta_L1_{mode}_KGE']:+.4f} "
            f"ΔGRU={metrics[f'delta_L1_{mode}_GRU']:+.4f}"
        )
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
