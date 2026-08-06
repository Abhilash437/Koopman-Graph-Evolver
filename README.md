# Koopman Graph Evolver: Long-Horizon Graph Dynamical Simulation

A deep learning framework for predicting deterministic, long-horizon molecular and physical dynamics using Graph-Aware Koopman operator theory and orthogonal latent transitions.

## Overview

This project implements the **Koopman Graph Evolver (KGE)** (and equivariant variations like E-GKN and EGNN) to model complex physical and molecular systems. Supported systems span 14 physical benchmarks: 8 MD17 molecules, 4 MD22 macromolecules, and 2 N-body physical systems. We conduct multi-seed 3-way ablation sweeps comparing orthogonal matrix-exponential Graph Koopman transitions ($\mathbf{K} \in SO(n)$) against unconstrained Graph GRUs and graph-free Flat Koopman baselines.

**Key Innovation:** Enforcing matrix-exponential orthogonal transitions ($\mathbf{K} = \exp(\mathbf{A}_{\text{skew}}\Delta t) \in SO(n)$) in latent space. This guarantees exact latent norm and volume preservation by construction ($R_{\text{norm}} = 1.0000$), preventing latent drift and suppressing unphysical structural deformation over extended autoregressive rollouts.

---

## Repository Structure

```text
.
├── koopman_evolver/           # Core Python package
│   ├── data/                  # MD17, MD22, N-Body data loaders & adapters
│   ├── models/                # GraphAwareKoopmanNet, GraphAwareGRUNet, FlatKoopmanNet, E-GKN
│   ├── training/              # PyTorch training loops with physical loss formulation
│   ├── evaluation/            # PhysicsEval suite computing physical geometric diagnostics
│   └── cli.py                 # Command Line Interface entrypoint
├── paper/                     # Manuscript source files, LaTeX tables, and figures
│   └── main.tex               # Double-blind conference manuscript
├── app.py                     # Streamlit Web GUI Dashboard
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Containerization setup
└── docker-compose.yml         # Container execution setup
```

---

## Quick Start (Docker)

Run training and evaluation via Docker:

### Launch Web GUI Dashboard
```bash
docker compose up koopman-gui
```
*Access the dashboard at `http://localhost:8501` in your browser.*

### Using the CLI via Docker

**Train a model:**
```bash
# Models: 'koopman', 'gru', 'flat', 'e-gkn', or 'egnn'
# Systems: 'ethanol' (MD17), 'stachyose' (MD22), 'charged' / 'springs' (N-body)
docker compose run --build --rm koopman train --md22 stachyose --model koopman --epochs 100
```

**Evaluate trained models (3-Way Ablation):**
```bash
docker compose run --rm koopman eval --md22 stachyose \
  --koopman-ckpt checkpoints/graph_aware_koopman_stachyose_best.pt \
  --gru-ckpt checkpoints/graph_aware_gru_stachyose_best.pt \
  --flat-ckpt checkpoints/flat_koopman_stachyose_best.pt
```

---

## Quick Start (Native Python)

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Run the Streamlit GUI
streamlit run app.py

# 3. OR Run the CLI
python -m koopman_evolver.cli train --md22 stachyose --model koopman --epochs 100
```

---

## Benchmark Results Across 14 Physical Systems

Multi-seed evaluation across random seeds ($\{42, 1337, 2026\}$) on 14 physical systems comparing **Graph Koopman (KGE)**, **Graph GRU (G-GRU)**, and **Flat Koopman (Flat-K)**:

### 1. Multi-Seed Robustness (Representative Benchmark Systems)

| System | Model | Rollout MSE | Bond Drift (Å) | Angle Drift (°) | Torsion Drift (°) | Physical Coord Edge Ratio ($R_{\text{edge}}$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **aspirin** | Flat Koopman | 0.0715 ± 0.008 | 0.0816 ± 0.004 | 4.55 ± 0.45 | 5.38 ± 0.32 | 0.9707 |
| | **Graph Koopman** | 0.2411 ± 0.003 | **0.0045 ± 0.004** | **0.09 ± 0.02** | **0.15 ± 0.06** | **0.9974** |
| | Graph GRU | 0.1388 ± 0.031 | 0.0689 ± 0.012 | 5.49 ± 1.13 | 6.42 ± 0.62 | 0.9584 |
| **malonaldehyde** | Flat Koopman | 0.4002 ± 0.003 | 0.1699 ± 0.022 | 10.51 ± 1.29 | 16.73 ± 1.28 | 0.9367 |
| | **Graph Koopman** | 0.9151 ± 0.048 | **0.0905 ± 0.030** | **0.46 ± 0.34** | **0.83 ± 0.40** | **0.9416** |
| | Graph GRU | 0.3532 ± 0.005 | 0.0981 ± 0.005 | 3.72 ± 0.66 | 4.19 ± 1.36 | 0.9262 |
| **at-at** | Flat Koopman | 3.6341 ± 1.041 | 0.4499 ± 0.146 | 37.14 ± 9.74 | 48.21 ± 11.20 | 0.9972 |
| | **Graph Koopman** | 6.2917 ± 0.514 | **0.0240 ± 0.010** | **0.63 ± 0.26** | **1.15 ± 0.48** | **0.9868** |
| | Graph GRU | 2.6390 ± 0.158 | 0.2514 ± 0.038 | 17.98 ± 1.96 | 26.37 ± 3.13 | 0.8675 |
| **springs** | Flat Koopman | 0.1756 ± 0.001 | 0.1075 ± 0.026 | 14.55 ± 2.87 | 29.38 ± 5.56 | 0.9827 |
| | **Graph Koopman** | 0.1764 ± 0.003 | **0.0248 ± 0.009** | **2.59 ± 1.01** | **6.05 ± 1.79** | **1.0112** |
| | Graph GRU | 0.0531 ± 0.002 | 0.6167 ± 0.013 | 46.51 ± 0.87 | 80.21 ± 0.95 | 1.6290 |

---

### Key Empirical Findings

1. **Physical Structural Stability:** Graph Koopman (\GKE{}) reduces bond drift across 13/14 systems and angle/torsion drift across **14/14 systems** compared to Graph GRU ($p < 3.1 \times 10^{-4}$, Wilcoxon signed-rank test).
2. **Latent & Coordinate Regularization:** Feature-space latent norm ratios are strictly $R_{\text{norm}} = 1.0000$ by construction. Decoded 3D physical coordinate edge ratios remain highly stable ($R_{\text{edge}} \approx 1.0$), whereas Graph GRUs experience coordinate explosion ($1.63\times$ expansion on N-body springs) or coordinate decay ($0.73\times$ collapse on stachyose).
3. **The MSE Paradox:** Pointwise rollout MSE can reward models that uniformly expand or collapse toward spatial centroids. Physical topology metrics (bond, angle, torsion drift and coordinate edge ratios) accurately reflect true geometric fidelity.

---

## License

MIT License - feel free to use, fork, and modify for your research.
