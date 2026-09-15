# Beyond MSE: Orthogonal Latent Dynamics for Long-Horizon Graph Simulation

Official implementation of **"Beyond MSE: Orthogonal Latent Dynamics for Long-Horizon Graph Simulation"**. The reported model is Kronecker **GraphAwareKoopmanNet** (`K_glob`). Legacy `GraphKoopmanNet` (node-wise 64×64) is **not CLI-wired and not the paper**.

---

## Executive summary

GNNs can keep one-step error low while decoded topology degrades over autoregressive rollouts (MSE-versus-topology). **KGE** replaces an unconstrained recurrent cell with `K = exp(A_skew) ∈ SO(n)` (implicit `Δt = 1`). That constrains **latent** norm/volume; it is **not** energy conservation, Liouville mechanics, or decoded-bond preservation by construction.

Bond / angle / torsion in the tables below are **drift from decoded t=0**, not vs ground-truth topology. Bonded margins are also trained by an iso loss (weights **10 / 1 / 2 / 5**). Across 14 systems we report a stability–accuracy tradeoff: lower t=0 drift vs G-GRU, often **higher** MSE. E-GKN vs EGNN is empirical under this protocol.

---

## Key points (matches code)

1. **MSE-versus-topology:** rollout MSE can miss decoded t=0 bond/angle/torsion drift.
2. **SO(n) is latent-only:** `K^T K = I`, `det(K)=1`, `||Kz||_2 = ||z||_2` in latent space (Theorem 1). Decoder geometry is empirical.
3. **Iso trains bonds.** `R_edge ≈ 1` is not mandated by SO(n) alone.
4. **E-GKN** uses the same Kronecker `K_glob` on invariant features (not independent node-local cells). Finite rollouts vs EGNN overflow are a protocol result, not physical energy.

---

## Training objective (matches `compute_loss`)

Default weights are **fixed** at **10 / 1 / 2 / 5** (reconstruction / dynamics / collapse / iso). There is no annealed λ.

- **Reconstruction:** teacher-forced autoencoder — encode the current graph, decode, score vs the **same-timestep** coordinates. Not next-frame prediction.
- **Dynamics:** one-step latent consistency (`K s_t` vs encoder `s_{t+1}`).
- **Collapse:** encoder anti-freeze hinge. Does **not** train `R_norm`.
- **Iso:** bonded-distance MSE on decoded coordinates; this term **does** push bond / bond-margin numbers.
- **`R_norm`:** architectural `SO(n)` from `K = exp(A_glob)` with implicit `Δt = 1` (not a physical integrator step). Collapse does not enforce it.
- Bond / angle / torsion eval scores in `physics_eval.py` are **drift from decoded t=0**, not vs ground-truth topology.

---

## Empirical Benchmark Results (14 Physical Systems)

### 1. Multi-seed robustness (P3, seeds {42, 1337, 2026})

Bond / angle / torsion = **decoded t=0 drift**, not vs GT. Baselines are **not** an identical-objective bake-off (G-GRU uses 4-step dyn unroll; Flat-K drops iso). **P3** = Table 2 3-seed aggregate; **P1** = appendix single-seed sweep (`sweep_20260716_161829`) — do not cite P1 cells as multi-seed means. Springs MSE is mean±sample stdev from 3-seed nbody logs (`multiseed_results_nbody*.txt`, seeds {42,1337,2026}).

| System | Model | Rollout MSE (29-step) | Bond Drift (Å) | Angle Drift (°) | Torsion Drift (°) | Physical Coord Edge Ratio ($R_{\text{edge}}$) |
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
| | **Graph Koopman** | 0.1768 ± 0.003 | **0.0248 ± 0.009** | **2.59 ± 1.01** | **6.05 ± 1.79** | **1.0112** |
| | Graph GRU | 0.0455 ± 0.0185 | 0.6167 ± 0.013 | 46.51 ± 0.87 | 80.21 ± 0.95 | 1.6290 |

### 2. Statistical Significance Across All 14 Systems

One-sided Wilcoxon on **decoded t=0 drift** (KGE vs G-GRU). Not a proof of physical superiority. MSE is often lower for G-GRU. **Caveat:** available logs do not name whether the pairing used P3 means or P1 sweep cells; p-values are as previously reported and are not recomputed.

| Metric | KGE Win Rate | Wilcoxon Statistic | p-value |
|:---|:---:|:---:|:---:|
| **Bond Drift (Å)** | 13/14 | 1.5 | $3.05 \times 10^{-4}$ |
| **Angle Drift (°)** | 14/14 | 0.0 | $6.10 \times 10^{-5}$ |
| **Torsion Drift (°)** | 14/14 | 0.0 | $6.10 \times 10^{-5}$ |
| **Latent Norm Ratio \|R_norm - 1\|** | 14/14 | 0.0 | $6.10 \times 10^{-5}$ |

---

## Quickstart & Installation

[![PyPI Version](https://img.shields.io/pypi/v/koopman-graph-evolver.svg?v=1)](https://pypi.org/project/koopman-graph-evolver/)
[![Python Versions](https://img.shields.io/pypi/pyversions/koopman-graph-evolver.svg?v=1)](https://pypi.org/project/koopman-graph-evolver/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### 1. Install via PyPI

```bash
# Core package (PyTorch & PyG models + PhysicsEval suite)
pip install koopman-graph-evolver

# Optional extras: GUI dashboard or dataset downloaders
pip install "koopman-graph-evolver[gui,data]"
```

---

### 2. Python API Quickstart

**Use Graph-Aware Koopman Net or E-GKN in your PyTorch code:**

```python
import torch
from koopman_evolver import GraphAwareKoopmanNet, EGKN, PhysicsEval

# 1. Define graph connectivity (e.g. molecular bond edges)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

# 2. Instantiate Graph-Aware Koopman Net or Equivariant Koopman Net (E-GKN)
model = GraphAwareKoopmanNet(
    edge_index=edge_index,
    node_dim=6,
    hidden_dim=64,
    n_atoms=3
)

# 3. Perform SO(n) latent rollout (decoded geometry is empirical)
h0 = torch.randn(1, 5, 3, 64)  # Initial trajectory (B, T, N, D)
rollout = model.forward_rollout(h0, steps=30)

# 4. Evaluate physical structural drift metrics
evaluator = PhysicsEval(koop_model=model, gru_model=model, test_split=None, n_atoms=3, molecule_name="custom")
bonds, angles, torsions = evaluator.extract_topology(edge_index)
```

---

### 3. Execution via Executable CLI (`kge`)

**Train models via `kge` CLI:**
```bash
# Train Graph Koopman on Aspirin (MD17)
kge train --md17 aspirin --model koopman --seed 42 --epochs 100

# Train E-GKN on DHA (MD22)
kge train --md22 dha --model e-gkn --seed 42 --epochs 100

# Train Graph Koopman on N-Body Charged
kge train --nbody charged --model koopman --seed 42 --epochs 100
```

**Evaluate Checkpoints via `kge` CLI:**
```bash
kge eval --md17 aspirin \
  --koopman-ckpt checkpoints/graph_aware_koopman_aspirin_seed42.pt \
  --gru-ckpt checkpoints/graph_aware_gru_aspirin_seed42.pt \
  --flat-ckpt checkpoints/flat_koopman_aspirin_seed42.pt \
  --rollout-steps 29
```

---

### 4. Quick Start via Docker (Recommended for GUI)

**Launch Interactive Web GUI Dashboard:**
```bash
docker compose up koopman-gui
```
*Access the dashboard at `http://localhost:8501` in your browser.*

**Train a model via Docker CLI:**
```bash
docker compose run --build --rm koopman train --md22 stachyose --model koopman --epochs 100
```

---

## Repository Structure

```text
.
├── koopman_evolver/           # Modular Python package
│   ├── data/                  # MD17, MD22, N-Body adapters and Kaggle downloaders
│   ├── models/                # GraphAwareKoopmanNet, GraphAwareGRUNet, FlatKoopmanNet, E-GKN
│   ├── training/              # PyTorch training loops with matrix exponential transitions
│   ├── evaluation/            # PhysicsEval suite & multi-system ablation metrics
│   └── cli.py                 # Command-line interface entrypoint
├── paper/                     # Manuscript source files, LaTeX tables, & figures
│   └── main.tex               # Conference manuscript LaTeX source
├── eval_logs/                 # Raw experimental log files & diagnostic evaluation outputs
├── app.py                     # Interactive Streamlit Web GUI Dashboard
├── requirements.txt           # Python package dependencies
├── Dockerfile                 # Container setup
└── docker-compose.yml         # Service definitions & volume mappings
```

---

## License

MIT License - feel free to use, fork, and modify for your research.
