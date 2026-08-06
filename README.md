# Beyond MSE: Geometry-Preserving Latent Dynamics for Long-Horizon Graph Simulation

Official implementation of the paper **"Beyond MSE: Geometry-Preserving Latent Dynamics for Long-Horizon Graph Simulation"**.

---

## Executive Summary & Abstract

Graph neural networks (GNNs) achieve high short-horizon accuracy for physical simulation, yet accumulate severe latent drift over long-horizon autoregressive rollouts. Unconstrained temporal transitions (e.g., GRUs) allow latent representation norms to progressively expand or contract, producing trajectories with competitive pointwise rollout MSE despite catastrophic physical structural breakdown (bond stretching, angle distortion, and centroid collapse).

We introduce the **Koopman Graph Evolver (KGE)** and its SE(3)-equivariant extension **E-GKN**, which replace unconstrained recurrent transitions with an orthogonal Koopman operator acting in latent space. Parameterizing the transition via a matrix exponential $K = \exp(A_{\text{skew}} \Delta t)$ guarantees $K \in \text{SO}(n)$, preserving latent norm and volume by construction ($R_{\text{norm}} = 1.0000$).

Evaluated across **14 physical systems** (8 MD17 molecules, 4 MD22 macromolecules, and 2 N-body particle systems), geometry-preserving latent transitions yield statistically significant reductions in structural drift ($p \le 3.1 \times 10^{-4}$, Wilcoxon signed-rank test), maintaining physical coordinate edge length ratios ($R_{\text{edge}} \approx 1.0$) across extended rollouts.

---

## Key Contributions & Mathematical Framework

1. **Characterization of Structural Drift & The MSE Paradox:** We show that standard rollout MSE fails to reflect physical degradation, rewarding models that expand uniformly or collapse toward spatial centroids. Physical topology metrics (bond, angle, torsion drift, and coordinate edge ratios) are required for faithful physical evaluation.
2. **Volume-Preserving Koopman Transitions:** Enforcing $K = \exp(A_{\text{skew}} \Delta t) \in \text{SO}(n)$ guarantees:
   - $K^T K = I$ (Orthogonality)
   - $\det(K) = 1$ (Orientation and volume preservation)
   - $\|Kz\|_2 = \|z\|_2$ (Norm preservation in latent feature space)
3. **Physical Regularization in 3D Space:** While the non-linear spatial decoder does not mathematically mandate 3D coordinate edge ratios $R_{\text{edge}} = 1.0$ identically, latent orthogonality strongly regularizes spatial decoding, keeping $R_{\text{edge}} \approx 1.0$ ($0.9416$–$1.0112$) across extended rollouts.
4. **SE(3)-Equivariant Extension (E-GKN):** Augmenting equivariant message passing with shared node-local Koopman transitions prevents numerical divergence ($10^{27}$ / NaNs) present in standard EGNNs on large flexible macromolecules.

---

## Empirical Benchmark Results (14 Physical Systems)

### 1. Multi-Seed Robustness (Averaged Over Seeds {42, 1337, 2026})

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
| | **Graph Koopman** | 0.1764 ± 0.003 | **0.0248 ± 0.009** | **2.59 ± 1.01** | **6.05 ± 1.79** | **1.0112** |
| | Graph GRU | 0.0531 ± 0.002 | 0.6167 ± 0.013 | 46.51 ± 0.87 | 80.21 ± 0.95 | 1.6290 |

### 2. Statistical Significance Across All 14 Systems

One-sided Wilcoxon signed-rank test results comparing Graph Koopman (KGE) vs. Graph GRU (G-GRU) across 14 physical systems:

| Metric | KGE Win Rate | Wilcoxon Statistic | p-value |
|:---|:---:|:---:|:---:|
| **Bond Drift (Å)** | 13/14 | 1.5 | $3.05 \times 10^{-4}$ |
| **Angle Drift (°)** | 14/14 | 0.0 | $6.10 \times 10^{-5}$ |
| **Torsion Drift (°)** | 14/14 | 0.0 | $6.10 \times 10^{-5}$ |
| **Latent Norm Ratio \|R_norm - 1\|** | 14/14 | 0.0 | $6.10 \times 10^{-5}$ |

---

## How-To Guide: Installation & Execution

### 1. Installation

**Prerequisites:** Python 3.9+ or Docker.

```bash
# Clone the repository
git clone https://github.com/Abhilash437/Koopman-Graph-Evolver.git
cd Koopman-Graph-Evolver

# Install requirements
pip install -r requirements.txt
```

---

### 2. Quick Start via Docker (Recommended)

**Launch Interactive Web GUI Dashboard:**
```bash
docker compose up koopman-gui
```
*Access the dashboard at `http://localhost:8501` in your browser.*

**Train a model via Docker CLI:**
```bash
# Models: 'koopman', 'gru', 'flat', 'e-gkn', 'egnn'
# Systems: 'ethanol' (MD17), 'stachyose' (MD22), 'charged' / 'springs' (N-body)
docker compose run --build --rm koopman train --md22 stachyose --model koopman --epochs 100
```

**Run 3-Way Ablation Evaluation Suite:**
```bash
docker compose run --rm koopman eval --md22 stachyose \
  --koopman-ckpt checkpoints/graph_aware_koopman_stachyose_best.pt \
  --gru-ckpt checkpoints/graph_aware_gru_stachyose_best.pt \
  --flat-ckpt checkpoints/flat_koopman_stachyose_best.pt
```

---

### 3. Quick Start via Native CLI

**Train Graph Koopman on MD17 / MD22 / N-Body:**
```bash
# Train Graph Koopman on Aspirin (MD17)
python -m koopman_evolver.cli train --md17 aspirin --model koopman --seed 42 --epochs 100

# Train E-GKN on DHA (MD22)
python -m koopman_evolver.cli train --md22 dha --model e-gkn --seed 42 --epochs 100

# Train Graph Koopman on N-Body Charged
python -m koopman_evolver.cli train --nbody charged --model koopman --seed 42 --epochs 100
```

**Evaluate Trained Checkpoints:**
```bash
python -m koopman_evolver.cli eval --md17 aspirin \
  --koopman-ckpt checkpoints/graph_aware_koopman_aspirin_seed42.pt \
  --gru-ckpt checkpoints/graph_aware_gru_aspirin_seed42.pt \
  --flat-ckpt checkpoints/flat_koopman_aspirin_seed42.pt \
  --rollout-steps 29
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
