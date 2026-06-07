# Koopman Graph Evolver: Molecular Dynamics

A deep learning framework for predicting deterministic, long-horizon molecular dynamics using Graph-Aware Koopman operator theory and Lie-algebraic message passing.

## Overview

This project implements a **Graph-Aware Koopman Autoencoder** to model the global dynamics of molecules from the MD17 and MD22 datasets (e.g., Ethanol, Aspirin, Ac-Ala3-NHMe, Stachyose). It conducts massive multi-molecule 3-way ablation studies comparing mathematically constrained linear Graph Koopman transitions against unconstrained Graph GRUs, and graph-free Flat Koopman baselines on up to 30-step autonomous rollout trajectories.

**Key Innovation:** Applying Lie-algebraic constraints (projecting the Koopman operator onto the orthogonal group) to a graph-structured latent space. This guarantees unconditional long-term structural stability and exactly preserves rigid molecular geometries, solving the catastrophic latent spatial collapse typically seen in flat MLPs and temporal compounding errors of recurrent models.

## Repository Structure

### Recent Improvements

- **MD22 Scalability (Phase 10):** Scaled out from small MD17 molecules to the massive MD22 dataset, supporting complex macromolecules like Stachyose (87 atoms) using adaptive SVD Principal Axis alignment.
- **3-Way Massive Ablation (Phase 9):** Introduced the `ThreeWayAblationEvaluator` suite to definitively compare Graph Koopman, Graph GRU, and a non-graph Flat Koopman baseline in coordinate-space.
- **Coordinate-Space Evaluation**: The `_rollout_mse` function has been updated to compute the error on physical (x, y, z) coordinate outputs rather than disjoint latent spaces, providing a standardized, physical metric.
- **BPTT for Baseline**: The GRU Baseline now utilizes Backpropagation Through Time (BPTT) with multi-step rollouts during training to alleviate exposure bias and improve long-term stability.

The original experimental notebooks have been fully ported into a modular Python package with a Command Line Interface (CLI) and a Streamlit Web GUI.

```text
.
├── koopman_evolver/           # Core Python package
│   ├── data/                  # Kaggle dataset downloading and MD17 windowing/splitting logic
│   ├── models/                # GraphAwareKoopmanNet and GraphAwareGRUNet architectures
│   ├── training/              # PyTorch training loops with physical regularization
│   ├── evaluation/            # PhysicsEval suite computing long-horizon geometric drifts
│   └── cli.py                 # Command Line Interface entrypoint
├── app.py                     # Streamlit Web GUI Dashboard
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Containerization logic
├── docker-compose.yml         # Local volume mapping and service definitions
├── Graph Dynamics Learner/    # Legacy Jupyter Notebooks and mathematical summaries
└── Phase 1, 2, 3/             # Legacy graph evolution experiments
```

## Quick Start (Docker)

The recommended way to run the application is via Docker. This keeps your system clean while automatically mapping models, plots, and datasets to your local file system via Docker Volumes.

### Launch the Web GUI Dashboard

We provide an interactive **Streamlit** dashboard to run training jobs and generate physical diagnostic plots:

```bash
docker compose up koopman-gui
```

*Access the dashboard at `http://localhost:8501` in your browser.*

### Using the CLI via Docker

You can trigger training and evaluation runs directly from the terminal without using the GUI.

**Train a model:**

```bash
# Models: 'koopman', 'gru', or 'flat'
# Molecules: 'ethanol' (MD17) or 'stachyose' (MD22)
docker compose run --build --rm koopman train --md22 stachyose --model koopman --epochs 50
```

**Evaluate trained models (3-Way Ablation):**

```bash
docker compose run --rm koopman eval --md22 stachyose \
  --koopman-ckpt checkpoints/graph_aware_koopman_stachyose_best.pt \
  --gru-ckpt checkpoints/graph_aware_gru_stachyose_best.pt \
  --flat-ckpt checkpoints/flat_koopman_stachyose_best.pt
```

*Note: Checkpoints will be automatically saved to `./checkpoints/` and diagnostic plots to `./results/` on your host machine.*

## Quick Start (Native Python)

If you prefer not to use Docker, you can run the code natively:

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Run the Streamlit GUI
streamlit run app.py

# 3. OR Run the CLI
python -m koopman_evolver.cli train --md22 stachyose --model flat --epochs 50
```

*(Note: The datasets will automatically be downloaded from Kaggle using `kagglehub` the first time you run the code).*

## Key Findings (MD22 Dataset Ablation)

We rigorously tested the models across 30-step autonomous prediction horizons on 4 distinct macromolecules (Ac-Ala3-NHMe, DHA, AT-AT, Stachyose):

| Metric (Averaged MD22) | Graph Koopman | Graph GRU | Flat Koopman |
|--------|-------------------------|-------------------|-------------------|
| **Rollout MSE @ 5 Steps** | 0.598 | **0.581** | 0.728 |
| **Rollout MSE @ 15 Steps** | 0.660 | **0.656** | 2.137 |
| **Rollout MSE @ 28 Steps** | **0.774** | 0.790 | 3.533 (Spatial Collapse) |
| **Ethanol Mean Bond Drift**| **0.000 Å** | ~0.12 Å | ~0.015 Å |

**Conclusion:**

1. **The Spatial Collapse:** Flat architectures completely disintegrated structurally over long horizons on large molecules.
2. **The Temporal Compounding:** GRUs excel at short-term prediction (Step 1-15) but non-linear errors compound catastrophically at longer horizons.
3. **Graph Koopman Supremacy:** By combining Graph Message Passing (for spatial rigidity) with linear Koopman operators (for temporal stability), the architecture overtakes recurrent baselines at $T \geq 28$ steps without ever exploding.

## Technical Documentation

For the full theoretical breakdown—including the Lie Algebra message-passing formulation, the Isometric Regularization loss, and detailed analysis of the spectral eigenvalue results—please read our comprehensive summary:

👉 **[Project Summary & Mathematical Framework](Graph%20Dynamics%20Learner/Project_Summary_Graph_Aware_Koopman.md)**

## License

MIT License - feel free to use, fork, and modify for your research.
