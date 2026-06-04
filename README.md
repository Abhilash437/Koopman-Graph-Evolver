# Koopman Graph Evolver: Molecular Dynamics

A deep learning framework for predicting deterministic, long-horizon molecular dynamics using Graph-Aware Koopman operator theory and Lie-algebraic message passing.

## Overview

This project implements a **Graph-Aware Koopman Autoencoder** to model the global dynamics of molecules from the MD17 dataset (e.g., Ethanol, Malonaldehyde, Aspirin). It compares the performance of mathematically constrained linear Koopman transitions against unconstrained, highly non-linear GRU baselines on 30-step autonomous rollout trajectories.

**Key Innovation:** Applying Lie-algebraic constraints (projecting the Koopman operator onto the orthogonal group) to a graph-structured latent space. This guarantees unconditional long-term structural stability and exactly preserves rigid molecular geometries, solving the catastrophic latent collapse typically seen in recurrent models.

## Repository Structure

### Recent Improvements

- **Coordinate-Space Evaluation**: The `_rollout_mse` function has been updated to compute the error on physical (x, y, z) coordinate outputs rather than disjoint latent spaces, providing a standardized, physical metric.
- **BPTT for Baseline**: The GRU Baseline now utilizes Backpropagation Through Time (BPTT) with multi-step rollouts during training to alleviate exposure bias and improve long-term stability.
- **Aligned Evaluation Sample Sizes**: Both models now evaluate their s-step errors by rolling out from all valid $(t, t+s)$ pairs in a trajectory, ensuring perfectly aligned performance comparisons.

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
# Models: 'koopman' or 'gru'
# Molecules: 'ethanol', 'malonaldehyde', or 'aspirin'
docker compose run --build --rm koopman train --molecule ethanol --model koopman --epochs 50
```

**Evaluate trained models:**

```bash
docker compose run --rm koopman eval --molecule ethanol \
  --koopman-ckpt checkpoints/graph_aware_koopman_ethanol_best.pt \
  --gru-ckpt checkpoints/graph_aware_gru_ethanol_best.pt
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
python -m koopman_evolver.cli train --molecule ethanol --model koopman --epochs 50
```

*(Note: The datasets will automatically be downloaded from Kaggle using `kagglehub` the first time you run the code).*

## Key Findings (MD17 Dataset)

We rigorously tested the models across 30-step autonomous prediction horizons:

| Metric | Koopman (Lie-Algebraic) | Unconstrained GRU |
|--------|-------------------------|-------------------|
| **Latent Explosion / Collapse** | **Zero (Graph Energy = 1.000)** | Catastrophic decay |
| **Ethanol Bond Drift** | **~0.005 Å** | ~0.12 Å |
| **Aspirin Rollout MSE** | **0.016** | 0.020 |
| **Aspirin Torsion Drift** | **0.8°** | 1.1° |

**Conclusion:** The Lie-Algebraic Koopman Operator strictly preserves the topology, explicitly learns the underlying vibrational frequencies, and natively respects the mass-frequency scaling laws of physics. It structurally dominates unconstrained Recurrent Neural Networks when modeling the deterministic dynamics of geometric structures.

## Technical Documentation

For the full theoretical breakdown—including the Lie Algebra message-passing formulation, the Isometric Regularization loss, and detailed analysis of the spectral eigenvalue results—please read our comprehensive summary:

👉 **[Project Summary & Mathematical Framework](Graph%20Dynamics%20Learner/Project_Summary_Graph_Aware_Koopman.md)**

## License

MIT License - feel free to use, fork, and modify for your research.
