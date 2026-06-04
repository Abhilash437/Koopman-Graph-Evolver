# Koopman Graph Evolver: Molecular Dynamics

A deep learning framework for predicting deterministic, long-horizon molecular dynamics using Graph-Aware Koopman operator theory and Lie-algebraic message passing.

## Overview

This project implements a **Graph-Aware Koopman Autoencoder** to model the global dynamics of molecules from the MD17 dataset (e.g., Ethanol, Malonaldehyde, Aspirin). It compares the performance of mathematically constrained linear Koopman transitions against unconstrained, highly non-linear GRU baselines on 30-step autonomous rollout trajectories.

**Key Innovation:** Applying Lie-algebraic constraints (projecting the Koopman operator onto the orthogonal group) to a graph-structured latent space. This guarantees unconditional long-term structural stability and exactly preserves rigid molecular geometries, solving the catastrophic latent collapse typically seen in recurrent models.

## Repository Structure

```
.
├── Graph Dynamics Learner/
│   ├── generalized_kgn.ipynb                               # Main notebook with MD17 data loading, training, and evaluation
│   ├── dynamics-learner-rmd17.ipynb                        # Secondary MD17 experiment notebook
│   └── Project_Summary_Graph_Aware_Koopman.md              # Comprehensive technical documentation & mathematical framework
├── Phase 1/                                                # Early graph evolution experiments (SBMs)
├── Phase 2/                                                # Graph-RNNs and non-linear baseline explorations
├── Phase 3/                                                # Scaling linear transitions and Koopman theory
├── Docs/                                                   # Legacy technical and planning documentation
└── README.md                                               # This file
```

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, Matplotlib
- kagglehub (for downloading MD17 data)

Install dependencies:

```bash
pip install torch numpy matplotlib kagglehub
```

### Running the Code

Open the Jupyter notebook and run the cells sequentially:

```bash
jupyter notebook "Graph Dynamics Learner/generalized_kgn.ipynb"
```

The notebook includes:

1. **Data Pipeline**: `MD17AdapterV2` for downloading, rotating, and translating raw coordinate trajectories into sliding windows of molecular graphs.
2. **Model Architectures**:
   - `GraphAwareKoopmanNet` (Linear, Lie-Algebra constrained)
   - `GraphAwareGRUNet` (Non-linear baseline)
3. **Training Engine**: `GraphTrainer` for batching, optimization, and checkpointing.
4. **Deep Physical Diagnostics**: `PhysicsEval` suite comparing real-world drift of Bond Lengths, Bond Angles, and Torsions.

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
