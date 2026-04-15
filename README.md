# Koopman Graph Evolver

A deep learning framework for predicting graph evolution using Koopman operator theory in latent space.

## Overview

This project implements a Graph Neural Network (GNN) autoencoder combined with linear Koopman operators to model discrete-time graph evolution. It compares the performance of linear Koopman transitions against non-linear MLP baselines on stochastic block model (SBM) graph sequences.

**Key Innovation**: Applying Koopman operator theory—traditionally used in continuous dynamical systems—to discrete graph-structured data via a learned latent space embedding.

## Repository Structure

```
.
├── graph-evolution-1.ipynb    # Main Jupyter notebook with experiments
├── TECHNICAL_DOCS.md          # In-depth technical documentation
└── README.md                  # This file
```

## Quick Start

### Requirements

- Python 3.8+
- PyTorch 2.10+
- PyTorch Geometric 2.7+
- NetworkX
- NumPy, Matplotlib

Install dependencies:
```bash
pip install torch torch_geometric networkx numpy matplotlib
```

### Running Experiments

Open the Jupyter notebook and run cells sequentially:
```bash
jupyter notebook graph-evolution-1.ipynb
```

The notebook includes:
1. **Data Generation**: Synthetic SBM graphs with controlled evolution
2. **Model Training**: GCN encoder/decoder with MLP or Koopman transitions
3. **Evaluation**: Multi-step rollout accuracy and F1 scores
4. **Ablation Study**: Testing different latent dimensions across multiple seeds

## Results Summary

| Metric | MLP | Koopman |
|--------|-----|---------|
| Parameters (Transition) | 2,128 | **256** (8× fewer) |
| Step-1 Accuracy | 91.3% | 91.3% |
| Step-15 Accuracy | 55.4% | **64.9%** |
| Accuracy Drop | 35.8% | **26.4%** |

**Key Finding**: The Koopman operator achieves superior long-term stability with significantly fewer parameters, demonstrating that linear dynamics in an appropriate latent space can outperform complex non-linear models for graph evolution prediction.

## Technical Documentation

For detailed mathematical foundations, model architectures, and implementation details, see [`TECHNICAL_DOCS.md`](TECHNICAL_DOCS.md).

Topics covered:
- Stochastic Block Model (SBM) graph generation
- Koopman operator theory in finite-dimensional approximation
- GCN encoder and MLP decoder architectures
- Spectral stability constraints
- Multi-step rollout evaluation methodology

## Citation

If you use this code, please cite:

```bibtex
@article{meys2019ablation,
  title={Ablation Studies in Artificial Neural Networks},
  author={Meyes, Richard and Lu, Melanie and Paisley, Christopher W and Meisen, Tobias and Runger, Gitta},
  journal={arXiv preprint arXiv:1901.08644},
  year={2019}
}
```

## License

MIT License - feel free to use and modify for your research.
