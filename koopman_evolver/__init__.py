"""
Koopman Graph Evolver (koopman-evolver)
========================================
Beyond MSE: Orthogonal Latent Dynamics for Long-Horizon Graph Simulation.
"""

from .models import (
    GraphAwareKoopmanNet,
    EquivariantKoopmanNet,
    EGKN,
    GraphAwareGRUNet,
    FlatKoopmanNet,
)
from .evaluation import PhysicsEval

__version__ = "0.1.1"

__all__ = [
    "GraphAwareKoopmanNet",
    "EquivariantKoopmanNet",
    "EGKN",
    "GraphAwareGRUNet",
    "FlatKoopmanNet",
    "PhysicsEval",
    "__version__",
]
