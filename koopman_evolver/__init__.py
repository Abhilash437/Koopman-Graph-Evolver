"""
Koopman Graph Evolver (koopman-evolver)
========================================
Geometry-Preserving Latent Dynamics & SE(3)-Equivariant Koopman Operators
for Long-Horizon Graph Simulation.
"""

from .models import (
    GraphAwareKoopmanNet,
    EquivariantKoopmanNet,
    EGKN,
    GraphAwareGRUNet,
    FlatKoopmanNet,
)
from .evaluation import PhysicsEval

__version__ = "0.1.0"

__all__ = [
    "GraphAwareKoopmanNet",
    "EquivariantKoopmanNet",
    "EGKN",
    "GraphAwareGRUNet",
    "FlatKoopmanNet",
    "PhysicsEval",
    "__version__",
]
