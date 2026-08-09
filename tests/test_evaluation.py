import pytest
import torch
import numpy as np
from koopman_evolver import PhysicsEval, GraphAwareKoopmanNet, GraphAwareGRUNet


def test_physics_eval_topology_extraction():
    """Verify PhysicsEval extracts correct bonds, angles, and torsions from edge_index."""
    n_atoms = 4
    # Ring of 4 nodes: 0-1, 1-2, 2-3, 3-0
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                              [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
    
    koop_model = GraphAwareKoopmanNet(edge_index=edge_index, n_atoms=n_atoms)
    gru_model = GraphAwareGRUNet(edge_index=edge_index, n_atoms=n_atoms)
    
    evaluator = PhysicsEval(koop_model=koop_model, gru_model=gru_model, test_split=None, n_atoms=n_atoms, molecule_name="synthetic")
    bonds, angles, torsions = evaluator.extract_topology(edge_index)
    
    assert len(bonds) == 4, f"Expected 4 bonds in a 4-cycle graph, got {len(bonds)}"
    assert len(angles) == 4, f"Expected 4 angles in a 4-cycle graph, got {len(angles)}"


def test_physics_eval_compute_angles():
    """Verify angle calculation for a 90 degree right triangle."""
    n_atoms = 3
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    koop_model = GraphAwareKoopmanNet(edge_index=edge_index, n_atoms=n_atoms)
    gru_model = GraphAwareGRUNet(edge_index=edge_index, n_atoms=n_atoms)
    
    evaluator = PhysicsEval(koop_model=koop_model, gru_model=gru_model, test_split=None, n_atoms=n_atoms, molecule_name="triangle")
    
    # 3 atoms in 2D forming right angle at atom 1:
    # atom 0: (1, 0, 0), atom 1: (0, 0, 0), atom 2: (0, 1, 0)
    coords = torch.tensor([
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]]
    ])  # (B=1, N=3, 3)
    
    angles = [(0, 1, 2)]  # angle between (1,0,0)-(0,0,0)-(0,1,0) should be 90 degrees
    angle_vals = evaluator.compute_angles(coords, angles)
    
    assert angle_vals is not None
    assert torch.allclose(angle_vals, torch.tensor([90.0]), atol=1e-3)
