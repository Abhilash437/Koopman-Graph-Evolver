import pytest
import torch
import torch.nn as nn
import numpy as np

from koopman_evolver import GraphAwareKoopmanNet, EquivariantKoopmanNet, EGKN, GraphAwareGRUNet, FlatKoopmanNet


def test_matrix_exp_orthogonality():
    """Verify matrix exponential of skew-symmetric matrix yields an orthogonal matrix K^T K = I."""
    hidden_dim = 16
    A_raw = torch.randn(hidden_dim, hidden_dim)
    A_skew = A_raw - A_raw.T
    K = torch.matrix_exp(A_skew)
    
    # Check K^T K == I
    I = torch.eye(hidden_dim)
    KtK = torch.matmul(K.T, K)
    assert torch.allclose(KtK, I, atol=1e-5), f"Max diff from I: {torch.max(torch.abs(KtK - I))}"


def test_latent_norm_preservation():
    """Verify that multiplication by orthogonal matrix K preserves Euclidean norm ||K z|| = ||z||."""
    hidden_dim = 16
    A_raw = torch.randn(hidden_dim, hidden_dim)
    A_skew = A_raw - A_raw.T
    K = torch.matrix_exp(A_skew)
    
    z0 = torch.randn(4, 9, hidden_dim)  # (Batch=4, N_atoms=9, hidden_dim=16)
    z0_norm = torch.norm(z0, dim=-1)
    
    z1 = torch.matmul(z0, K.T)
    z1_norm = torch.norm(z1, dim=-1)
    
    assert torch.allclose(z0_norm, z1_norm, atol=1e-5), "Latent norm changed after Koopman transition step!"


def test_egkn_alias_identity():
    """Verify EGKN is an alias for EquivariantKoopmanNet."""
    assert EGKN is EquivariantKoopmanNet


def test_graph_aware_koopman_net_forward():
    """Verify GraphAwareKoopmanNet forward pass and rollout shapes."""
    n_atoms = 5
    hidden_dim = 16
    # Simple chain graph edge_index: 0-1, 1-2, 2-3, 3-4
    edge_index = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4],
                              [1, 2, 3, 4, 0, 1, 2, 3]], dtype=torch.long)
    
    model = GraphAwareKoopmanNet(
        edge_index=edge_index,
        node_dim=6,
        edge_dim=1,
        hidden_dim=hidden_dim,
        latent_dim=n_atoms * hidden_dim,
        n_atoms=n_atoms
    )
    
    # Initial state: (B=2, T=5, N=5, hidden_dim=16)
    h0 = torch.randn(2, 5, n_atoms, hidden_dim)
    rollout_steps = 10
    
    rollout = model.forward_rollout(h0, steps=rollout_steps, latent_seed=True)
    assert rollout.shape == (2, rollout_steps, n_atoms, hidden_dim)


def test_equivariant_koopman_net_forward():
    """Verify EquivariantKoopmanNet (E-GKN) forward pass and rollout shapes."""
    n_atoms = 4
    hidden_dim = 16
    edge_index = torch.tensor([[0, 1, 2, 0, 2, 3],
                              [1, 2, 0, 2, 3, 0]], dtype=torch.long)
    
    model = EquivariantKoopmanNet(
        edge_index=edge_index,
        node_dim=6,
        edge_dim=1,
        hidden_dim=hidden_dim,
        latent_dim=n_atoms * hidden_dim,
        n_atoms=n_atoms
    )
    
    z0 = torch.randn(2, 5, n_atoms, hidden_dim)
    rollout_steps = 8
    
    rollout = model.forward_rollout(z0, steps=rollout_steps)
    assert rollout.shape == (2, rollout_steps, n_atoms, hidden_dim)

