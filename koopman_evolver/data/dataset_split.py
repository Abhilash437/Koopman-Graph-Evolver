import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from abc import ABC, abstractmethod
import networkx as nx
import os
import urllib.request
import zipfile
import math
import warnings

# PyTorch Geometric
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import to_dense_batch
except ImportError:
    pass


@dataclass
class DatasetSplit:
    """
    Standard container produced by every adapter.

    Fields
    ------
    X       np.ndarray  shape (N, T, D)
                N = number of trajectory windows
                T = max sequence length across the split (shorter ones padded
                    with zeros to this length)
                D = feature dimension per timestep

    y       np.ndarray  shape (N,)  int64
                Class label per trajectory.
                If the dataset has no meaningful labels, this is all zeros.

    lengths List[int]   length N
                True (unpadded) length of each trajectory.
                The model uses these to ignore the zero-padded tail.

    meta    dict
                Dataset-specific metadata for logging and plot titles.
                Guaranteed keys: 'dataset', 'molecule', 'D', 'split'
    """
    X      : np.ndarray
    y      : np.ndarray
    lengths: List[int]
    meta   : dict = field(default_factory=dict)


@dataclass
class GraphDatasetSplit:
    """
    Data split container for molecular graphs.

    Fields
    ------
    node_features : np.ndarray  shape (N, T, n_atoms, 6)
                    Positions [x, y, z] concatenated with finite-difference
                    velocities [vx, vy, vz].
    edge_index    : torch.Tensor shape (2, n_edges)
                    Long tensor representing the fixed molecular graph.
    edge_features : np.ndarray  shape (N, T, n_edges, 1)
                    Evolving bond distances for each edge in edge_index.
    y             : np.ndarray  shape (N,)
                    Median-split energy classification labels (retained for signature).
    lengths       : List[int]   length N
                    True (unpadded) sequence lengths.
    meta          : dict
                    Logging and visualization metadata.
    """
    node_features: np.ndarray
    edge_index: torch.Tensor
    edge_features: np.ndarray
    y: np.ndarray
    lengths: List[int]
    meta: dict = field(default_factory=dict)


