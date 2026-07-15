import numpy as np
import torch
import os
import pickle
from typing import Tuple
import pandas as pd

from .dataset_split import GraphDatasetSplit
from .md17_adapter import DynamicsDatasetAdapter

class TrafficAdapter(DynamicsDatasetAdapter):
    """
    Adapter for METR-LA traffic speed forecasting dataset.
    207 sensors (nodes), 5-min intervals.
    
    Expected files in data_dir:
    - metr-la.h5: Time-series speed matrix
    - adj_mx.pkl: Precomputed adjacency matrix
    
    Node features are padded to 6D [speed_norm, delta_speed, 0, 0, 0, 0] 
    to be seamlessly drop-in compatible with the existing Graph Koopman models 
    which were built for MD (x, y, z, vx, vy, vz).
    """

    def __init__(
        self,
        data_dir: str,
        window_len: int = 24, # 2 hours (24 * 5 min)
        train_frac: float = 0.7,
        sub_sampling: int = 1,
    ):
        self.data_dir = data_dir
        self.window_len = window_len
        self.train_frac = train_frac
        self.sub_sampling = sub_sampling

    @property
    def name(self) -> str:
        return f"METR-LA-Traffic-Graph"

    @property
    def input_dim(self) -> int:
        return 6  # Padded to 6 to match MD models

    def load(self) -> Tuple[GraphDatasetSplit, GraphDatasetSplit]:
        h5_path = os.path.join(self.data_dir, "metr-la.h5")
        adj_path = os.path.join(self.data_dir, "adj_mx.pkl")

        if not os.path.exists(h5_path) or not os.path.exists(adj_path):
            raise FileNotFoundError(
                f"Missing METR-LA data in {self.data_dir}. Expected metr-la.h5 and adj_mx.pkl"
            )

        # 1. Load speeds (Time, Sensors)
        df = pd.read_hdf(h5_path)
        speeds = df.values  # (T, 207)
        self._n_atoms = speeds.shape[1]

        # 2. Load Adjacency
        with open(adj_path, 'rb') as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
            
        # 3. Create Edge Index and Features from Adjacency (Static)
        # Threshold to remove weak long-distance edges
        adj_mx[adj_mx < 0.1] = 0
        np.fill_diagonal(adj_mx, 0)
        
        src, dst = np.where(adj_mx > 0)
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        
        # Edge weights (constant over time for traffic)
        edge_weights = adj_mx[src, dst].astype(np.float32)
        # Shape: (1, 1, num_edges, 1) -> we will broadcast to (N, T, num_edges, 1) later
        
        # 4. Normalize and calculate velocity
        speeds = speeds[::self.sub_sampling]
        
        # Standard scaling
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        speeds_norm = (speeds - mean_speed) / (std_speed + 1e-8)
        
        # Delta speed (velocity)
        delta_speeds = np.zeros_like(speeds_norm)
        delta_speeds[1:] = speeds_norm[1:] - speeds_norm[:-1]

        # 5. Windowing
        total_steps = speeds_norm.shape[0]
        n_windows = (total_steps - self.window_len) // self.window_len

        coord_windows = []
        vel_windows = []
        for idx in range(n_windows):
            start = idx * self.window_len
            end = start + self.window_len
            coord_windows.append(speeds_norm[start:end])
            vel_windows.append(delta_speeds[start:end])

        # Convert to numpy arrays
        # Shape: (N_windows, T, N_sensors)
        coord_windows = np.array(coord_windows)
        vel_windows = np.array(vel_windows)

        # 6. Pad to 6D: [speed, delta_speed, 0, 0, 0, 0]
        N, T, V = coord_windows.shape
        node_features = np.zeros((N, T, V, 6), dtype=np.float32)
        node_features[:, :, :, 0] = coord_windows
        node_features[:, :, :, 3] = vel_windows

        # Broadcast static edge weights across N and T
        E = edge_index.shape[1]
        edge_features = np.zeros((N, T, E, 1), dtype=np.float32)
        edge_features[:, :, :, 0] = edge_weights

        # 7. Split train / test
        split_idx = int(N * self.train_frac)
        
        train_nodes = node_features[:split_idx]
        train_edges = edge_features[:split_idx]
        test_nodes = node_features[split_idx:]
        test_edges = edge_features[split_idx:]

        dummy_y_train = np.zeros(train_nodes.shape[0], dtype=np.int64)
        dummy_y_test = np.zeros(test_nodes.shape[0], dtype=np.int64)
        
        train_lengths = [T] * train_nodes.shape[0]
        test_lengths = [T] * test_nodes.shape[0]

        train_meta = {"dataset": "METR-LA", "split": "train", "n_atoms": V, "n_edges": E}
        test_meta = {"dataset": "METR-LA", "split": "test", "n_atoms": V, "n_edges": E}

        train_split = GraphDatasetSplit(train_nodes, edge_index, train_edges, dummy_y_train, train_lengths, train_meta)
        test_split = GraphDatasetSplit(test_nodes, edge_index, test_edges, dummy_y_test, test_lengths, test_meta)

        return train_split, test_split
