import numpy as np
import torch
from typing import Tuple
import os

from .dataset_split import GraphDatasetSplit
from .md17_adapter import DynamicsDatasetAdapter

class NBodyAdapter(DynamicsDatasetAdapter):
    """
    Adapter for Kipf NRI N-Body simulation datasets.
    Supports both 'charged' (Coulomb forces) and 'springs' systems.
    
    Data format expected (from original NRI GitHub):
    - loc_train.npy: (num_samples, num_atoms, num_dims, num_timesteps) 
    - vel_train.npy: (num_samples, num_atoms, num_dims, num_timesteps)
    - edges_train.npy: (num_samples, num_atoms, num_atoms)
    
    Note: The NRI simulation natively uses 2D spatial dimensions (x,y), resulting 
    in 4D node features (x, y, vx, vy). To maintain seamless compatibility with our 
    existing Molecular Dynamics models, this adapter automatically zero-pads the 
    features to 6D (x, y, 0, vx, vy, 0).
    """

    SYSTEMS = ["charged", "springs"]

    def __init__(
        self,
        data_dir: str,
        system: str = "charged",
        n_particles: int = 5,
        split: str = "train",
        sub_sampling: int = 1,
    ):
        self.data_dir = data_dir
        self.system = system.lower()
        self._n_atoms = n_particles
        self.split_name = split
        self.sub_sampling = sub_sampling

        if self.system not in self.SYSTEMS:
            raise ValueError(f"Unknown system '{self.system}'. Must be one of {self.SYSTEMS}")

    @property
    def name(self) -> str:
        return f"NBody-{self.system.capitalize()}-{self._n_atoms}p"

    @property
    def input_dim(self) -> int:
        return 6  # Padded from 4 to match existing MD models

    def load(self) -> Tuple[GraphDatasetSplit, GraphDatasetSplit]:
        """
        Loads the dataset. Because the NRI format provides pre-split train/valid/test 
        files, we load the requested split and return it twice (as train and test) 
        to satisfy the legacy DynamicsDatasetAdapter signature. 
        """
        loc_path = os.path.join(self.data_dir, f"loc_{self.split_name}_{self.system}{self._n_atoms}.npy")
        vel_path = os.path.join(self.data_dir, f"vel_{self.split_name}_{self.system}{self._n_atoms}.npy")
        edges_path = os.path.join(self.data_dir, f"edges_{self.split_name}_{self.system}{self._n_atoms}.npy")

        if not all(os.path.exists(p) for p in [loc_path, vel_path, edges_path]):
            raise FileNotFoundError(
                f"Missing N-body data files in {self.data_dir}. "
                f"Expected format: loc_{self.split_name}_{self.system}{self._n_atoms}.npy, etc."
            )

        # NRI shapes: (num_samples, num_atoms, dims(2), timesteps(49))
        loc = np.load(loc_path)
        vel = np.load(vel_path)
        edges = np.load(edges_path)

        # Transpose to (num_samples, timesteps, num_atoms, dims)
        loc = np.transpose(loc, [0, 3, 1, 2])
        vel = np.transpose(vel, [0, 3, 1, 2])

        # Subsample time if requested
        loc = loc[:, ::self.sub_sampling, :, :]
        vel = vel[:, ::self.sub_sampling, :, :]

        num_samples, timesteps, n_atoms, _ = loc.shape

        # The NRI graph is fully connected (all particles influence each other)
        # However, diagonal is 0. We'll build a fully connected edge_index.
        # Edges shape is (num_samples, n_atoms, n_atoms), representing interaction strength.
        # We will extract the topology from the first sample.
        adj = np.ones((n_atoms, n_atoms), dtype=bool)
        np.fill_diagonal(adj, False)
        src, dst = np.where(adj)
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        # Build padded node features
        # loc is [x, y]. vel is [vx, vy]. We want [x, y, 0, vx, vy, 0].
        padded_loc = np.zeros((num_samples, timesteps, n_atoms, 3), dtype=np.float32)
        padded_vel = np.zeros((num_samples, timesteps, n_atoms, 3), dtype=np.float32)
        
        padded_loc[:, :, :, :2] = loc
        padded_vel[:, :, :, :2] = vel
        
        node_features = np.concatenate([padded_loc, padded_vel], axis=-1)

        # Compute dynamic edge features (Euclidean distance over time)
        diff = loc[:, :, src] - loc[:, :, dst]  # Shape: (samples, time, num_edges, 2)
        edge_dist = np.sqrt(np.sum(diff ** 2, axis=-1, keepdims=True))  # (samples, time, num_edges, 1)

        # Dummy classification labels (NRI is purely dynamic, no classification task)
        y = np.zeros(num_samples, dtype=np.int64)
        
        # Sequence lengths
        true_lengths = [timesteps] * num_samples

        meta = {
            "dataset": "NBody",
            "system": self.system,
            "D": self.input_dim,
            "n_atoms": self._n_atoms,
            "n_edges": edge_index.shape[1],
            "split": self.split_name,
            "n_trajectories": num_samples,
        }

        split_obj = GraphDatasetSplit(
            node_features=node_features,
            edge_features=edge_dist,
            edge_index=edge_index,
            y=y,
            lengths=true_lengths,
            meta=meta,
        )

        return split_obj, split_obj
