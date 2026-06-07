import os
import urllib.request
import numpy as np
import torch
from typing import Tuple

from .dataset_split import GraphDatasetSplit
from .md17_adapter import DynamicsDatasetAdapter

class MD22Adapter(DynamicsDatasetAdapter):
    """
    Graph-based adapter for the MD22 molecular dynamics benchmark.
    Applies Principal Axis alignment (SVD) to scale robustly to large peptides 
    like Ac-Ala3-NHMe (42 atoms).
    """

    _ATOM_COUNTS = {
        "ac-ala3-nhme": 42
    }

    def __init__(
        self,
        path: str,
        molecule: str = "ac-ala3-nhme",
        sub_sampling: int = 10,
        window_len: int = 150,
        train_frac: float = 0.8,
        bond_cutoff: float = 1.6,
    ):
        self.path = path
        self.molecule = molecule.lower()
        
        # MD22 trajectories are very dense (fs resolution), a larger sub_sampling is recommended
        # compared to MD17 to capture meaningful macroscopic dynamics over a window.
        self.sub_sampling = sub_sampling
        
        self.window_len = window_len
        self.train_frac = train_frac
        self.bond_cutoff = bond_cutoff

        if self.molecule in self._ATOM_COUNTS:
            self._n_atoms = self._ATOM_COUNTS[self.molecule]
        else:
            self._n_atoms = None

    @property
    def name(self) -> str:
        return f"MD22-{self.molecule}-Graph"

    @property
    def input_dim(self) -> int:
        return 6  # [x, y, z, vx, vy, vz]

    def load(self) -> Tuple[GraphDatasetSplit, GraphDatasetSplit]:
        # 1. Read raw file
        raw = np.load(self.path, allow_pickle=True)
        coords = raw["R"] if "R" in raw else raw["coords"]
        energies = raw["E"] if "E" in raw else raw["energies"]

        # Sort chronologically if needed (usually MD22 is already chronological, but safe to check)
        if "old_indices" in raw:
            sorted_idx = np.argsort(raw["old_indices"])
            coords = coords[sorted_idx]
            energies = energies[sorted_idx]

        if "z" in raw:
            self._n_atoms = int(raw["z"].shape[0])
        elif "nuclear_charges" in raw:
            self._n_atoms = int(raw["nuclear_charges"].shape[0])

        n_frames, n_atoms, _ = coords.shape
        if self._n_atoms is None:
            self._n_atoms = n_atoms

        # 2. Sub-sample frames
        coords   = coords[::self.sub_sampling]
        energies = energies[::self.sub_sampling]

        # 3. Dynamic Bond Detection (on first frame before centering)
        # 1.6 cutoff is standard for strict covalent bonds, even in large peptides
        diff0 = coords[0, :, None, :] - coords[0, None, :, :]
        dist0 = np.sqrt(np.sum(diff0 ** 2, axis=-1))
        adj = dist0 < self.bond_cutoff
        np.fill_diagonal(adj, False)
        src, dst = np.where(adj)
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        # 4. Partition sub-sampled trajectory into windows
        coord_windows, labels = self._make_coordinate_windows(coords, energies)

        # 5. Split train / test
        split_idx = int(len(coord_windows) * self.train_frac)
        train_coords = coord_windows[:split_idx]
        train_labels = labels[:split_idx]
        test_coords  = coord_windows[split_idx:]
        test_labels  = labels[split_idx:]

        # 6. Apply centering/alignment and finite-difference velocities
        train_split = self._build_graph_split(train_coords, train_labels, edge_index, split="train")
        test_split  = self._build_graph_split(test_coords,  test_labels,  edge_index, split="test")

        return train_split, test_split

    def _make_coordinate_windows(self, coords, energies):
        total_steps = coords.shape[0]
        n_windows   = (total_steps - self.window_len) // self.window_len

        threshold     = np.median(energies)
        coord_windows = []
        labels        = []

        for idx in range(n_windows):
            start = idx * self.window_len
            end   = start + self.window_len
            coord_windows.append(coords[start:end])
            mean_energy = np.mean(energies[start:end])
            labels.append(1 if mean_energy > threshold else 0)

        return coord_windows, labels

    def _build_graph_split(self, coord_windows, labels, edge_index, split):
        node_feats_list = []
        edge_feats_list = []
        true_lengths    = []

        src, dst = edge_index.numpy()

        for win in coord_windows:
            # win shape: (window_len, n_atoms, 3)
            # Principal Axis Alignment via SVD
            P0 = win[0]
            P0_centered = P0 - P0.mean(axis=0, keepdims=True)
            C0 = np.dot(P0_centered.T, P0_centered)
            U0, _, _ = np.linalg.svd(C0)

            # Resolve principal axis sign flip ambiguity using the first frame
            proj_first = np.dot(P0_centered[0], U0)
            for col in range(3):
                if proj_first[col] < 0:
                    U0[:, col] *= -1

            win_aligned = []
            for t in range(win.shape[0]):
                P = win[t]  # (n_atoms, 3)
                P_centered = P - P.mean(axis=0, keepdims=True)
                win_aligned.append(np.dot(P_centered, U0))

            win_aligned = np.array(win_aligned)  # (window_len, n_atoms, 3)

            pos = win_aligned[1:]                                    # (window_len - 1, n_atoms, 3)
            vel = win_aligned[1:] - win_aligned[:-1]                 # (window_len - 1, n_atoms, 3)
            node_feat = np.concatenate([pos, vel], axis=-1)          # (window_len - 1, n_atoms, 6)
            node_feats_list.append(node_feat)

            # Compute evolving edge distances over rollout time
            diff = pos[:, src] - pos[:, dst]
            edge_dist = np.sqrt(np.sum(diff ** 2, axis=-1, keepdims=True))
            edge_feats_list.append(edge_dist)

            true_lengths.append(pos.shape[0])

        max_len   = max(true_lengths)
        n_windows = len(coord_windows)
        n_edges   = edge_index.shape[1]

        node_features_padded = np.zeros((n_windows, max_len, self._n_atoms, 6), dtype=np.float32)
        edge_features_padded = np.zeros((n_windows, max_len, n_edges, 1), dtype=np.float32)

        for i in range(n_windows):
            node_features_padded[i, :true_lengths[i]] = node_feats_list[i]
            edge_features_padded[i, :true_lengths[i]] = edge_feats_list[i]

        y = np.array(labels, dtype=np.int64)

        meta = {
            "dataset": "MD22",
            "molecule": self.molecule,
            "D": 6,
            "n_atoms": self._n_atoms,
            "n_edges": n_edges,
            "window_len": self.window_len,
            "sub_sampling": self.sub_sampling,
            "split": split,
            "n_trajectories": n_windows,
        }

        return GraphDatasetSplit(
            node_features = node_features_padded,
            edge_features = edge_features_padded,
            edge_index    = edge_index,
            y             = y,
            lengths       = true_lengths,
            meta          = meta,
        )
