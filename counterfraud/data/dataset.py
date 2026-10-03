import numpy as np
import torch
from torch.utils.data import Dataset


class SFFSDGraphDataset(Dataset):
    def __init__(self, data_dict: dict, split_indices: np.ndarray, max_neighbors: int):
        self.max_neighbors = max_neighbors
        self.indices = split_indices
        self.X_cont = data_dict["X_cont"]
        self.X_cat = data_dict["X_cat"]
        self.Y = data_dict["Y"]
        self.adj_list = data_dict["adj"]
        self.times = data_dict["times"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        node_id = self.indices[idx]
        x_target_cont = self.X_cont[node_id]
        x_target_cat = self.X_cat[node_id]
        y = self.Y[node_id]
        t_target = self.times[node_id]

        neighbors = self.adj_list[node_id]
        sampled = neighbors[: self.max_neighbors]
        actual_M = len(sampled)

        x_neighbor_cont = torch.zeros(self.max_neighbors, self.X_cont.shape[1])
        x_neighbor_cat = torch.zeros(self.max_neighbors, self.X_cat.shape[1], dtype=torch.long)
        t_gap = torch.zeros(self.max_neighbors)
        rel_ids = torch.full((self.max_neighbors,), 4, dtype=torch.long)
        valid_mask = torch.zeros(self.max_neighbors, dtype=torch.bool)

        if actual_M > 0:
            sampled_nodes = [item[0] for item in sampled]
            sampled_rels = [item[1] for item in sampled]
            sampled_idx = np.array(sampled_nodes, dtype=np.int64)

            x_neighbor_cont[:actual_M] = self.X_cont[sampled_idx]
            x_neighbor_cat[:actual_M] = self.X_cat[sampled_idx]
            t_gap[:actual_M] = torch.abs(t_target - self.times[sampled_idx])
            rel_ids[:actual_M] = torch.tensor(sampled_rels, dtype=torch.long)
            valid_mask[:actual_M] = True

        return (
            x_target_cont,
            x_target_cat,
            x_neighbor_cont,
            x_neighbor_cat,
            t_gap,
            rel_ids,
            valid_mask,
            y,
        )
