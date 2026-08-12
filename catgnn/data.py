import os
from collections import defaultdict
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


def build_features(
    csv_path: str, cache_path: str = "sffsd_ai4risk_preprocessed.pt"
) -> dict:
    """Generate the temporal multi-window feature maps (122 dimensions) grouped per 
    source account and multi-relational temporal adjacency on the full S-FFSD graph"""

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}.")

    df = pd.read_csv(csv_path).sort_values("Time").reset_index(drop=True)
    num_nodes = len(df)
    time_spans = [2, 3, 5, 15, 20, 50, 100, 150, 200, 300, 864, 2590, 5100, 10000, 24000]
    num_spans = len(time_spans)

    fe_matrix = np.zeros((num_nodes, num_spans * 8 + 2), dtype=np.float32)
    times = df["Time"].values
    amts = df["Amount"].values.astype(np.float32)

    fe_matrix[:, 0] = amts
    fe_matrix[:, 1] = np.log1p(np.maximum(0.0, amts))

    tgt_codes = pd.factorize(df["Target"])[0]
    loc_codes = pd.factorize(df["Location"])[0]
    type_codes = pd.factorize(df["Type"])[0]

    source_to_indices = defaultdict(list)
    for idx, s in enumerate(df["Source"].values):
        source_to_indices[s].append(idx)

    for s, idxs in source_to_indices.items():
        if len(idxs) == 1:
            i = idxs[0]
            amt_i = amts[i]
            for s_idx in range(num_spans):
                base_col = 2 + s_idx * 8
                fe_matrix[i, base_col + 0] = amt_i
                fe_matrix[i, base_col + 1] = amt_i
                fe_matrix[i, base_col + 2] = 0.0
                fe_matrix[i, base_col + 3] = 0.0
                fe_matrix[i, base_col + 4] = 1.0
                fe_matrix[i, base_col + 5] = 1.0
                fe_matrix[i, base_col + 6] = 1.0
                fe_matrix[i, base_col + 7] = 1.0
        else:
            idxs = np.array(idxs, dtype=np.int32)
            group_times = times[idxs]
            group_amts = amts[idxs]
            group_tgts = tgt_codes[idxs]
            group_locs = loc_codes[idxs]
            group_types = type_codes[idxs]
            m = len(idxs)

            for s_idx, length in enumerate(time_spans):
                base_col = 2 + s_idx * 8
                left_bounds = np.searchsorted(group_times, group_times - length, side="left")
                for pos in range(m):
                    orig_idx = idxs[pos]
                    lb = left_bounds[pos]
                    window_amts = group_amts[lb : pos + 1]
                    cnt = len(window_amts)
                    tot = float(np.sum(window_amts))
                    avg = tot / cnt
                    std = float(np.std(window_amts)) if cnt > 1 else 0.0
                    bias = float(group_amts[pos] - avg)

                    num_tgt = float(len(np.unique(group_tgts[lb : pos + 1])))
                    num_loc = float(len(np.unique(group_locs[lb : pos + 1])))
                    num_type = float(len(np.unique(group_types[lb : pos + 1])))

                    fe_matrix[orig_idx, base_col + 0] = avg
                    fe_matrix[orig_idx, base_col + 1] = tot
                    fe_matrix[orig_idx, base_col + 2] = std
                    fe_matrix[orig_idx, base_col + 3] = bias
                    fe_matrix[orig_idx, base_col + 4] = float(cnt)
                    fe_matrix[orig_idx, base_col + 5] = num_tgt
                    fe_matrix[orig_idx, base_col + 6] = num_loc
                    fe_matrix[orig_idx, base_col + 7] = num_type

    scaler = StandardScaler()
    scaled_cont = scaler.fit_transform(fe_matrix)

    cat_cols = ["Location", "Type"]
    cat_dims = []
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_dims.append(len(le.classes_))

    adj_list: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    pair = ["Source", "Target", "Location", "Type"]
    for rel_id, column in enumerate(pair):
        entity_to_txs = defaultdict(list)
        for tx_idx, entity_id in enumerate(df[column].values):
            entity_to_txs[entity_id].append(tx_idx)
        for tx_indices in entity_to_txs.values():
            if len(tx_indices) > 1:
                for idx_pos, i in enumerate(tx_indices):
                    prior_txs = tx_indices[max(0, idx_pos - 10) : idx_pos]
                    for p in prior_txs:
                        adj_list[i].append((p, rel_id))

    times_raw = df["Time"].values
    for i in range(num_nodes):
        seen: set[int] = set()
        deduped: list[tuple[int, int]] = []
        sorted_neighbors = sorted(adj_list[i], key=lambda item: times_raw[item[0]], reverse=True)
        for nbr, rel in sorted_neighbors:
            if nbr not in seen:
                seen.add(nbr)
                deduped.append((nbr, rel))
        adj_list[i] = deduped

    labeled_indices = np.where(df["Labels"].isin([0, 1]).values)[0]

    data_dict = {
        "X_cont": torch.tensor(scaled_cont, dtype=torch.float32),
        "X_cat": torch.tensor(df[cat_cols].values, dtype=torch.long),
        "Y": torch.tensor(df["Labels"].values, dtype=torch.float32),
        "adj": adj_list,
        "times": torch.tensor(times_raw, dtype=torch.float32),
        "cat_dims": cat_dims,
        "cont_dim": scaled_cont.shape[1],
        "labeled_indices": labeled_indices,
    }
    torch.save(data_dict, cache_path)
    return data_dict


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
        rel_ids = torch.full((self.max_neighbors,), 4, dtype=torch.long)  # 4 is padding relation
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


class FraudGraphDataModule(pl.LightningDataModule):
    """DataModule following the CaT-GNN paper with train/test split only."""

    def __init__(
        self,
        data_path: str = "S-FFSD.csv",
        max_neighbors: int = 20,
        train_ratio: float = 0.7,
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dict = None

    def setup(self, stage: str | None = None):
        if self.data_dict is None:
            self.data_dict = build_features(self.hparams.data_path)
            self.cont_dim = self.data_dict["cont_dim"]
            self.cat_dims = self.data_dict["cat_dims"]

            labeled_idx = self.data_dict["labeled_indices"]
            num_labeled = len(labeled_idx)
            train_end = int(num_labeled * self.hparams.train_ratio)

            self.train_idx = labeled_idx[:train_end]
            self.test_idx = labeled_idx[train_end:]

            train_labels = self.data_dict["Y"][self.train_idx]
            num_pos = train_labels.sum().item()
            num_neg = len(train_labels) - num_pos
            self.pos_weight = num_neg / max(num_pos, 1.0)

        self.train_ds = SFFSDGraphDataset(
            self.data_dict, self.train_idx, self.hparams.max_neighbors
        )
        self.test_ds = SFFSDGraphDataset(
            self.data_dict, self.test_idx, self.hparams.max_neighbors
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):

        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )