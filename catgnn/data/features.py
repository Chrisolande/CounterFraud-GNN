import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler


def build_features(
    csv_path: str, cache_path: str = "sffsd_ai4risk_preprocessed.pt"
) -> dict:
    """Extract 122-dim temporal multi-window features and relational graph adjacency."""
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

    # Multi-window statistical features: [avg, tot, std, bias, count, num_tgt, num_loc, num_type]
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

    # Construct temporal multi-relational edges
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
