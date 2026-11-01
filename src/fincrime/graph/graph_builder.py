import numpy as np
import pandas as pd
import torch


def build_edge_index(
    node_ids,
    edges_df: pd.DataFrame,
    source_col: str = "txId1",
    target_col: str = "txId2",
) -> tuple[torch.Tensor, dict]:
    if not isinstance(edges_df, pd.DataFrame):
        raise TypeError("edges_df must be a pandas DataFrame.")

    if source_col not in edges_df.columns or target_col not in edges_df.columns:
        raise ValueError("Edge DataFrame is missing the required source or target columns.")

    node_index = pd.Index(node_ids)

    if node_index.hasnans:
        raise ValueError("Node IDs must not contain missing values.")

    if node_index.has_duplicates:
        raise ValueError("Node IDs must be unique.")

    if edges_df[[source_col, target_col]].isna().any().any():
        raise ValueError("Edges contain missing source or target node IDs.")

    node_to_idx = {
        node_id: idx
        for idx, node_id in enumerate(node_index)
    }

    source_indices = edges_df[source_col].map(node_to_idx)
    target_indices = edges_df[target_col].map(node_to_idx)

    if source_indices.isna().any() or target_indices.isna().any():
        raise ValueError("An edge references a node that is missing from node_ids.")

    # The shape is of type (2, no_of_edges) instead of the intuitive (no_of_edges, 2) because PyG needs inputs like this
    edge_array = np.vstack([
        source_indices.to_numpy(dtype=np.int64),
        target_indices.to_numpy(dtype=np.int64),
    ])

    edge_index = torch.from_numpy(edge_array)

    return edge_index, node_to_idx