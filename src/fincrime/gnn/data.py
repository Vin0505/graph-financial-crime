import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from fincrime.evaluation.splits import (
    build_label_masks,
    build_split_masks,
)
from fincrime.graph.graph_builder import build_edge_index


def build_gnn_graph_structure(
    labels: pd.Series,
    timesteps: pd.Series,
    edges_df: pd.DataFrame,
    split_config: dict,
    protocol: str = "tuning",
    track_b_name: str | None = None,
    source_col: str = "txId1",
    target_col: str = "txId2",
) -> dict:
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series.")
    if not isinstance(timesteps, pd.Series):
        raise TypeError("timesteps must be a pandas Series.")
    if not labels.index.equals(timesteps.index):
        raise ValueError("Labels and timesteps must have identical indices.")
    if labels.index.has_duplicates:
        raise ValueError("Transaction IDs must be unique.")

    split_masks = build_split_masks(
        timesteps=timesteps,
        split_config=split_config,
        protocol=protocol,
        track_b_name=track_b_name,
    )
    label_masks = build_label_masks(
        labels=labels,
        split_masks=split_masks,
        label_config=split_config["labels"],
    )

    # All nodes in fit, validation and test periods remain in the graph.
    # Unknown labels are retained as graph nodes, but excluded from loss
    # and labelled evaluation masks.
    graph_node_mask = split_masks["fit"] | split_masks["validation"]
    if split_masks["test"] is not None:
        graph_node_mask = graph_node_mask | split_masks["test"]

    node_ids = labels.index[graph_node_mask]
    if len(node_ids) == 0:
        raise ValueError("No nodes found in the selected graph period.")

    node_id_set = set(node_ids)
    edge_mask = (
        edges_df[source_col].isin(node_id_set)
        & edges_df[target_col].isin(node_id_set)
    )
    graph_edges = edges_df.loc[edge_mask, [source_col, target_col]].copy()

    edge_index, node_to_idx = build_edge_index(
        node_ids=node_ids,
        edges_df=graph_edges,
        source_col=source_col,
        target_col=target_col,
    )

    node_labels = labels.loc[node_ids]
    binary_labels = node_labels.map({
        split_config["labels"]["illicit"]: 1,
        split_config["labels"]["licit"]: 0,
    }).fillna(-1).to_numpy(dtype=np.int64)

    train_mask = label_masks["train_loss"].loc[node_ids].to_numpy()
    validation_mask = label_masks["validation_eval"].loc[node_ids].to_numpy()

    graph_structure = {
        "source_node_ids": labels.index.copy(),
        "node_ids": node_ids,
        "node_to_idx": node_to_idx,
        "edge_index": edge_index,
        "y": torch.from_numpy(binary_labels),
        "train_mask": torch.from_numpy(train_mask.astype(bool)),
        "validation_mask": torch.from_numpy(validation_mask.astype(bool)),
        "protocol": protocol,
        "num_nodes": len(node_ids),
        "num_edges": edge_index.shape[1],
    }

    if label_masks["test_eval"] is not None:
        test_mask = label_masks["test_eval"].loc[node_ids].to_numpy()
        graph_structure["test_mask"] = torch.from_numpy(
            test_mask.astype(bool)
        )

    return graph_structure


def attach_node_features(
    graph_structure: dict,
    features: pd.DataFrame,
) -> Data:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if not features.index.equals(graph_structure["source_node_ids"]):
        raise ValueError(
            "Feature indices must match the original transaction ordering."
        )
    if features.columns.has_duplicates:
        raise ValueError("Feature columns must be unique.")
    if any(str(col).lower() == "timestep" for col in features.columns):
        raise ValueError("Timestep must not be included as a model feature.")

    selected_features = features.loc[graph_structure["node_ids"]]
    feature_array = selected_features.to_numpy(dtype=np.float32, copy=True)

    if not np.isfinite(feature_array).all():
        raise ValueError("Node features contain missing or infinite values.")

    return Data(
        x=torch.from_numpy(feature_array),
        edge_index=graph_structure["edge_index"],
        y=graph_structure["y"],
        train_mask=graph_structure["train_mask"],
        val_mask=graph_structure["validation_mask"],
        test_mask=graph_structure.get("test_mask"),
    )