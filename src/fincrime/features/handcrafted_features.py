import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_handcrafted_features(
    local_features: pd.DataFrame,
    edges: pd.DataFrame,
    timesteps: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build label-free graph features using directed 1-hop and 2-hop neighbours.

    Returns the handcrafted feature matrix and a separate diagnostics DataFrame.
    Nodes without neighbours receive zero-valued neighbour means.
    """

    node_ids = local_features.index
    n_nodes = len(node_ids)

    if not node_ids.is_unique:
        raise ValueError("Transaction IDs in local_features must be unique.")

    if not timesteps.index.is_unique:
        raise ValueError("Transaction IDs in timesteps must be unique.")

    if not node_ids.isin(timesteps.index).all():
        raise ValueError("Some transactions are missing timestep information.")

    required_columns = {"txId1", "txId2"}

    if not required_columns.issubset(edges.columns):
        raise ValueError("edges must contain txId1 and txId2 columns.")

    node_to_idx = pd.Series(np.arange(n_nodes), index=node_ids)

    src = edges["txId1"].map(node_to_idx)
    dst = edges["txId2"].map(node_to_idx)

    if src.isna().any() or dst.isna().any():
        raise ValueError(
            "An edge references a transaction absent from local_features."
        )

    src = src.to_numpy(dtype=np.int64)
    dst = dst.to_numpy(dtype=np.int64)

    source_steps = timesteps.reindex(edges["txId1"]).to_numpy()
    target_steps = timesteps.reindex(edges["txId2"]).to_numpy()

    if pd.isna(source_steps).any() or pd.isna(target_steps).any():
        raise ValueError(
            "An edge references a transaction without timestep data."
        )

    if not np.array_equal(source_steps, target_steps):
        raise ValueError("Cross-timestep edges detected.")

    A = csr_matrix(
        (
            np.ones(len(src), dtype=np.int32),
            (src, dst),
        ),
        shape=(n_nodes, n_nodes),
        dtype=np.int32,
    )

    A.sum_duplicates()
    A.data[:] = 1
    A.eliminate_zeros()

    A2 = (A @ A).tocsr()
    A2.data[:] = 1
    A2.setdiag(0)
    A2.eliminate_zeros()

    X = local_features.to_numpy(dtype=np.float32, copy=True)

    if not np.isfinite(X).all():
        raise ValueError("local_features contains missing or infinite values.")

    def neighbour_mean(adjacency: csr_matrix):
        counts = np.asarray(adjacency.sum(axis=1)).ravel()
        sums = adjacency @ X

        means = np.zeros_like(sums, dtype=np.float32)
        has_neighbours = counts > 0

        means[has_neighbours] = (
            sums[has_neighbours] / counts[has_neighbours, None]
        )

        return means, counts

    incoming_1, in_degree = neighbour_mean(A.T.tocsr())
    outgoing_1, out_degree = neighbour_mean(A)

    incoming_2, in_degree_2 = neighbour_mean(A2.T.tocsr())
    outgoing_2, out_degree_2 = neighbour_mean(A2)

    result = pd.DataFrame(index=node_ids)

    result["hand_in_degree"] = in_degree
    result["hand_out_degree"] = out_degree

    feature_names = local_features.columns

    feature_blocks = [
        ("hand_in1_mean", incoming_1),
        ("hand_out1_mean", outgoing_1),
        ("hand_in2_mean", incoming_2),
        ("hand_out2_mean", outgoing_2),
    ]

    for prefix, values in feature_blocks:
        block = pd.DataFrame(
            values,
            index=node_ids,
            columns=[f"{prefix}_{name}" for name in feature_names],
        )

        result = pd.concat([result, block], axis=1)

    diagnostics = pd.DataFrame(
        {
            "in_degree_2hop_unique": in_degree_2,
            "out_degree_2hop_unique": out_degree_2,
        },
        index=node_ids,
    )

    return result, diagnostics
