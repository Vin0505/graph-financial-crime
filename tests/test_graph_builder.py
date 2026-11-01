import pandas as pd
import pytest
import torch

from fincrime.graph.graph_builder import build_edge_index


@pytest.fixture
def toy_graph():
    node_ids = [105, 210, 315, 420]

    edges_df = pd.DataFrame({
        "txId1": [210, 105, 315],
        "txId2": [315, 210, 420],
    })

    return node_ids, edges_df


def test_edge_index_preserves_node_order(toy_graph):
    node_ids, edges_df = toy_graph

    edge_index, node_to_idx = build_edge_index(node_ids, edges_df)

    assert node_to_idx == {
        105: 0,
        210: 1,
        315: 2,
        420: 3,
    }

    expected = torch.tensor([
        [1, 0, 2],
        [2, 1, 3],
    ], dtype=torch.long)

    assert torch.equal(edge_index, expected)


def test_edge_direction_is_preserved():
    node_ids = [105, 210]

    forward_edges = pd.DataFrame({
        "txId1": [105],
        "txId2": [210],
    })

    reverse_edges = pd.DataFrame({
        "txId1": [210],
        "txId2": [105],
    })

    forward_index, _ = build_edge_index(node_ids, forward_edges)
    reverse_index, _ = build_edge_index(node_ids, reverse_edges)

    assert torch.equal(
        forward_index,
        torch.tensor([[0], [1]], dtype=torch.long),
    )

    assert torch.equal(
        reverse_index,
        torch.tensor([[1], [0]], dtype=torch.long),
    )


def test_missing_node_reference_is_rejected():
    node_ids = [105, 210]

    edges_df = pd.DataFrame({
        "txId1": [105],
        "txId2": [999],
    })

    with pytest.raises(ValueError, match="missing from node_ids"):
        build_edge_index(node_ids, edges_df)


def test_duplicate_node_ids_are_rejected():
    node_ids = [105, 210, 105]

    edges_df = pd.DataFrame({
        "txId1": [105],
        "txId2": [210],
    })

    with pytest.raises(ValueError, match="must be unique"):
        build_edge_index(node_ids, edges_df)


def test_missing_edge_endpoints_are_rejected():
    node_ids = [105, 210]

    edges_df = pd.DataFrame({
        "txId1": [105],
        "txId2": [None],
    })

    with pytest.raises(ValueError, match="missing source or target"):
        build_edge_index(node_ids, edges_df)


def test_missing_edge_columns_are_rejected():
    node_ids = [105, 210]

    edges_df = pd.DataFrame({
        "source": [105],
        "target": [210],
    })

    with pytest.raises(ValueError, match="missing the required"):
        build_edge_index(node_ids, edges_df)


def test_empty_edge_list_returns_correct_shape():
    node_ids = [105, 210]

    edges_df = pd.DataFrame({
        "txId1": pd.Series(dtype="int64"),
        "txId2": pd.Series(dtype="int64"),
    })

    edge_index, node_to_idx = build_edge_index(node_ids, edges_df)

    assert edge_index.shape == (2, 0)
    assert edge_index.dtype == torch.long
    assert node_to_idx == {105: 0, 210: 1}


def test_missing_node_ids_are_rejected():
    node_ids = [105, None]

    edges_df = pd.DataFrame({
        "txId1": [105],
        "txId2": [105],
    })

    with pytest.raises(ValueError, match="Node IDs must not contain missing"):
        build_edge_index(node_ids, edges_df)