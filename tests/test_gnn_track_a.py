import numpy as np
import torch

from fincrime.gnn.track_a_runner import (
    _calculate_ranking_metrics,
    _select_validation_threshold,
    _symmetrise_edge_index,
)


def test_symmetrise_edge_index_adds_reverse_edges():
    edge_index = torch.tensor([
        [0, 1, 1],
        [1, 0, 2],
    ])

    result = _symmetrise_edge_index(
        edge_index=edge_index,
        num_nodes=3,
    )

    actual_edges = set(map(tuple, result.t().tolist()))

    expected_edges = {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }

    assert actual_edges == expected_edges
    assert result.shape == (2, 4)


def test_symmetrise_edge_index_does_not_duplicate_edges():
    edge_index = torch.tensor([
        [0, 1],
        [1, 0],
    ])

    result = _symmetrise_edge_index(
        edge_index=edge_index,
        num_nodes=2,
    )

    assert result.shape == (2, 2)


def test_select_validation_threshold_returns_valid_metrics():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    result = _select_validation_threshold(
        labels=labels,
        probabilities=probabilities,
    )

    assert 0.0 <= result["threshold"] <= 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert np.isclose(result["f1"], 1.0)
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0


def test_ranking_metrics_calculate_precision_and_recall_at_k():
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    probabilities = np.array([
        0.99, 0.95, 0.90, 0.85, 0.80,
        0.75, 0.70, 0.65, 0.60, 0.55,
    ])

    result = _calculate_ranking_metrics(
        labels=labels,
        probabilities=probabilities,
    )

    assert result["average_precision"] > 0
    assert result["roc_auc"] > 0

    assert result["precision_at_10pct"] == 1.0
    assert result["recall_at_10pct"] == 1 / 3

    assert result["precision_at_5pct"] == 1.0
    assert result["recall_at_5pct"] == 1 / 3

    assert result["precision_at_1pct"] == 1.0
    assert result["recall_at_1pct"] == 1 / 3
