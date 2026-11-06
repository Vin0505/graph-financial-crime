import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from fincrime.gnn.training import (
    calculate_class_weights,
    set_random_seeds,
    train_gnn_model,
)


def create_synthetic_graph() -> Data:
    torch.manual_seed(42)

    x = torch.randn(12, 4)

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6],
            [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6, 2, 4, 0, 8],
        ],
        dtype=torch.long,
    )

    y = torch.tensor(
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        dtype=torch.long,
    )

    train_mask = torch.tensor(
        [True, True, True, True, True, True,
         False, False, False, False, False, False]
    )

    val_mask = torch.tensor(
        [False, False, False, False, False, False,
         True, True, True, True, True, True]
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
    )


def test_class_weights_are_calculated_correctly():
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    train_mask = torch.ones(8, dtype=torch.bool)

    weights = calculate_class_weights(
        y=y,
        train_mask=train_mask,
        device=torch.device("cpu"),
    )

    expected = torch.tensor([8 / 6, 8 / 10])

    assert torch.allclose(weights, expected)


def test_class_weights_use_training_nodes_only():
    y = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    train_mask = torch.tensor(
        [True, True, True, True, True, True,
         False, False, False, False]
    )

    weights = calculate_class_weights(
        y=y,
        train_mask=train_mask,
        device=torch.device("cpu"),
    )

    expected = torch.tensor([1.0, 1.0])

    assert torch.allclose(weights, expected)


def test_class_weights_reject_empty_training_mask():
    y = torch.tensor([0, 1, 0, 1])
    train_mask = torch.zeros(4, dtype=torch.bool)

    with pytest.raises(ValueError, match="No labelled training nodes"):
        calculate_class_weights(
            y=y,
            train_mask=train_mask,
            device=torch.device("cpu"),
        )


def test_class_weights_reject_missing_class():
    y = torch.tensor([0, 0, 0, 0])
    train_mask = torch.ones(4, dtype=torch.bool)

    with pytest.raises(ValueError, match="Both classes"):
        calculate_class_weights(
            y=y,
            train_mask=train_mask,
            device=torch.device("cpu"),
        )


def test_random_seeds_reproduce_torch_random_values():
    set_random_seeds(42)
    first = torch.rand(5)

    set_random_seeds(42)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_training_returns_valid_results():
    data = create_synthetic_graph()

    result = train_gnn_model(
        data=data,
        model_name="gcn",
        hidden_channels=8,
        dropout=0.2,
        learning_rate=0.01,
        max_epochs=5,
        patience=5,
        seed=42,
        device="cpu",
    )

    assert result["best_epoch"] >= 1
    assert result["best_epoch"] <= 5
    assert np.isfinite(result["best_val_ap"])
    assert np.isfinite(result["history"][-1]["train_loss"])
    assert len(result["validation_probabilities"]) == 6
    assert len(result["validation_labels"]) == 6
    assert result["device"] == "cpu"

    assert np.all(
        (result["validation_probabilities"] >= 0)
        & (result["validation_probabilities"] <= 1)
    )


def test_training_rejects_overlapping_masks():
    data = create_synthetic_graph()
    data.val_mask[0] = True

    with pytest.raises(ValueError, match="overlap"):
        train_gnn_model(
            data=data,
            model_name="gcn",
            max_epochs=2,
            device="cpu",
        )


def test_training_rejects_invalid_training_labels():
    data = create_synthetic_graph()
    data.y[0] = -1

    with pytest.raises(ValueError, match="invalid labels"):
        train_gnn_model(
            data=data,
            model_name="gcn",
            max_epochs=2,
            device="cpu",
        )


def test_training_rejects_validation_with_one_class():
    data = create_synthetic_graph()
    data.y[7:12] = 0

    with pytest.raises(ValueError, match="both classes"):
        train_gnn_model(
            data=data,
            model_name="gcn",
            max_epochs=2,
            device="cpu",
        )


def test_best_checkpoint_matches_returned_predictions():
    data = create_synthetic_graph()

    result = train_gnn_model(
        data=data,
        model_name="gcn",
        hidden_channels=8,
        dropout=0.2,
        learning_rate=0.01,
        max_epochs=5,
        patience=5,
        seed=42,
        device="cpu",
    )

    model = result["model"]
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = torch.softmax(logits[data.val_mask], dim=1)[:, 1]

    assert np.allclose(
        probabilities.numpy(),
        result["validation_probabilities"],
    )

    history_best_epoch = max(
        result["history"],
        key=lambda record: record["validation_ap"],
    )["epoch"]

    assert result["best_epoch"] == history_best_epoch
