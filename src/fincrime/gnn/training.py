import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data

from fincrime.gnn.models import build_gnn_model


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_class_weights(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    train_labels = y[train_mask]

    if train_labels.numel() == 0:
        raise ValueError("No labelled training nodes were provided.")

    if not torch.all((train_labels == 0) | (train_labels == 1)):
        raise ValueError("Training labels must contain only 0 and 1.")

    class_counts = torch.bincount(train_labels, minlength=2)

    if torch.any(class_counts == 0):
        raise ValueError(
            "Both classes must be present in the training nodes."
        )

    total_samples = class_counts.sum()
    class_weights = total_samples / (2.0 * class_counts.float())

    return class_weights.to(device)


def train_gnn_model(
    data: Data,
    model_name: str,
    hidden_channels: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    max_epochs: int = 200,
    patience: int = 20,
    min_delta: float = 0.0,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> dict:
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")

    if patience <= 0:
        raise ValueError("patience must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    if min_delta < 0:
        raise ValueError("min_delta cannot be negative.")

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    required_attributes = [
        "x",
        "edge_index",
        "y",
        "train_mask",
        "val_mask",
    ]

    for attribute in required_attributes:
        if not hasattr(data, attribute):
            raise ValueError(f"Graph data is missing {attribute}.")

    if data.x.ndim != 2:
        raise ValueError("Node features must be a 2D tensor.")

    if data.y.ndim != 1:
        raise ValueError("Labels must be a 1D tensor.")

    num_nodes = data.x.shape[0]

    if data.y.shape[0] != num_nodes:
        raise ValueError("Features and labels have different node counts.")

    if data.train_mask.shape != (num_nodes,):
        raise ValueError("Training mask has an invalid shape.")

    if data.val_mask.shape != (num_nodes,):
        raise ValueError("Validation mask has an invalid shape.")

    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()

    if torch.any(train_mask & val_mask):
        raise ValueError("Training and validation masks overlap.")

    if not train_mask.any() or not val_mask.any():
        raise ValueError("Training and validation masks cannot be empty.")

    y = data.y.long()

    if not torch.all((y[train_mask] == 0) | (y[train_mask] == 1)):
        raise ValueError("Training nodes contain invalid labels.")

    if not torch.all((y[val_mask] == 0) | (y[val_mask] == 1)):
        raise ValueError("Validation nodes contain invalid labels.")

    if torch.unique(y[val_mask]).numel() != 2:
        raise ValueError(
            "Validation nodes must contain both classes for AP evaluation."
        )

    if not torch.isfinite(data.x).all():
        raise ValueError("Node features contain missing or infinite values.")

    set_random_seeds(seed)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    training_start = time.perf_counter()

    class_weights = calculate_class_weights(
        y=y,
        train_mask=train_mask,
        device=device,
    )

    model = build_gnn_model(
        model_name=model_name,
        in_channels=x.shape[1],
        hidden_channels=hidden_channels,
        dropout=dropout,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
    )

    best_val_ap = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x, edge_index)

        loss = criterion(
            logits[train_mask],
            y[train_mask],
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch}."
            )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_logits = model(x, edge_index)

            validation_probabilities = torch.softmax(
                validation_logits[val_mask],
                dim=1,
            )[:, 1]

            validation_labels = y[val_mask]

            val_ap = average_precision_score(
                validation_labels.cpu().numpy(),
                validation_probabilities.cpu().numpy(),
            )

        epoch_loss = loss.item()

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "validation_ap": val_ap,
        })

        if val_ap > best_val_ap + min_delta:
            best_val_ap = val_ap
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {epoch_loss:.5f} | "
                f"Validation AP: {val_ap:.5f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        best_logits = model(x, edge_index)

        best_validation_probabilities = torch.softmax(
            best_logits[val_mask],
            dim=1,
        )[:, 1]

    if device.type == "cuda":
        torch.cuda.synchronize(device)

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
    else:
        peak_gpu_memory_mb = None

    training_runtime_seconds = time.perf_counter() - training_start

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_ap": best_val_ap,
        "class_weights": class_weights.detach().cpu(),
        "validation_probabilities": (
            best_validation_probabilities.cpu().numpy()
        ),
        "validation_labels": y[val_mask].cpu().numpy(),
        "seed": seed,
        "device": str(device),
        "training_runtime_seconds": training_runtime_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }
