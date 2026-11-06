from collections.abc import Mapping
from pathlib import Path
from typing import Any
import math

import yaml


ALLOWED_HIDDEN_CHANNELS = {32, 64}
ALLOWED_DROPOUTS = {0.3, 0.4, 0.5}


def load_gnn_tuning_config(
    config_path: str | Path,
    expected_model: str | None = None,
) -> dict[str, Any]:
    """
    Load and validate a GCN or GraphSAGE tuning configuration.

    Each configuration must define exactly six uniquely named trials.
    The trials cover hidden dimensions 32/64 and dropout 0.3/0.4/0.5.
    """
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"GNN configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, Mapping):
        raise ValueError("GNN configuration must be a YAML mapping.")

    model_config = config.get("model")
    training_config = config.get("training")
    trials = config.get("trials")

    if not isinstance(model_config, Mapping):
        raise ValueError("Configuration must contain a model mapping.")

    if not isinstance(training_config, Mapping):
        raise ValueError("Configuration must contain a training mapping.")

    model_name = model_config.get("name")

    if model_name not in {"gcn", "graphsage"}:
        raise ValueError("model.name must be 'gcn' or 'graphsage'.")

    if expected_model is not None and model_name != expected_model:
        raise ValueError(
            f"Expected model.name to be '{expected_model}', "
            f"received '{model_name}'."
        )

    if not isinstance(trials, list) or len(trials) != 6:
        raise ValueError("Exactly six GNN trials must be configured.")

    selection_metric = config.get("tuning", {}).get(
        "selection_metric",
        "average_precision",
    )

    if selection_metric != "average_precision":
        raise ValueError(
            "The tuning selection metric must be 'average_precision'."
        )

    trial_names = []
    trial_combinations = set()

    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping):
            raise ValueError(f"Trial at index {index} must be a mapping.")

        name = trial.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Trial at index {index} must have a non-empty name."
            )

        trial_names.append(name)

        hidden_channels = trial.get("hidden_channels")
        dropout = trial.get("dropout")

        if (
            isinstance(hidden_channels, bool)
            or not isinstance(hidden_channels, int)
            or hidden_channels not in ALLOWED_HIDDEN_CHANNELS
        ):
            raise ValueError(
                f"Trial '{name}' must use hidden_channels 32 or 64."
            )

        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or float(dropout) not in ALLOWED_DROPOUTS
        ):
            raise ValueError(
                f"Trial '{name}' must use dropout 0.3, 0.4 or 0.5."
            )

        trial_combinations.add((hidden_channels, float(dropout)))

    if len(set(trial_names)) != 6:
        raise ValueError("GNN trial names must be unique.")

    expected_combinations = {
        (hidden_channels, dropout)
        for hidden_channels in ALLOWED_HIDDEN_CHANNELS
        for dropout in ALLOWED_DROPOUTS
    }

    if trial_combinations != expected_combinations:
        raise ValueError(
            "The six trials must cover every combination of "
            "hidden_channels {32, 64} and dropout {0.3, 0.4, 0.5} "
            "exactly once."
        )

    required_training_parameters = {
        "optimizer",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "patience",
        "min_delta",
        "class_weighting",
    }

    missing_parameters = (
        required_training_parameters - set(training_config)
    )

    if missing_parameters:
        raise ValueError(
            "Training configuration is missing: "
            f"{sorted(missing_parameters)}"
        )

    if training_config["optimizer"] != "adam":
        raise ValueError("Only the Adam optimizer is currently supported.")

    if training_config["class_weighting"] != "balanced":
        raise ValueError(
            "Only balanced class weighting is currently supported."
        )

    learning_rate = training_config["learning_rate"]
    weight_decay = training_config["weight_decay"]
    max_epochs = training_config["max_epochs"]
    patience = training_config["patience"]
    min_delta = training_config["min_delta"]

    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be a positive finite number.")

    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be a non-negative finite number.")

    if (
        isinstance(max_epochs, bool)
        or not isinstance(max_epochs, int)
        or max_epochs < 1
    ):
        raise ValueError("max_epochs must be a positive integer.")

    if (
        isinstance(patience, bool)
        or not isinstance(patience, int)
        or patience < 1
    ):
        raise ValueError("patience must be a positive integer.")

    if (
        isinstance(min_delta, bool)
        or not isinstance(min_delta, (int, float))
        or not math.isfinite(min_delta)
        or min_delta < 0
    ):
        raise ValueError("min_delta must be a non-negative finite number.")

    return {
        "model": dict(model_config),
        "training": dict(training_config),
        "tuning": {
            "selection_metric": selection_metric,
            "trials": [dict(trial) for trial in trials],
        },
    }


def resolve_gnn_trial(
    config: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Combine global training settings and trial-specific parameters.
    """
    if not isinstance(config, Mapping) or not isinstance(trial, Mapping):
        raise TypeError("config and trial must be mappings.")

    model_config = config.get("model")
    training_config = config.get("training")

    if not isinstance(model_config, Mapping):
        raise ValueError("Configuration must contain a model mapping.")

    if not isinstance(training_config, Mapping):
        raise ValueError("Configuration must contain a training mapping.")

    model_name = model_config.get("name")

    if model_name not in {"gcn", "graphsage"}:
        raise ValueError("Expected a GCN or GraphSAGE configuration.")

    if not trial.get("name"):
        raise ValueError("Trial must have a non-empty name.")

    return {
        "model_name": model_name,
        "trial_name": trial["name"],
        "hidden_channels": trial["hidden_channels"],
        "dropout": float(trial["dropout"]),
        "optimizer": training_config["optimizer"],
        "learning_rate": float(training_config["learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
        "max_epochs": training_config["max_epochs"],
        "patience": training_config["patience"],
        "min_delta": float(training_config["min_delta"]),
        "class_weighting": training_config["class_weighting"],
    }