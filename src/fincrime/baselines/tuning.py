from collections.abc import Mapping
from pathlib import Path
from typing import Any
import math

import numpy as np
import yaml


def load_xgboost_tuning_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """
    Load and validate the XGBoost tuning configuration.

    The configuration must define exactly six uniquely named trials.
    """
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"XGBoost configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, Mapping):
        raise ValueError("XGBoost configuration must be a YAML mapping.")

    model_config = config.get("model")
    tuning_config = config.get("tuning")

    if not isinstance(model_config, Mapping):
        raise ValueError("Configuration must contain a model mapping.")

    if not isinstance(tuning_config, Mapping):
        raise ValueError("Configuration must contain a tuning mapping.")

    if model_config.get("name") != "xgboost":
        raise ValueError("Expected model.name to be 'xgboost'.")

    if tuning_config.get("selection_metric") != "average_precision":
        raise ValueError(
            "The tuning selection metric must be 'average_precision'."
        )

    trials = tuning_config.get("trials")

    if not isinstance(trials, list) or len(trials) != 6:
        raise ValueError("Exactly six XGBoost trials must be configured.")

    trial_names = []

    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping):
            raise ValueError(f"Trial at index {index} must be a mapping.")

        name = trial.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Trial at index {index} must have a non-empty name."
            )

        trial_names.append(name)

        if "scale_pos_weight" not in trial:
            raise ValueError(
                f"Trial '{name}' is missing scale_pos_weight."
            )

        weight = trial["scale_pos_weight"]

        if isinstance(weight, str):
            if weight != "train_ratio":
                raise ValueError(
                    f"Trial '{name}' has unsupported scale_pos_weight: "
                    f"{weight}"
                )
        elif (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(
                f"Trial '{name}' must have a positive numeric "
                "scale_pos_weight or 'train_ratio'."
            )

    if len(set(trial_names)) != len(trial_names):
        raise ValueError("XGBoost trial names must be unique.")

    return {
        "model": dict(model_config),
        "tuning": {
            "selection_metric": tuning_config["selection_metric"],
            "trials": [dict(trial) for trial in trials],
        },
    }


def resolve_xgboost_trial(
    config: Mapping[str, Any],
    trial: Mapping[str, Any],
    y_train: Any,
) -> dict[str, Any]:
    """
    Resolve one trial's parameters.

    If scale_pos_weight is 'train_ratio', calculate:

        number of negative training samples
        ------------------------------------
        number of positive training samples

    Only the supplied training labels are used.

    Returns the model parameters, random state and class weight separately.
    """
    if not isinstance(config, Mapping) or not isinstance(trial, Mapping):
        raise TypeError("config and trial must be mappings.")

    model_config = config.get("model")

    if not isinstance(model_config, Mapping):
        raise ValueError("Configuration must contain a model mapping.")

    if model_config.get("name") != "xgboost":
        raise ValueError("Expected an XGBoost configuration.")

    if not trial.get("name"):
        raise ValueError("Trial must have a non-empty name.")

    random_state = model_config.get("random_state", 42)

    if isinstance(random_state, bool) or not isinstance(
        random_state, int
    ):
        raise ValueError("model.random_state must be an integer.")

    # Validate and convert training labels.
    try:
        y = np.asarray(y_train)
    except (TypeError, ValueError) as exc:
        raise ValueError("Could not convert y_train to an array.") from exc

    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_train must be a non-empty one-dimensional array.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_train must contain only binary labels 0 and 1.")

    positive_count = int(np.count_nonzero(y == 1))
    negative_count = int(np.count_nonzero(y == 0))

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "y_train must contain both positive and negative classes."
        )

    # Resolve class weighting using training labels only.
    weight = trial.get("scale_pos_weight")

    if weight == "train_ratio":
        scale_pos_weight = negative_count / positive_count
    elif (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError(
            "scale_pos_weight must be a positive number or 'train_ratio'."
        )
    else:
        scale_pos_weight = float(weight)

    # Global XGBoost settings inherited by every trial.
    model_params = {
        key: value
        for key, value in model_config.items()
        if key not in {"name", "random_state"}
    }

    # Trial-specific parameters override model-level defaults if present.
    trial_params = {
        key: value
        for key, value in trial.items()
        if key not in {"name", "scale_pos_weight"}
    }

    model_params.update(trial_params)

    return {
        "model_name": "xgboost",
        "trial_name": trial["name"],
        "random_state": random_state,
        "model_params": model_params,
        "scale_pos_weight": float(scale_pos_weight),
        "positive_training_count": positive_count,
        "negative_training_count": negative_count,
    }
