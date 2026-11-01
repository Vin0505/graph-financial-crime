from pathlib import Path

import pandas as pd
import yaml


def load_split_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def _get_bounds(section) -> tuple[int, int]:
    if isinstance(section, dict):
        start = section["start"]
        end = section["end"]
    elif isinstance(section, (list, tuple)) and len(section) == 2:
        start, end = section
    else:
        raise ValueError("Invalid timestep range configuration.")

    if start > end:
        raise ValueError("Timestep range start cannot exceed end.")

    return start, end


def _create_step_mask(
    timesteps: pd.Series,
    bounds: tuple[int, int],
) -> pd.Series:
    start, end = bounds
    return timesteps.between(start, end, inclusive="both")


def build_split_masks(
    timesteps: pd.Series,
    split_config: dict,
    protocol: str,
    track_b_name: str | None = None,
) -> dict:
    if protocol == "tuning":
        split = split_config["tuning"]
        test_bounds = None

    elif protocol == "track_a":
        split = split_config["track_a"]
        test_bounds = _get_bounds(split["test"])

    elif protocol == "track_b":
        if track_b_name is None:
            raise ValueError("A Track B cutoff name must be provided.")

        matching_splits = [
            item for item in split_config["track_b"]
            if item["name"] == track_b_name
        ]

        if len(matching_splits) != 1:
            raise ValueError(f"Unknown or duplicate Track B split: {track_b_name}")

        split = matching_splits[0]
        test_bounds = _get_bounds(split["test"])

    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    fit_mask = _create_step_mask(
        timesteps,
        _get_bounds(split["fit"]),
    )

    validation_mask = _create_step_mask(
        timesteps,
        _get_bounds(split["validation"]),
    )

    test_mask = (
        _create_step_mask(timesteps, test_bounds)
        if test_bounds is not None
        else None
    )

    if (fit_mask & validation_mask).any():
        raise ValueError("Fit and validation periods overlap.")

    if test_mask is not None:
        if (fit_mask & test_mask).any() or (validation_mask & test_mask).any():
            raise ValueError("Test period overlaps with fit or validation.")

    return {
        "fit": fit_mask,
        "validation": validation_mask,
        "test": test_mask,
    }

def build_label_masks(
    labels: pd.Series,
    split_masks: dict,
    label_config: dict,
) -> dict:
    expected_labels = {
        label_config["illicit"],
        label_config["licit"],
        label_config["unknown"],
    }

    if labels.isna().any():
        raise ValueError("Labels contain missing values.")

    unexpected_labels = set(labels.unique()) - expected_labels

    if unexpected_labels:
        raise ValueError(f"Unexpected labels found: {unexpected_labels}")

    known_label_mask = labels.isin(
        [label_config["illicit"], label_config["licit"]]
    )

    for mask in split_masks.values():
        if mask is not None and not labels.index.equals(mask.index):
            raise ValueError("Labels and split masks must have identical indices.")

    train_loss_mask = split_masks["fit"] & known_label_mask

    validation_eval_mask = (
        split_masks["validation"] & known_label_mask
    )

    test_eval_mask = (
        split_masks["test"] & known_label_mask
        if split_masks["test"] is not None
        else None
    )

    return {
        "train_loss": train_loss_mask,
        "validation_eval": validation_eval_mask,
        "test_eval": test_eval_mask,
    }