"""Data preparation utilities for tabular baseline experiments."""

import pandas as pd

from fincrime.evaluation.splits import build_split_masks
from fincrime.features.feature_validation import validate_model_features


def prepare_tabular_data(
    features: pd.DataFrame,
    labels: pd.Series,
    timesteps: pd.Series,
    split_config: dict,
    split_name: str = "tuning",
) -> dict:
    """
    Prepare leakage-safe tabular datasets for a temporal split.

    Unknown-labelled transactions are excluded from supervised
    training and labelled evaluation.

    Parameters
    ----------
    features : pd.DataFrame
        Model feature matrix. Timestep must not be included.
    labels : pd.Series
        Original transaction labels: 1, 2, or "unknown".
    timesteps : pd.Series
        Timestep associated with each transaction.
    split_config : dict
        Loaded split configuration.
    split_name : str
        Split to construct, e.g. "tuning" or "track_a".

    Returns
    -------
    dict
        Training and validation datasets, with test datasets
        included when the requested split defines a test period.
        Illicit = 1, licit = 0.
    """

    # 1. Validate the feature matrix.
    validate_model_features(features)

    # 2. Validate input types.
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series.")

    if not isinstance(timesteps, pd.Series):
        raise TypeError("timesteps must be a pandas Series.")

    # 3. Validate index alignment.
    if not features.index.equals(labels.index):
        raise ValueError("Feature and label indices do not match.")

    if not features.index.equals(timesteps.index):
        raise ValueError("Feature and timestep indices do not match.")

    if not features.index.is_unique:
        raise ValueError("Feature index contains duplicate transaction IDs.")

    # 4. Validate missing values.
    if features.isna().any().any():
        raise ValueError("Feature matrix contains missing values.")

    if labels.isna().any():
        raise ValueError("Labels contain missing values.")

    if timesteps.isna().any():
        raise ValueError("Timesteps contain missing values.")

    # 5. Map original labels to binary labels.
    label_config = split_config["labels"]

    illicit_label = label_config["illicit"]
    licit_label = label_config["licit"]
    unknown_label = label_config["unknown"]

    label_mapping = {
        illicit_label: 1,
        licit_label: 0,
    }

    original_labels = labels.astype(str)

    allowed_labels = {
        illicit_label,
        licit_label,
        unknown_label,
    }

    unexpected_labels = set(original_labels.unique()) - allowed_labels

    if unexpected_labels:
        raise ValueError(
            f"Unexpected labels found: {unexpected_labels}"
        )

    binary_labels = original_labels.map(label_mapping)

    # 6. Construct temporal split masks.
    split_masks = build_split_masks(
        timesteps,
        split_config,
        split_name,
    )

    fit_mask = split_masks["fit"]
    validation_mask = split_masks["validation"]

    # 7. Exclude unknown labels from supervised datasets.
    known_label_mask = binary_labels.notna()

    train_mask = fit_mask & known_label_mask
    validation_eval_mask = validation_mask & known_label_mask

    # 8. Construct training and validation datasets.
    X_train = features.loc[train_mask].copy()
    y_train = binary_labels.loc[train_mask].astype("int8")

    X_validation = features.loc[validation_eval_mask].copy()
    y_validation = binary_labels.loc[validation_eval_mask].astype("int8")

    # 9. Final training and validation sanity checks.
    if X_train.empty or X_validation.empty:
        raise ValueError("Training or validation dataset is empty.")

    if y_train.nunique() != 2:
        raise ValueError("Training data must contain both classes.")

    if y_validation.nunique() != 2:
        raise ValueError("Validation data must contain both classes.")

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "train_ids": X_train.index,
        "validation_ids": X_validation.index,
    }

    # 10. Include test data only when the split defines a test period.
    if split_masks["test"] is not None:
        test_eval_mask = split_masks["test"] & known_label_mask

        X_test = features.loc[test_eval_mask].copy()
        y_test = binary_labels.loc[test_eval_mask].astype("int8")
        test_ids = X_test.index

        if X_test.empty:
            raise ValueError("Test set contains no labelled transactions.")

        if y_test.nunique() != 2:
            raise ValueError("Test set must contain both classes for evaluation.")

        result.update({
            "X_test": X_test,
            "y_test": y_test,
            "test_ids": test_ids,
        })

    return result
