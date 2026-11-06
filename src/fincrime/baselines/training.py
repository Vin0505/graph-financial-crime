"""Training and evaluation utilities for tabular baselines."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fincrime.baselines.models import build_tabular_model


def train_and_evaluate(
    model_name: str,
    X_train,
    y_train,
    X_validation,
    y_validation,
    random_state: int = 42,
    imbalance_strategy: str = "none",
    scale_pos_weight: float = 1.0,
    model_params: dict | None = None,
) -> dict:
    """Train a tabular classifier and evaluate validation ranking."""

    if X_train.empty or X_validation.empty:
        raise ValueError("Training and validation features must not be empty.")

    if not X_train.index.equals(y_train.index):
        raise ValueError("Training feature and label indices do not match.")

    if not X_validation.index.equals(y_validation.index):
        raise ValueError("Validation feature and label indices do not match.")

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError("Training and validation feature columns do not match.")

    if y_train.nunique() != 2 or y_validation.nunique() != 2:
        raise ValueError("Training and validation data must contain both classes.")

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training labels must be binary: 0 and 1.")

    if set(y_validation.unique()) != {0, 1}:
        raise ValueError("Validation labels must be binary: 0 and 1.")

    model = build_tabular_model(
        model_name=model_name,
        random_state=random_state,
        imbalance_strategy=imbalance_strategy,
        scale_pos_weight=scale_pos_weight,
        model_params=model_params,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_validation)

    # Identify the probability column corresponding to the illicit class (1).
    positive_class_indices = np.flatnonzero(model.classes_ == 1)

    if len(positive_class_indices) != 1:
        raise ValueError("Could not identify the positive class in model output.")

    illicit_probabilities = probabilities[:, positive_class_indices[0]]

    ap = average_precision_score(y_validation, illicit_probabilities)
    roc_auc = roc_auc_score(y_validation, illicit_probabilities)

    return {
        "model": model,
        "validation_probabilities": illicit_probabilities,
        "average_precision": ap,
        "roc_auc": roc_auc,
        "n_train": len(X_train),
        "n_validation": len(X_validation),
        "validation_prevalence": float(y_validation.mean()),
    }