"""Validation-based probability threshold and investigation-budget evaluation."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_PROBABILITY_THRESHOLDS = (
    0.10,
    0.20,
    0.30,
    0.50,
    0.70,
    0.90,
    0.95,
    0.99,
)

DEFAULT_REVIEW_BUDGETS = (
    0.01,
    0.02,
    0.05,
    0.10,
)


def _calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate confusion-matrix counts and binary classification metrics."""

    true_positives = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_negatives = int(np.sum((y_true == 0) & (y_pred == 0)))

    n_flagged = true_positives + false_positives

    precision = (
        true_positives / n_flagged
        if n_flagged > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "n_flagged": n_flagged,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_validation_thresholds(
    y_true,
    y_scores,
    model_name: str,
    feature_set: str,
    probability_thresholds: Sequence[float] = (
        DEFAULT_PROBABILITY_THRESHOLDS
    ),
    review_budgets: Sequence[float] = DEFAULT_REVIEW_BUDGETS,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate probability thresholds and fixed review budgets.

    Parameters
    ----------
    y_true : array-like
        Binary validation labels: 1 = illicit, 0 = licit.
    y_scores : array-like
        Predicted probabilities for the illicit class.
    model_name : str
        Model name, e.g. xgboost or random_forest.
    feature_set : str
        Feature set name, e.g. Local or Local+Hand.
    probability_thresholds : sequence of float
        Probability cutoffs. A transaction is flagged when
        its score is greater than or equal to the cutoff.
    review_budgets : sequence of float
        Fractions of the validation population to review.

    Returns
    -------
    dict
        Two DataFrames:
        - probability_thresholds
        - review_budgets
    """

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    if y_true.ndim != 1 or y_scores.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(y_scores):
        raise ValueError(
            "Labels and scores must be non-empty and have equal lengths."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Validation labels must contain only 0 and 1.")

    if not np.isfinite(y_scores).all():
        raise ValueError("Predicted scores must be finite.")

    if ((y_scores < 0) | (y_scores > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("Validation labels must contain both classes.")

    if not model_name or not feature_set:
        raise ValueError("model_name and feature_set must be non-empty.")

    probability_thresholds = list(probability_thresholds)
    review_budgets = list(review_budgets)

    if not probability_thresholds:
        raise ValueError("At least one probability threshold is required.")

    if not review_budgets:
        raise ValueError("At least one review budget is required.")

    for threshold in probability_thresholds:
        if (
            isinstance(threshold, (bool, np.bool_))
            or not np.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                "Probability thresholds must be finite values between 0 and 1."
            )

    for budget in review_budgets:
        if (
            isinstance(budget, (bool, np.bool_))
            or not np.isfinite(budget)
            or not 0 < budget <= 1
        ):
            raise ValueError(
                "Review budgets must be finite values greater than 0 "
                "and at most 1."
            )

    base_metadata = {
        "model": model_name,
        "feature_set": feature_set,
        "n_validation": len(y_true),
        "validation_prevalence": float(y_true.mean()),
    }


    # 1. Probability threshold evaluation

    threshold_rows = []

    for threshold in probability_thresholds:

        predictions = (y_scores >= threshold).astype(int)

        metrics = _calculate_binary_metrics(
            y_true=y_true,
            y_pred=predictions,
        )

        threshold_rows.append({
            **base_metadata,
            "probability_threshold": float(threshold),
            "flagged_rate": metrics["n_flagged"] / len(y_true),
            **metrics,
        })

    threshold_df = pd.DataFrame(threshold_rows)


    # 2. Fixed investigation-budget evaluation

    budget_rows = []

    # Stable sorting makes tie handling deterministic.
    ranked_indices = np.argsort(
        -y_scores,
        kind="stable",
    )

    for budget in review_budgets:

        n_to_review = min(
            len(y_true),
            max(1, int(np.ceil(len(y_true) * budget))),
        )

        predictions = np.zeros(len(y_true), dtype=int)
        selected_indices = ranked_indices[:n_to_review]
        predictions[selected_indices] = 1

        metrics = _calculate_binary_metrics(
            y_true=y_true,
            y_pred=predictions,
        )

        budget_rows.append({
            **base_metadata,
            "review_budget": float(budget),
            "n_to_review": n_to_review,
            "cutoff_probability": float(
                y_scores[selected_indices[-1]]
            ),
            "review_rate": n_to_review / len(y_true),
            **metrics,
        })

    budget_df = pd.DataFrame(budget_rows)

    return {
        "probability_thresholds": threshold_df,
        "review_budgets": budget_df,
    }