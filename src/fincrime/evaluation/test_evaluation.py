"""Utilities for leakage-safe validation threshold selection and test evaluation."""

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_binary_inputs(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    if y_true.ndim != 1 or y_scores.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional.")

    if len(y_true) != len(y_scores):
        raise ValueError("Labels and scores must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Labels and scores cannot be empty.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only binary values 0 and 1.")

    if np.unique(y_true).size != 2:
        raise ValueError("Both classes must be present.")

    if not np.isfinite(y_scores).all():
        raise ValueError("Scores contain non-finite values.")

    if ((y_scores < 0) | (y_scores > 1)).any():
        raise ValueError("Scores must be probabilities between 0 and 1.")

    return y_true.astype(int), y_scores


def select_validation_threshold(y_true, y_scores):
    """
    Select the probability threshold that maximises validation F1.

    If multiple thresholds achieve the same F1, choose the highest one.
    This function must only receive validation labels and probabilities.
    """
    y_true, y_scores = _validate_binary_inputs(y_true, y_scores)

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_scores,
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to calculate a validation threshold.")

    denominator = precision[:-1] + recall[:-1]

    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_f1 = f1_values.max()
    best_indices = np.flatnonzero(
        np.isclose(f1_values, best_f1)
    )

    best_index = best_indices[-1]

    return {
        "threshold": float(thresholds[best_index]),
        "validation_f1": float(f1_values[best_index]),
    }


def evaluate_test_predictions(
    y_true,
    y_scores,
    threshold,
    review_budgets=(0.01, 0.02, 0.05, 0.10),
):
    """
    Evaluate test predictions using a threshold selected on validation data.

    Also calculates precision and recall at fixed review budgets.
    """
    y_true, y_scores = _validate_binary_inputs(y_true, y_scores)

    if not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError("threshold must be numeric.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    for budget in review_budgets:
        if not isinstance(budget, (int, float, np.integer, np.floating)):
            raise TypeError("Review budgets must be numeric.")

        if not np.isfinite(budget) or not 0 < budget <= 1:
            raise ValueError("Review budgets must be in the interval (0, 1].")

    predictions = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    results = {
        "n_test": int(len(y_true)),
        "test_prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "average_precision": float(
            average_precision_score(y_true, y_scores)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    review_results = []

    ranked_indices = np.argsort(-y_scores, kind="stable")

    for budget in review_budgets:
        k = max(1, int(np.ceil(len(y_true) * budget)))
        selected_indices = ranked_indices[:k]

        selected_labels = y_true[selected_indices]
        true_positives = int(selected_labels.sum())
        false_positives = int(k - true_positives)

        review_results.append({
            "review_budget": float(budget),
            "n_reviewed": int(k),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision_at_k": float(true_positives / k),
            "recall_at_k": float(true_positives / y_true.sum()),
            "cutoff_probability": float(y_scores[selected_indices[-1]]),
        })

    results["review_budgets"] = review_results

    return results