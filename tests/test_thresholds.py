import numpy as np
import pandas as pd
import pytest

from fincrime.evaluation.thresholds import (
    evaluate_validation_thresholds,
)


@pytest.fixture
def sample_data():
    y_true = pd.Series([1, 0, 1, 0, 0, 1, 0, 0])
    y_scores = np.array([0.95, 0.80, 0.70, 0.60, 0.40, 0.30, 0.20, 0.10])

    return y_true, y_scores


def test_probability_threshold_metrics(sample_data):
    y_true, y_scores = sample_data

    result = evaluate_validation_thresholds(
        y_true,
        y_scores,
        model_name="xgboost",
        feature_set="Local",
        probability_thresholds=[0.5],
        review_budgets=[0.25],
    )

    row = result["probability_thresholds"].iloc[0]

    assert row["n_flagged"] == 4
    assert row["true_positives"] == 2
    assert row["false_positives"] == 2
    assert row["false_negatives"] == 1
    assert row["true_negatives"] == 3
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(2 / 3)


def test_review_budget_metrics(sample_data):
    y_true, y_scores = sample_data

    result = evaluate_validation_thresholds(
        y_true,
        y_scores,
        model_name="xgboost",
        feature_set="Local",
        probability_thresholds=[0.5],
        review_budgets=[0.25],
    )

    row = result["review_budgets"].iloc[0]

    assert row["n_to_review"] == 2
    assert row["true_positives"] == 1
    assert row["false_positives"] == 1
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(1 / 3)


def test_budget_rounds_up(sample_data):
    y_true, y_scores = sample_data

    result = evaluate_validation_thresholds(
        y_true,
        y_scores,
        model_name="xgboost",
        feature_set="Local",
        probability_thresholds=[0.5],
        review_budgets=[0.01],
    )

    assert result["review_budgets"].iloc[0]["n_to_review"] == 1


def test_invalid_probability_scores(sample_data):
    y_true, _ = sample_data

    with pytest.raises(ValueError):
        evaluate_validation_thresholds(
            y_true,
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.5]),
            model_name="xgboost",
            feature_set="Local",
        )


def test_invalid_labels(sample_data):
    _, y_scores = sample_data

    with pytest.raises(ValueError):
        evaluate_validation_thresholds(
            np.array([0, 1, 2, 0, 1, 0, 1, 0]),
            y_scores,
            model_name="xgboost",
            feature_set="Local",
        )


def test_mismatched_lengths(sample_data):
    y_true, y_scores = sample_data

    with pytest.raises(ValueError):
        evaluate_validation_thresholds(
            y_true,
            y_scores[:-1],
            model_name="xgboost",
            feature_set="Local",
        )