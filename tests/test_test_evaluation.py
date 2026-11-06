import numpy as np
import pytest

from fincrime.evaluation.test_evaluation import (
    evaluate_test_predictions,
    select_validation_threshold,
)


def test_select_validation_threshold_returns_valid_result():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.3, 0.7, 0.9])

    result = select_validation_threshold(y_true, y_scores)

    assert 0 <= result["threshold"] <= 1
    assert 0 <= result["validation_f1"] <= 1


def test_select_validation_threshold_uses_validation_data():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = select_validation_threshold(y_true, y_scores)

    assert result["validation_f1"] == pytest.approx(1.0)


def test_evaluate_test_predictions_returns_expected_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = evaluate_test_predictions(
        y_true,
        y_scores,
        threshold=0.5,
        review_budgets=(0.5,),
    )

    assert result["n_test"] == 4
    assert result["tp"] == 2
    assert result["tn"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_review_budget_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.7, 0.9])

    result = evaluate_test_predictions(
        y_true,
        y_scores,
        threshold=0.5,
        review_budgets=(0.5,),
    )

    budget_result = result["review_budgets"][0]

    assert budget_result["n_reviewed"] == 2
    assert budget_result["true_positives"] == 2
    assert budget_result["false_positives"] == 0
    assert budget_result["precision_at_k"] == pytest.approx(1.0)
    assert budget_result["recall_at_k"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "y_true, y_scores",
    [
        ([0, 0], [0.1, 0.2]),
        ([0, 1], [0.1]),
        ([0, 2], [0.1, 0.9]),
        ([0, 1], [0.1, np.nan]),
        ([0, 1], [0.1, 1.2]),
    ],
)
def test_invalid_prediction_inputs_raise_error(y_true, y_scores):
    with pytest.raises(ValueError):
        select_validation_threshold(y_true, y_scores)


def test_invalid_threshold_raises_error():
    with pytest.raises(ValueError):
        evaluate_test_predictions(
            [0, 1],
            [0.1, 0.9],
            threshold=1.5,
        )


def test_invalid_review_budget_raises_error():
    with pytest.raises(ValueError):
        evaluate_test_predictions(
            [0, 1],
            [0.1, 0.9],
            threshold=0.5,
            review_budgets=(0,),
        )