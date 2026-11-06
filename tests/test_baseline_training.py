import numpy as np
import pandas as pd
import pytest

from fincrime.baselines.training import train_and_evaluate


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)

    X_train = pd.DataFrame(
        rng.normal(size=(100, 4)),
        columns=["f1", "f2", "f3", "f4"],
    )
    y_train = pd.Series(
        np.array([0, 1] * 50),
        index=X_train.index,
    )

    X_validation = pd.DataFrame(
        rng.normal(size=(40, 4)),
        columns=X_train.columns,
        index=range(100, 140),
    )
    y_validation = pd.Series(
        np.array([0, 1] * 20),
        index=X_validation.index,
    )

    return X_train, y_train, X_validation, y_validation


@pytest.mark.parametrize(
    "model_name",
    ["logistic_regression", "random_forest", "xgboost"],
)
def test_train_and_evaluate_returns_valid_results(
    model_name,
    training_data,
):
    X_train, y_train, X_validation, y_validation = training_data

    result = train_and_evaluate(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    assert 0 <= result["average_precision"] <= 1
    assert 0 <= result["roc_auc"] <= 1
    assert result["validation_probabilities"].shape == (40,)
    assert np.isfinite(result["validation_probabilities"]).all()
    assert result["n_train"] == 100
    assert result["n_validation"] == 40


def test_rejects_mismatched_feature_columns(training_data):
    X_train, y_train, X_validation, y_validation = training_data

    X_validation = X_validation.rename(columns={"f1": "different_feature"})

    with pytest.raises(ValueError, match="columns do not match"):
        train_and_evaluate(
            "logistic_regression",
            X_train,
            y_train,
            X_validation,
            y_validation,
        )


def test_rejects_mismatched_indices(training_data):
    X_train, y_train, X_validation, y_validation = training_data

    y_train.index = range(1, 101)

    with pytest.raises(ValueError, match="indices do not match"):
        train_and_evaluate(
            "logistic_regression",
            X_train,
            y_train,
            X_validation,
            y_validation,
        )