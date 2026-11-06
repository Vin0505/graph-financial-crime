import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fincrime.baselines.models import build_tabular_model


def test_logistic_regression_factory():
    model = build_tabular_model("logistic_regression")

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(
        model.named_steps["classifier"],
        LogisticRegression,
    )


def test_random_forest_factory():
    model = build_tabular_model("random_forest")

    assert isinstance(model, RandomForestClassifier)


def test_xgboost_factory():
    model = build_tabular_model("xgboost")

    assert isinstance(model, XGBClassifier)


def test_balanced_class_weight():
    lr = build_tabular_model(
        "logistic_regression",
        imbalance_strategy="balanced",
    )

    rf = build_tabular_model(
        "random_forest",
        imbalance_strategy="balanced",
    )

    assert lr.named_steps["classifier"].class_weight == "balanced"
    assert rf.class_weight == "balanced"


def test_xgboost_scale_pos_weight():
    model = build_tabular_model(
        "xgboost",
        scale_pos_weight=9.64,
    )

    assert model.get_params()["scale_pos_weight"] == 9.64


def test_custom_hyperparameters():
    model = build_tabular_model(
        "random_forest",
        model_params={
            "n_estimators": 100,
            "max_depth": 8,
        },
    )

    assert model.n_estimators == 100
    assert model.max_depth == 8


def test_invalid_model_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_tabular_model("invalid_model")


def test_invalid_imbalance_strategy():
    with pytest.raises(ValueError, match="imbalance_strategy"):
        build_tabular_model(
            "random_forest",
            imbalance_strategy="oversampling",
        )