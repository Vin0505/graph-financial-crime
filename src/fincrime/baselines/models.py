"""Model factories for tabular baseline experiments."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


def build_tabular_model(
    model_name: str,
    random_state: int = 42,
    imbalance_strategy: str = "none",
    scale_pos_weight: float = 1.0,
    model_params: dict | None = None,
):
    """
    Build a tabular classification model.

    Parameters
    ----------
    model_name : str
        One of: logistic_regression, random_forest, xgboost.
    random_state : int
        Random seed for reproducibility.
    imbalance_strategy : str
        'none' or 'balanced'. Used by Logistic Regression
        and Random Forest.
    scale_pos_weight : float
        Positive-class weight for XGBoost.
    model_params : dict, optional
        Model-specific hyperparameters. Overrides defaults.
    """

    if imbalance_strategy not in {"none", "balanced"}:
        raise ValueError(
            "imbalance_strategy must be 'none' or 'balanced'."
        )

    params = model_params.copy() if model_params else {}

    class_weight = (
        "balanced" if imbalance_strategy == "balanced" else None
    )

    if model_name == "logistic_regression":

        default_params = {
            "C": 1.0,
            "max_iter": 2000,
            "solver": "lbfgs",
            "random_state": random_state,
        }
        default_params.update(params)

        classifier = LogisticRegression(
            class_weight=class_weight,
            **default_params,
        )

        # Scaling is fitted only on the training data because
        # it is contained inside the Pipeline.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

    elif model_name == "random_forest":

        default_params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": random_state,
        }
        default_params.update(params)

        return RandomForestClassifier(
            class_weight=class_weight,
            **default_params,
        )

    elif model_name == "xgboost":

        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": random_state,
        }
        default_params.update(params)

        return XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            **default_params,
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            "Choose logistic_regression, random_forest, or xgboost."
        )