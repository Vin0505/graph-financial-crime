"""MLflow-integrated runner for XGBoost tuning experiments."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow

from fincrime.baselines.data import prepare_tabular_data
from fincrime.baselines.training import train_and_evaluate
from fincrime.baselines.tuning import (
    load_xgboost_tuning_config,
    resolve_xgboost_trial,
)


def run_xgboost_tuning(
    feature_sets: Mapping[str, Any],
    labels,
    timesteps,
    split_config: dict,
    config_path: str | Path,
    experiment_name: str = "elliptic-xgboost-tuning",
    tracking_uri: str = "http://127.0.0.1:5000",
) -> dict:
    """
    Run the six configured XGBoost trials for each feature set.

    Each trial is logged as an independent MLflow run. The best
    model for each feature set is selected using validation AP.

    Parameters
    ----------
    feature_sets : Mapping[str, DataFrame]
        Named feature sets, e.g. Local, All and Local+Hand.
    labels : Series
        Original Elliptic labels: 1, 2 or "unknown".
    timesteps : Series
        Timestep for each transaction.
    split_config : dict
        Loaded split configuration.
    config_path : str or Path
        Path to configs/xgboost.yaml.
    experiment_name : str
        MLflow experiment name.
    tracking_uri : str
        MLflow tracking server URI.

    Returns
    -------
    dict
        Trial results and best model for each feature set.
    """

    if not isinstance(feature_sets, Mapping) or not feature_sets:
        raise ValueError("feature_sets must be a non-empty mapping.")

    if any(not isinstance(name, str) or not name.strip() for name in feature_sets):
        raise ValueError("Every feature set must have a non-empty string name.")

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"XGBoost tuning configuration not found: {config_path}"
        )

    tuning_config = load_xgboost_tuning_config(config_path)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    all_results = {}

    for feature_set_name, features in feature_sets.items():

        print(f"\n{'=' * 70}")
        print(f"Feature set: {feature_set_name}")
        print(f"{'=' * 70}")

        # Prepare the same tuning split for this feature set.
        data = prepare_tabular_data(
            features=features,
            labels=labels,
            timesteps=timesteps,
            split_config=split_config,
            split_name="tuning",
        )

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_validation = data["X_validation"]
        y_validation = data["y_validation"]

        feature_set_results = []

        for trial in tuning_config["tuning"]["trials"]:

            resolved = resolve_xgboost_trial(
                config=tuning_config,
                trial=trial,
                y_train=y_train,
            )

            trial_name = resolved["trial_name"]

            print(f"\nRunning {feature_set_name} | {trial_name}")

            with mlflow.start_run(
                run_name=f"{feature_set_name}__{trial_name}"
            ):

                # Record experiment identity and reproducibility details.
                mlflow.set_tags({
                    "model": "xgboost",
                    "feature_set": feature_set_name,
                    "trial_name": trial_name,
                    "selection_metric": "average_precision",
                    "split": "tuning",
                })

                mlflow.log_params({
                    "model_name": resolved["model_name"],
                    "feature_set": feature_set_name,
                    "trial_name": trial_name,
                    "random_state": resolved["random_state"],
                    "scale_pos_weight": resolved["scale_pos_weight"],
                    "positive_training_count": resolved[
                        "positive_training_count"
                    ],
                    "negative_training_count": resolved[
                        "negative_training_count"
                    ],
                    "n_features": X_train.shape[1],
                    **{
                        f"model__{key}": value
                        for key, value in resolved["model_params"].items()
                    },
                })

                # Train and evaluate using the existing tested utility.
                result = train_and_evaluate(
                    model_name=resolved["model_name"],
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    random_state=resolved["random_state"],
                    scale_pos_weight=resolved["scale_pos_weight"],
                    model_params=resolved["model_params"],
                )

                # Record validation metrics and dataset information.
                mlflow.log_metrics({
                    "validation_average_precision": result[
                        "average_precision"
                    ],
                    "validation_roc_auc": result["roc_auc"],
                    "n_train": result["n_train"],
                    "n_validation": result["n_validation"],
                    "validation_prevalence": result[
                        "validation_prevalence"
                    ],
                })

                # Keep the exact configuration used for this run.
                mlflow.log_artifact(
                    str(config_path),
                    artifact_path="config",
                )

                run_id = mlflow.active_run().info.run_id

                trial_result = {
                    "trial_name": trial_name,
                    "run_id": run_id,
                    "average_precision": result["average_precision"],
                    "roc_auc": result["roc_auc"],
                    "model": result["model"],
                    "validation_probabilities": result[
                        "validation_probabilities"
                    ],
                    "resolved_parameters": resolved,
                }

                feature_set_results.append(trial_result)

                print(
                    f"  Validation AP: "
                    f"{result['average_precision']:.6f}"
                )
                print(
                    f"  Validation ROC-AUC: "
                    f"{result['roc_auc']:.6f}"
                )

        # Select the best trial using validation average precision.
        best_trial = max(
            feature_set_results,
            key=lambda item: item["average_precision"],
        )

        all_results[feature_set_name] = {
            "trials": feature_set_results,
            "best_trial": best_trial,
            "best_model": best_trial["model"],
            "best_run_id": best_trial["run_id"],
        }

        print(f"\nBest trial for {feature_set_name}:")
        print(f"  Trial: {best_trial['trial_name']}")
        print(f"  Validation AP: {best_trial['average_precision']:.6f}")
        print(f"  MLflow run ID: {best_trial['run_id']}")

    return all_results
