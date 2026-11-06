"""MLflow runner for Track A XGBoost evaluation."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow

from fincrime.baselines.data import prepare_tabular_data
from fincrime.baselines.training import train_and_evaluate
from fincrime.evaluation.test_evaluation import (
    evaluate_test_predictions,
    select_validation_threshold,
)


def run_track_a_xgboost(
    feature_sets: Mapping[str, Any],
    labels,
    timesteps,
    split_config: dict,
    selected_config_path: str | Path,
    split_config_path: str | Path,
    experiment_name: str = "elliptic-track-a-xgboost",
    tracking_uri: str = "http://127.0.0.1:5000",
    random_seeds: tuple[int, ...] = (42, 123, 2024, 7, 21),
) -> dict:
    """
    Evaluate previously selected XGBoost configurations on Track A.

    Hyperparameters are loaded from the tuning-stage JSON.
    No additional hyperparameter tuning is performed.

    Track A:
        Fit:        timesteps 1-30
        Validation: timesteps 31-34
        Test:       timesteps 35-49

    Classification thresholds are selected using validation data only.
    Test metrics are calculated using those frozen thresholds.
    """

    expected_feature_sets = {"Local", "All", "Local+Hand"}

    if not isinstance(feature_sets, Mapping):
        raise TypeError("feature_sets must be a mapping.")

    if set(feature_sets) != expected_feature_sets:
        raise ValueError(
            "Expected exactly three feature sets: "
            "Local, All and Local+Hand."
        )

    if not random_seeds:
        raise ValueError("At least one random seed is required.")

    if len(set(random_seeds)) != len(random_seeds):
        raise ValueError("random_seeds must contain unique values.")

    selected_config_path = Path(selected_config_path)
    split_config_path = Path(split_config_path)

    if not selected_config_path.is_file():
        raise FileNotFoundError(
            f"Selected configurations not found: {selected_config_path}"
        )

    if not split_config_path.is_file():
        raise FileNotFoundError(
            f"Split configuration not found: {split_config_path}"
        )

    with selected_config_path.open("r", encoding="utf-8") as file:
        selected_configs = json.load(file)

    if set(selected_configs) != expected_feature_sets:
        raise ValueError(
            "Selected configuration file must contain exactly "
            "Local, All and Local+Hand."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    all_results = {}

    for feature_set_name, features in feature_sets.items():

        selected = selected_configs[feature_set_name]

        model_params = selected.get("model_params")
        scale_pos_weight = selected.get("scale_pos_weight")
        trial_name = selected.get("trial_name")

        if not isinstance(model_params, dict) or not model_params:
            raise ValueError(
                f"Invalid model parameters for {feature_set_name}."
            )

        if scale_pos_weight is None or not trial_name:
            raise ValueError(
                f"Missing selected configuration details for "
                f"{feature_set_name}."
            )

        print(f"\n{'=' * 65}")
        print(f"Track A | XGBoost | {feature_set_name}")
        print(f"Selected tuning trial: {trial_name}")
        print(f"{'=' * 65}")

        # Prepare Track A training, validation and test data.
        data = prepare_tabular_data(
            features=features,
            labels=labels,
            timesteps=timesteps,
            split_config=split_config,
            split_name="track_a",
        )

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_validation = data["X_validation"]
        y_validation = data["y_validation"]

        for random_state in random_seeds:

            run_name = (
                f"Track_A__XGBoost__{feature_set_name}__seed_{random_state}"
            )

            with mlflow.start_run(run_name=run_name):

                mlflow.set_tags({
                    "track": "A",
                    "model": "xgboost",
                    "feature_set": feature_set_name,
                    "split": "track_a",
                    "source_tuning_trial": trial_name,
                    "evaluation_stage": "validation_and_test",
                    "random_seed": str(random_state),
                })

                mlflow.log_params({
                    "source_tuning_run_id": selected["mlflow_run_id"],
                    "random_state": random_state,
                    "scale_pos_weight": float(scale_pos_weight),
                    "n_features": X_train.shape[1],
                    "fit_timestep_start": 1,
                    "fit_timestep_end": 30,
                    "validation_timestep_start": 31,
                    "validation_timestep_end": 34,
                    "test_timestep_start": 35,
                    "test_timestep_end": 49,
                    **{
                        f"model__{key}": value
                        for key, value in model_params.items()
                    },
                })

                result = train_and_evaluate(
                    model_name="xgboost",
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    random_state=random_state,
                    scale_pos_weight=float(scale_pos_weight),
                    model_params=model_params,
                )

                # Select the classification threshold using validation data only.
                threshold_result = select_validation_threshold(
                    y_true=y_validation.to_numpy(),
                    y_scores=result["validation_probabilities"],
                )

                # Generate test probabilities using the fitted model.
                test_probabilities = result["model"].predict_proba(
                    data["X_test"]
                )[:, 1]

                # Evaluate test predictions using the frozen validation threshold.
                test_metrics = evaluate_test_predictions(
                    y_true=data["y_test"].to_numpy(),
                    y_scores=test_probabilities,
                    threshold=threshold_result["threshold"],
                )

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
                    "validation_selected_threshold": (
                        threshold_result["threshold"]
                    ),
                    "validation_selected_f1": (
                        threshold_result["validation_f1"]
                    ),
                    "test_average_precision": test_metrics[
                        "average_precision"
                    ],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                    "test_threshold": test_metrics["threshold"],
                    "test_n": test_metrics["n_test"],
                    "test_prevalence": test_metrics["test_prevalence"],
                    "test_tp": test_metrics["tp"],
                    "test_fp": test_metrics["fp"],
                    "test_tn": test_metrics["tn"],
                    "test_fn": test_metrics["fn"],
                    **{
                        f"test_{int(item['review_budget'] * 100)}pct_{metric}": (
                            item[key]
                        )
                        for item in test_metrics["review_budgets"]
                        for key, metric in [
                            ("precision_at_k", "precision"),
                            ("recall_at_k", "recall"),
                            ("n_reviewed", "n_reviewed"),
                            ("true_positives", "true_positives"),
                            ("false_positives", "false_positives"),
                            ("cutoff_probability", "cutoff_probability"),
                        ]
                    },
                })

                mlflow.log_artifact(
                    str(selected_config_path),
                    artifact_path="config",
                )

                mlflow.log_artifact(
                    str(split_config_path),
                    artifact_path="config",
                )

                run_id = mlflow.active_run().info.run_id

                all_results[(feature_set_name, random_state)] = {
                    "model": result["model"],
                    "validation_probabilities": result[
                        "validation_probabilities"
                    ],
                    "validation_labels": y_validation.to_numpy(),
                    "average_precision": result["average_precision"],
                    "roc_auc": result["roc_auc"],
                    "n_train": result["n_train"],
                    "n_validation": result["n_validation"],
                    "validation_prevalence": result[
                        "validation_prevalence"
                    ],
                    "source_tuning_trial": trial_name,
                    "random_state": random_state,
                    "run_id": run_id,
                    "selected_threshold": threshold_result["threshold"],
                    "test_probabilities": test_probabilities,
                    "test_labels": data["y_test"].to_numpy(),
                    "test_metrics": test_metrics,
                }

                print(
                    f"Seed {random_state} | "
                    f"Validation AP: {result['average_precision']:.6f} | "
                    f"Test AP: {test_metrics['average_precision']:.6f} | "
                    f"Test ROC-AUC: {test_metrics['roc_auc']:.6f}"
                )
                print(
                    f"Validation-selected threshold: "
                    f"{threshold_result['threshold']:.6f}"
                )
                print(f"MLflow run ID: {run_id}")

    return all_results