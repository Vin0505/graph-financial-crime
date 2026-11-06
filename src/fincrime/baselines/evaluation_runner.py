
"""MLflow runner for Track A Logistic Regression and Random Forest."""

from collections.abc import Mapping
from pathlib import Path
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd

from fincrime.baselines.data import prepare_tabular_data
from fincrime.baselines.training import train_and_evaluate
from fincrime.evaluation.test_evaluation import (
    evaluate_test_predictions,
    select_validation_threshold,
)


def run_track_a_tabular_baselines(
    feature_sets: Mapping,
    labels,
    timesteps,
    split_config: dict,
    split_config_path: str | Path,
    experiment_name: str = "elliptic-track-a-tabular-baselines",
    tracking_uri: str = "http://127.0.0.1:5000",
    random_seeds: tuple[int, ...] = (42, 123, 2024, 7, 21),
    model_names: tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
    ),
) -> dict:

    expected_feature_sets = {"Local", "All"}

    if not isinstance(feature_sets, Mapping):
        raise TypeError("feature_sets must be a mapping.")

    if set(feature_sets) != expected_feature_sets:
        raise ValueError("Expected exactly Local and All feature sets.")

    if not random_seeds or len(set(random_seeds)) != len(random_seeds):
        raise ValueError("random_seeds must be non-empty and unique.")

    allowed_models = {"logistic_regression", "random_forest"}

    if not model_names or not set(model_names).issubset(allowed_models):
        raise ValueError("Unsupported model_names.")

    split_config_path = Path(split_config_path)

    if not split_config_path.is_file():
        raise FileNotFoundError(split_config_path)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    all_results = {}

    for feature_set_name, features in feature_sets.items():

        data = prepare_tabular_data(
            features=features,
            labels=labels,
            timesteps=timesteps,
            split_config=split_config,
            split_name="track_a",
        )

        required = {
            "X_train", "y_train",
            "X_validation", "y_validation",
            "X_test", "y_test",
        }

        missing = required - set(data)

        if missing:
            raise KeyError(
                f"Test data missing from prepared data: {missing}"
            )

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_validation = data["X_validation"]
        y_validation = data["y_validation"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        for model_name in model_names:

            seeds = (
                (42,)
                if model_name == "logistic_regression"
                else random_seeds
            )

            for seed in seeds:

                run_name = (
                    f"Track_A__{feature_set_name}__"
                    f"{model_name}__test_evaluation__seed_{seed}"
                )

                print(f"\nRunning {run_name}")

                with mlflow.start_run(run_name=run_name):

                    mlflow.set_tags({
                        "track": "A",
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "split": "track_a",
                        "evaluation_stage": "validation_and_test",
                        "random_seed": str(seed),
                    })

                    mlflow.log_params({
                        "random_state": seed,
                        "imbalance_strategy": "none",
                        "n_features": X_train.shape[1],
                        "fit_timestep_start": 1,
                        "fit_timestep_end": 30,
                        "validation_timestep_start": 31,
                        "validation_timestep_end": 34,
                        "test_timestep_start": 35,
                        "test_timestep_end": 49,
                    })

                    result = train_and_evaluate(
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_validation=X_validation,
                        y_validation=y_validation,
                        random_state=seed,
                        imbalance_strategy="none",
                    )

                    threshold_result = select_validation_threshold(
                        y_true=y_validation.to_numpy(),
                        y_scores=result["validation_probabilities"],
                    )

                    model = result["model"]

                    test_probabilities = model.predict_proba(X_test)[:, 1]

                    test_metrics = evaluate_test_predictions(
                        y_true=y_test.to_numpy(),
                        y_scores=test_probabilities,
                        threshold=threshold_result["threshold"],
                    )

                    metrics = {
                        "validation_average_precision": result[
                            "average_precision"
                        ],
                        "validation_roc_auc": result["roc_auc"],
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
                    }

                    for item in test_metrics["review_budgets"]:
                        prefix = (
                            f"test_{int(item['review_budget'] * 100)}pct"
                        )

                        for key in (
                            "precision_at_k",
                            "recall_at_k",
                            "n_reviewed",
                            "true_positives",
                            "false_positives",
                            "cutoff_probability",
                        ):
                            metric_name = {
                                "precision_at_k": "precision",
                                "recall_at_k": "recall",
                                "n_reviewed": "n_reviewed",
                                "true_positives": "true_positives",
                                "false_positives": "false_positives",
                                "cutoff_probability": "cutoff_probability",
                            }[key]

                            metrics[f"{prefix}_{metric_name}"] = item[key]

                    mlflow.log_metrics(metrics)

                    mlflow.log_artifact(
                        str(split_config_path),
                        artifact_path="config",
                    )

                    # Save the fitted model.
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name="model",
                        serialization_format="pickle",
                    )

                    # Save test predictions with transaction IDs.
                    predictions = pd.DataFrame({
                        "txId": X_test.index.astype(str),
                        "y_true": y_test.to_numpy(),
                        "y_score": test_probabilities,
                        "y_pred": (
                            test_probabilities >= threshold_result["threshold"]
                        ).astype(int),
                    })

                    with tempfile.TemporaryDirectory() as tmpdir:
                        prediction_path = (
                            Path(tmpdir) / "test_predictions.csv"
                        )
                        predictions.to_csv(prediction_path, index=False)

                        mlflow.log_artifact(
                            str(prediction_path),
                            artifact_path="predictions",
                        )

                    run_id = mlflow.active_run().info.run_id

                    all_results[
                        (feature_set_name, model_name, seed)
                    ] = {
                        "model": model,
                        "validation_probabilities": (
                            result["validation_probabilities"]
                        ),
                        "test_probabilities": test_probabilities,
                        "test_labels": y_test.to_numpy(),
                        "test_metrics": test_metrics,
                        "selected_threshold": threshold_result["threshold"],
                        "run_id": run_id,
                        "random_state": seed,
                    }

                    print(
                        f"Validation AP: {result['average_precision']:.5f} | "
                        f"Test AP: {test_metrics['average_precision']:.5f} | "
                        f"Test ROC-AUC: {test_metrics['roc_auc']:.5f}"
                    )
                    print(f"MLflow run: {run_id}")

    return all_results
