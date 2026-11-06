
"""Resumable MLflow runner for Track A GNN evaluation."""

from collections.abc import Mapping
from pathlib import Path
import csv
import gc
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch_geometric.utils import coalesce

from fincrime.evaluation.test_evaluation import (
    evaluate_test_predictions,
    select_validation_threshold,
)
from fincrime.gnn.data import (
    attach_node_features,
    build_gnn_graph_structure,
)
from fincrime.gnn.training import train_gnn_model


TRACK_A_SEEDS = (42, 123, 2024, 7, 21)


def _select_validation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )

    if thresholds.size == 0:
        raise ValueError("Cannot select a threshold without score variation.")

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_f1 = f1_scores.max()
    best_indices = np.flatnonzero(
        np.isclose(f1_scores, best_f1)
    )

    # In a tie, use the higher threshold to reduce unnecessary alerts.
    best_index = best_indices[-1]
    threshold = float(thresholds[best_index])

    predictions = (probabilities >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(best_f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def _calculate_ranking_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    metrics = {
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
    }

    n_samples = len(labels)
    n_positives = int(labels.sum())

    for budget in (0.01, 0.05, 0.10):
        k = max(1, int(np.ceil(n_samples * budget)))
        top_indices = np.argsort(
            -probabilities,
            kind="stable",
        )[:k]

        true_positives = int(labels[top_indices].sum())

        suffix = f"{int(budget * 100)}pct"

        metrics[f"precision_at_{suffix}"] = true_positives / k
        metrics[f"recall_at_{suffix}"] = (
            true_positives / n_positives
            if n_positives
            else 0.0
        )

    return metrics


def _symmetrise_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    reverse_edges = edge_index.flip(0)

    combined_edges = torch.cat(
        [edge_index, reverse_edges],
        dim=1,
    )

    return coalesce(
        combined_edges,
        num_nodes=num_nodes,
    )


def _get_selected_parameters(
    tuning_results: dict,
    model_name: str,
    feature_set: str,
) -> dict:
    try:
        best_trial = tuning_results["results"][
            (model_name, feature_set)
        ]["best_trial"]

        parameters = best_trial["resolved_parameters"]

    except KeyError as exc:
        raise ValueError(
            f"Missing selected tuning result for "
            f"{model_name} / {feature_set}."
        ) from exc

    required = {
        "hidden_channels",
        "dropout",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "patience",
        "min_delta",
    }

    missing = required.difference(parameters)

    if missing:
        raise ValueError(
            f"Selected parameters are missing: {sorted(missing)}"
        )

    return {
        "trial_name": best_trial["trial_name"],
        "tuning_run_id": best_trial["run_id"],
        **{
            key: parameters[key]
            for key in required
        },
    }


def _predict_test_probabilities(
    model: torch.nn.Module,
    data,
) -> np.ndarray:
    """
    Generate probabilities for labelled test nodes.

    Test labels are not accessed during inference. All graph nodes and
    edges remain available for message passing, as specified by the
    transductive evaluation protocol.
    """
    device = next(model.parameters()).device

    model.eval()

    with torch.no_grad():
        logits = model(
            data.x.to(device),
            data.edge_index.to(device),
        )

        test_mask = data.test_mask.to(device)

        probabilities = torch.softmax(logits, dim=1)[
            test_mask, 1
        ]

    return probabilities.detach().cpu().numpy()


# ---------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------

def _has_model_checkpoint(
    client: MlflowClient,
    run_id: str,
) -> bool:
    """
    Check recursively for model/model_state.pt in the MLflow artifacts.

    MLflow may return artifact paths differently depending on whether
    the artifact directory or its parent is listed. Recursive checking
    avoids relying on one particular path representation.
    """
    try:
        pending_paths = [""]

        while pending_paths:
            current_path = pending_paths.pop()

            artifacts = client.list_artifacts(
                run_id,
                current_path or None,
            )

            for artifact in artifacts:
                if artifact.is_dir:
                    pending_paths.append(artifact.path)

                elif Path(artifact.path).name == "model_state.pt":
                    if artifact.path.replace("\\", "/").endswith(
                        "model/model_state.pt"
                    ):
                        return True

    except Exception as exc:
        print(
            f"Could not inspect checkpoint for run {run_id}: {exc}"
        )
        return False

    return False


def _find_completed_runs(
    client: MlflowClient,
    experiment_id: str,
) -> dict:
    """
    Find completed Track A runs that are safe to skip.

    A run is considered complete only when:
      - MLflow status is FINISHED.
      - Track/model/feature/protocol tags match.
      - Evaluation stage is validation_and_test.
      - Test AP has been logged.
      - The model checkpoint exists.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=1000,
    )

    completed_runs = {}

    for run in runs:
        if run.info.status != "FINISHED":
            continue

        tags = run.data.tags
        params = run.data.params
        metrics = run.data.metrics

        if tags.get("track") != "A":
            continue

        if tags.get("evaluation_stage") != "validation_and_test":
            continue

        model_name = tags.get("model")
        feature_set = tags.get("feature_set")
        graph_protocol = tags.get("graph_protocol")

        if not all(
            [model_name, feature_set, graph_protocol]
        ):
            continue

        seed_value = params.get("random_seed")

        if seed_value is None:
            continue

        # Test metrics are required; validation-only runs are excluded.
        if "test_average_precision" not in metrics:
            continue

        if not _has_model_checkpoint(
            client,
            run.info.run_id,
        ):
            continue

        key = (
            model_name,
            feature_set,
            graph_protocol,
            int(seed_value),
        )

        # Retain the most recently started matching completed run.
        existing = completed_runs.get(key)

        if (
            existing is None
            or run.info.start_time > existing.info.start_time
        ):
            completed_runs[key] = run

    return completed_runs


def _recover_completed_result(
    run,
    result_key: tuple,
) -> dict:
    """
    Reconstruct a completed run's scalar results from MLflow.

    Prediction arrays are unavailable for the earlier runs because
    they were not logged as artifacts.
    """
    model_name, feature_set, graph_protocol, seed = result_key

    metrics = run.data.metrics
    params = run.data.params

    result = {
        "model": model_name,
        "feature_set": feature_set,
        "graph_protocol": graph_protocol,
        "seed": seed,
        "run_id": run.info.run_id,
        "source_tuning_run_id": params.get(
            "source_tuning_run_id"
        ),
        "best_epoch": metrics.get("best_epoch"),
        "training_runtime_seconds": metrics.get(
            "training_runtime_seconds"
        ),
        "peak_gpu_memory_mb": metrics.get(
            "peak_gpu_memory_mb"
        ),
    }

    # Recover all original validation metrics.
    validation_metric_names = [
        "average_precision",
        "roc_auc",
        "precision_at_1pct",
        "recall_at_1pct",
        "precision_at_5pct",
        "recall_at_5pct",
        "precision_at_10pct",
        "recall_at_10pct",
    ]

    for name in validation_metric_names:
        result[f"validation_{name}"] = metrics.get(
            f"validation_{name}"
        )

    validation_threshold_names = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ]

    for name in validation_threshold_names:
        result[f"validation_{name}"] = metrics.get(
            f"validation_{name}"
        )

    result["validation_selected_threshold"] = metrics.get(
        "validation_selected_threshold"
    )
    result["validation_selected_f1"] = metrics.get(
        "validation_selected_f1"
    )

    # Recover all test metrics logged by the original runner.
    test_metric_names = [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "threshold",
        "n",
        "prevalence",
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    for name in test_metric_names:
        result[f"test_{name}"] = metrics.get(
            f"test_{name}"
        )

    for budget in (1, 5, 10):
        for name in (
            "precision",
            "recall",
            "n_reviewed",
            "true_positives",
            "false_positives",
            "cutoff_probability",
        ):
            metric_name = f"test_{budget}pct_{name}"
            result[metric_name] = metrics.get(metric_name)

    # These were not saved for the earlier completed runs.
    result.update({
        "validation_probabilities": None,
        "validation_labels": None,
        "test_probabilities": None,
        "test_labels": None,
        "test_metrics": None,
    })

    return result


def run_track_a_gnn(
    feature_sets: Mapping,
    labels,
    timesteps,
    edges_df,
    split_config: dict,
    tuning_results: dict,
    split_config_path: str | Path,
    experiment_name: str = "elliptic-track-a-gnn",
    tracking_uri: str = "http://127.0.0.1:5000",
    random_seeds: tuple[int, ...] = TRACK_A_SEEDS,
    summary_path: str | Path = "reports/track_a_gnn_summary.csv",
) -> dict:
    """
    Evaluate the selected primary and secondary GNNs on Track A.

    Runs:
        Primary GNN: Local and All, five seeds each.
        Secondary GNN: Local, five seeds.
        Primary GNN: Local with symmetrised edges, five seeds.

    Test labels are not used for training, model selection,
    checkpoint selection, or threshold selection.

    Completed MLflow runs are skipped when resuming.
    """

    if not isinstance(feature_sets, Mapping):
        raise TypeError("feature_sets must be a mapping.")

    if set(feature_sets) != {"Local", "All"}:
        raise ValueError(
            "feature_sets must contain exactly Local and All."
        )

    if not random_seeds:
        raise ValueError("At least one random seed is required.")

    if len(set(random_seeds)) != len(random_seeds):
        raise ValueError("random_seeds must be unique.")

    if len(random_seeds) != 5:
        raise ValueError(
            "Track A requires exactly five random seeds."
        )

    split_config_path = Path(split_config_path)

    if not split_config_path.is_file():
        raise FileNotFoundError(
            f"Split configuration not found: {split_config_path}"
        )

    primary_model = tuning_results.get("primary_model")
    secondary_model = tuning_results.get("secondary_model")

    if primary_model not in {"gcn", "graphsage"}:
        raise ValueError("Invalid primary GNN in tuning results.")

    if secondary_model not in {"gcn", "graphsage"}:
        raise ValueError("Invalid secondary GNN in tuning results.")

    if primary_model == secondary_model:
        raise ValueError("Primary and secondary GNNs must differ.")

    selected_parameters = {
        (primary_model, feature_set): _get_selected_parameters(
            tuning_results,
            primary_model,
            feature_set,
        )
        for feature_set in ("Local", "All")
    }

    selected_parameters[(secondary_model, "Local")] = (
        _get_selected_parameters(
            tuning_results,
            secondary_model,
            "Local",
        )
    )

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    completed_runs = _find_completed_runs(
        client=client,
        experiment_id=experiment.experiment_id,
    )

    print(
        f"\nPreviously completed runs with test metrics "
        f"and checkpoints: {len(completed_runs)}"
    )

    print("\nBuilding the Track A graph structure...")

    graph_structure = build_gnn_graph_structure(
        labels=labels,
        timesteps=timesteps,
        edges_df=edges_df,
        split_config=split_config,
        protocol="track_a",
    )

    all_results = {}
    summary_rows = []

    experiment_cells = [
        (primary_model, "Local", "directed"),
        (primary_model, "All", "directed"),
        (secondary_model, "Local", "directed"),
        (primary_model, "Local", "undirected"),
    ]

    for model_name, feature_set_name, graph_protocol in experiment_cells:

        parameters = selected_parameters[
            (model_name, feature_set_name)
        ]

        data = attach_node_features(
            graph_structure=graph_structure,
            features=feature_sets[feature_set_name],
        )

        if graph_protocol == "undirected":
            data.edge_index = _symmetrise_edge_index(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
            )

        print(
            f"\n{'=' * 65}\n"
            f"Track A | {model_name.upper()} | {feature_set_name} | "
            f"{graph_protocol}\n"
            f"{'=' * 65}"
        )

        for seed in random_seeds:

            result_key = (
                model_name,
                feature_set_name,
                graph_protocol,
                seed,
            )

            # ---------------------------------------------------------
            # Resume: skip only verified completed runs.
            # ---------------------------------------------------------
            if result_key in completed_runs:
                existing_run = completed_runs[result_key]

                print(
                    f"\nSKIPPING completed run: {result_key}\n"
                    f"MLflow run ID: {existing_run.info.run_id}"
                )

                run_result = _recover_completed_result(
                    run=existing_run,
                    result_key=result_key,
                )

                all_results[result_key] = run_result

                summary_rows.append({
                    key: value
                    for key, value in run_result.items()
                    if key not in {
                        "validation_probabilities",
                        "validation_labels",
                        "test_probabilities",
                        "test_labels",
                        "test_metrics",
                    }
                })

                continue

            run_name = (
                f"Track_A__{model_name}__{feature_set_name}"
                f"__{graph_protocol}__seed_{seed}"
            )

            print(f"\nStarting new run: {run_name}")

            with mlflow.start_run(run_name=run_name):

                mlflow.set_tags({
                    "track": "A",
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "graph_protocol": graph_protocol,
                    "split": "track_a",
                    "evaluation_stage": "validation_and_test",
                    "source_tuning_trial": parameters["trial_name"],
                })

                mlflow.log_params({
                    "source_tuning_run_id": parameters["tuning_run_id"],
                    "random_seed": seed,
                    "hidden_channels": parameters["hidden_channels"],
                    "dropout": parameters["dropout"],
                    "learning_rate": parameters["learning_rate"],
                    "weight_decay": parameters["weight_decay"],
                    "max_epochs": parameters["max_epochs"],
                    "patience": parameters["patience"],
                    "min_delta": parameters["min_delta"],
                    "n_features": data.x.shape[1],
                    "n_nodes": data.num_nodes,
                    "n_edges": data.edge_index.shape[1],
                    "fit_timestep_start": 1,
                    "fit_timestep_end": 30,
                    "validation_timestep_start": 31,
                    "validation_timestep_end": 34,
                    "test_timestep_start": 35,
                    "test_timestep_end": 49,
                })

                result = train_gnn_model(
                    data=data,
                    model_name=model_name,
                    hidden_channels=parameters["hidden_channels"],
                    dropout=parameters["dropout"],
                    learning_rate=parameters["learning_rate"],
                    weight_decay=parameters["weight_decay"],
                    max_epochs=parameters["max_epochs"],
                    patience=parameters["patience"],
                    min_delta=parameters["min_delta"],
                    seed=seed,
                )

                validation_labels = result["validation_labels"]
                validation_probabilities = result[
                    "validation_probabilities"
                ]

                ranking_metrics = _calculate_ranking_metrics(
                    validation_labels,
                    validation_probabilities,
                )

                threshold_metrics = _select_validation_threshold(
                    validation_labels,
                    validation_probabilities,
                )

                # Shared utility selects the threshold used for test evaluation.
                threshold_result = select_validation_threshold(
                    y_true=validation_labels,
                    y_scores=validation_probabilities,
                )

                # Generate test predictions before accessing test labels.
                test_probabilities = _predict_test_probabilities(
                    model=result["model"],
                    data=data,
                )

                # Test labels are accessed only after inference is complete.
                test_labels = (
                    data.y[data.test_mask]
                    .detach()
                    .cpu()
                    .numpy()
                )

                test_metrics = evaluate_test_predictions(
                    y_true=test_labels,
                    y_scores=test_probabilities,
                    threshold=threshold_result["threshold"],
                )

                test_flat_metrics = {
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
                }

                mlflow.log_metrics({
                    "validation_average_precision": (
                        ranking_metrics["average_precision"]
                    ),
                    "validation_roc_auc": ranking_metrics["roc_auc"],
                    "best_epoch": result["best_epoch"],
                    "training_runtime_seconds": (
                        result["training_runtime_seconds"]
                    ),
                    "validation_threshold": threshold_metrics["threshold"],
                    "validation_precision": threshold_metrics["precision"],
                    "validation_recall": threshold_metrics["recall"],
                    "validation_f1": threshold_metrics["f1"],
                    "validation_true_positives": (
                        threshold_metrics["true_positives"]
                    ),
                    "validation_false_positives": (
                        threshold_metrics["false_positives"]
                    ),
                    "validation_true_negatives": (
                        threshold_metrics["true_negatives"]
                    ),
                    "validation_false_negatives": (
                        threshold_metrics["false_negatives"]
                    ),
                    "validation_selected_threshold": (
                        threshold_result["threshold"]
                    ),
                    "validation_selected_f1": (
                        threshold_result["validation_f1"]
                    ),
                    **{
                        f"validation_{key}": value
                        for key, value in ranking_metrics.items()
                        if key not in {"average_precision", "roc_auc"}
                    },
                    **test_flat_metrics,
                })

                if result["peak_gpu_memory_mb"] is not None:
                    mlflow.log_metric(
                        "peak_gpu_memory_mb",
                        result["peak_gpu_memory_mb"],
                    )

                mlflow.log_artifact(
                    str(split_config_path),
                    artifact_path="config",
                )

                # Save the restored best-validation model checkpoint.
                with tempfile.TemporaryDirectory() as temp_dir:
                    checkpoint_path = (
                        Path(temp_dir) / "model_state.pt"
                    )

                    torch.save(
                        result["model"].state_dict(),
                        checkpoint_path,
                    )

                    mlflow.log_artifact(
                        str(checkpoint_path),
                        artifact_path="model",
                    )

                run_id = mlflow.active_run().info.run_id

                run_result = {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "graph_protocol": graph_protocol,
                    "seed": seed,
                    "run_id": run_id,
                    "source_tuning_run_id": parameters["tuning_run_id"],
                    "best_epoch": result["best_epoch"],
                    "training_runtime_seconds": (
                        result["training_runtime_seconds"]
                    ),
                    "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                    **{
                        f"validation_{key}": value
                        for key, value in ranking_metrics.items()
                    },
                    **{
                        f"validation_{key}": value
                        for key, value in threshold_metrics.items()
                    },
                    "validation_selected_threshold": (
                        threshold_result["threshold"]
                    ),
                    "validation_selected_f1": (
                        threshold_result["validation_f1"]
                    ),
                    **test_flat_metrics,
                    "validation_probabilities": validation_probabilities,
                    "validation_labels": validation_labels,
                    "test_probabilities": test_probabilities,
                    "test_labels": test_labels,
                    "test_metrics": test_metrics,
                }

                all_results[result_key] = run_result

                summary_rows.append({
                    key: value
                    for key, value in run_result.items()
                    if key not in {
                        "validation_probabilities",
                        "validation_labels",
                        "test_probabilities",
                        "test_labels",
                        "test_metrics",
                    }
                })

                print(
                    f"Seed {seed} | "
                    f"Validation AP: "
                    f"{ranking_metrics['average_precision']:.6f} | "
                    f"Test AP: "
                    f"{test_metrics['average_precision']:.6f} | "
                    f"Test ROC-AUC: "
                    f"{test_metrics['roc_auc']:.6f} | "
                    f"Test F1: {test_metrics['f1']:.6f} | "
                    f"Best epoch: {result['best_epoch']}"
                )

                print(
                    f"Validation-selected threshold: "
                    f"{threshold_result['threshold']:.6f}"
                )

                print(f"MLflow run ID: {run_id}")

                del result
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del data
        gc.collect()

    # -------------------------------------------------------------
    # Save combined summary, including previously completed runs.
    # -------------------------------------------------------------
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        raise RuntimeError(
            "No completed or newly evaluated Track A runs were found."
        )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    skipped_count = sum(
        1 for key in all_results
        if key in completed_runs
    )
    newly_completed_count = len(summary_rows) - skipped_count

    print(f"\nTrack A GNN summary saved to: {summary_path}")
    print(f"Previously completed runs skipped: {skipped_count}")
    print(f"New GNN fits completed: {newly_completed_count}")
    print(f"Total completed runs: {len(summary_rows)}")

    return {
        "results": all_results,
        "summary_path": summary_path,
        "primary_model": primary_model,
        "secondary_model": secondary_model,
        "random_seeds": random_seeds,
    }
