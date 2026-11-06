"""One-shot, resumable Track B temporal-robustness runner.

Runs the four preselected configurations over all four Track B windows.
Unknown labels are excluded from supervised loss/evaluation but remain in
the transductive GNN graph for message passing.
"""

from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from fincrime.baselines.training import train_and_evaluate
from fincrime.evaluation.splits import build_split_masks
from fincrime.evaluation.test_evaluation import (
    evaluate_test_predictions,
    select_validation_threshold,
)
from fincrime.gnn.data import (
    attach_node_features,
    build_gnn_graph_structure,
)
from fincrime.gnn.training import train_gnn_model


TRACK_B_SEEDS = (42, 123, 2024)
TRACK_B_WINDOWS = ("cutoff_34", "cutoff_38", "cutoff_42", "cutoff_46")

TRACK_B_CONFIGS = (
    {"model": "xgboost", "feature_set": "Local", "kind": "tabular"},
    {"model": "xgboost", "feature_set": "Local+Hand", "kind": "tabular"},
    {"model": "graphsage", "feature_set": "Local", "kind": "gnn"},
    {"model": "graphsage", "feature_set": "All", "kind": "gnn"},
)


def _number(value: Any, cast=float):
    if pd.isna(value):
        raise ValueError("A required tuning parameter is missing.")
    return cast(value)


def load_track_b_selected_configs(csv_path: str | Path) -> dict:
    """Read the selected tuning configurations exported from MLflow."""
    table = pd.read_csv(csv_path)

    required_pairs = {
        ("xgboost", "Local"),
        ("xgboost", "Local+Hand"),
        ("graphsage", "Local"),
        ("graphsage", "All"),
    }

    configs = {}
    for _, row in table.iterrows():
        pair = (str(row["model"]).lower(), str(row["feature_set"]))
        if pair not in required_pairs:
            continue

        if pair[0] == "xgboost":
            params = {
                "n_estimators": _number(row["param_model__n_estimators"], int),
                "max_depth": _number(row["param_model__max_depth"], int),
                "learning_rate": _number(row["param_model__learning_rate"]),
                "min_child_weight": _number(row["param_model__min_child_weight"]),
                "subsample": _number(row["param_model__subsample"]),
                "colsample_bytree": _number(row["param_model__colsample_bytree"]),
                "reg_lambda": _number(row["param_model__reg_lambda"]),
                "objective": str(row["param_model__objective"]),
                "eval_metric": str(row["param_model__eval_metric"]),
                "tree_method": str(row["param_model__tree_method"]),
                "n_jobs": _number(row["param_model__n_jobs"], int),
            }
            configs[pair] = {
                "model": "xgboost",
                "feature_set": pair[1],
                "params": params,
                "scale_pos_weight": _number(row["param_scale_pos_weight"]),
                "tuning_run_id": str(row["run_id"]),
                "tuning_experiment": str(row["experiment_name"]),
                "tuning_validation_ap": _number(row["validation_ap"]),
            }
        else:
            configs[pair] = {
                "model": "graphsage",
                "feature_set": pair[1],
                "params": {
                    "hidden_channels": _number(row["param_hidden_channels"], int),
                    "dropout": _number(row["param_dropout"]),
                    "learning_rate": _number(row["param_learning_rate"]),
                    "weight_decay": _number(row["param_weight_decay"]),
                    "max_epochs": _number(row["param_max_epochs"], int),
                    "patience": _number(row["param_patience"], int),
                    "min_delta": _number(row["param_min_delta"]),
                },
                "optimizer": str(row["param_optimizer"]),
                "class_weighting": str(row["param_class_weighting"]),
                "tuning_run_id": str(row["run_id"]),
                "tuning_experiment": str(row["experiment_name"]),
                "tuning_validation_ap": _number(row["validation_ap"]),
            }

    missing = required_pairs - set(configs)
    if missing:
        raise ValueError(f"Selected tuning CSV is missing configurations: {sorted(missing)}")

    if configs[("graphsage", "Local")]["params"]["hidden_channels"] != 64:
        raise ValueError("GraphSAGE Local hidden size differs from the frozen selection.")
    if configs[("graphsage", "Local")]["params"]["dropout"] != 0.4:
        raise ValueError("GraphSAGE Local dropout differs from the frozen selection.")
    if configs[("graphsage", "All")]["params"]["hidden_channels"] != 64:
        raise ValueError("GraphSAGE All hidden size differs from the frozen selection.")
    if configs[("graphsage", "All")]["params"]["dropout"] != 0.3:
        raise ValueError("GraphSAGE All dropout differs from the frozen selection.")

    return configs


def _prepare_tabular_split(
    features: pd.DataFrame,
    labels: pd.Series,
    timesteps: pd.Series,
    split_config: dict,
    split_name: str,
) -> dict:
    if not features.index.equals(labels.index):
        raise ValueError("Feature and label indices are not aligned.")
    if not features.index.equals(timesteps.index):
        raise ValueError("Feature and timestep indices are not aligned.")
    if not features.index.is_unique:
        raise ValueError("Feature index contains duplicate transaction IDs.")
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature matrix contains missing or non-finite values.")

    labels_str = labels.astype(str)
    label_cfg = split_config["labels"]
    allowed = {
        str(label_cfg["illicit"]),
        str(label_cfg["licit"]),
        str(label_cfg["unknown"]),
    }
    unexpected = set(labels_str.unique()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")

    binary = labels_str.map({
        str(label_cfg["illicit"]): 1,
        str(label_cfg["licit"]): 0,
    })

    masks = build_split_masks(
        timesteps=timesteps,
        split_config=split_config,
        protocol="track_b",
        track_b_name=split_name,
    )
    known = binary.notna()

    result = {}
    for part in ("fit", "validation", "test"):
        mask = masks[part] & known
        X = features.loc[mask].copy()
        y = binary.loc[mask].astype("int8")
        if X.empty or y.nunique() != 2:
            raise ValueError(
                f"{split_name}: {part} must contain labelled examples from both classes."
            )
        result[f"X_{part}"] = X
        result[f"y_{part}"] = y
        result[f"{part}_ids"] = X.index

    return result


def _completed_run_keys(client: MlflowClient, experiment_id: str) -> set:
    completed = set()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=5000,
    )
    for run in runs:
        tags = run.data.tags
        if (
            run.info.status == "FINISHED"
            and tags.get("track") == "B"
            and "test_average_precision" in run.data.metrics
        ):
            try:
                completed.add((
                    tags["split_name"],
                    tags["model"],
                    tags["feature_set"],
                    int(run.data.params["random_seed"]),
                ))
            except (KeyError, ValueError):
                continue
    return completed


def _log_prediction_artifact(
    frame: pd.DataFrame,
    artifact_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / artifact_name
        frame.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path="predictions")


def _aggregate_mlflow_results(
    client: MlflowClient,
    experiment_id: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=5000,
    )
    rows = []
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        tags = run.data.tags
        metrics = run.data.metrics
        if tags.get("track") != "B" or "test_average_precision" not in metrics:
            continue
        params = run.data.params
        row = {
            "run_id": run.info.run_id,
            "start_time": run.info.start_time,
            "split_name": tags.get("split_name"),
            "cutoff": int(params["cutoff"]),
            "model": tags.get("model"),
            "feature_set": tags.get("feature_set"),
            "graph_protocol": tags.get("graph_protocol", "tabular"),
            "seed": int(params["random_seed"]),
            "validation_average_precision": metrics.get("validation_average_precision"),
            "test_average_precision": metrics.get("test_average_precision"),
            "test_roc_auc": metrics.get("test_roc_auc"),
            "test_f1": metrics.get("test_f1"),
            "test_precision": metrics.get("test_precision"),
            "test_recall": metrics.get("test_recall"),
            "validation_selected_threshold": metrics.get("validation_selected_threshold"),
            "best_epoch": metrics.get("best_epoch"),
            "training_runtime_seconds": metrics.get("training_runtime_seconds"),
        }
        for budget_pct in (1, 2, 5, 10):
            for metric_name in (
                "precision_at_k", "recall_at_k", "n_reviewed",
                "true_positives", "false_positives", "cutoff_probability",
            ):
                row[f"test_{budget_pct}pct_{metric_name}"] = metrics.get(
                    f"test_{budget_pct}pct_{metric_name}"
                )
        rows.append(row)

    run_level = pd.DataFrame(rows)
    if run_level.empty:
        raise RuntimeError("No completed Track B runs were found in MLflow.")

    run_level = (
        run_level.sort_values("start_time", ascending=False)
        .drop_duplicates(
            ["split_name", "model", "feature_set", "graph_protocol", "seed"],
            keep="first",
        )
    )

    group_cols = [
        "split_name", "cutoff", "model", "feature_set", "graph_protocol"
    ]
    metric_cols = [
        "validation_average_precision",
        "test_average_precision",
        "test_roc_auc",
        "test_f1",
        "test_precision",
        "test_recall",
        "training_runtime_seconds",
        *[
            f"test_{budget_pct}pct_{metric_name}"
            for budget_pct in (1, 2, 5, 10)
            for metric_name in ("precision_at_k", "recall_at_k")
        ],
    ]
    agg = run_level.groupby(group_cols).agg(
        runs=("run_id", "count"),
        **{
            f"mean_{metric}": (metric, "mean")
            for metric in metric_cols
        },
        **{
            f"std_{metric}": (metric, "std")
            for metric in [
                "test_average_precision",
                "test_roc_auc",
                "test_f1",
                "test_precision",
                "test_recall",
                *[
                    f"test_{budget_pct}pct_{metric_name}"
                    for budget_pct in (1, 2, 5, 10)
                    for metric_name in ("precision_at_k", "recall_at_k")
                ],
            ]
        },
    ).reset_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_level.to_csv(output_dir / "track_b_run_level_results.csv", index=False)
    agg.to_csv(output_dir / "track_b_aggregated_results.csv", index=False)
    return run_level, agg


def run_track_b(
    feature_sets: dict[str, pd.DataFrame],
    labels: pd.Series,
    timesteps: pd.Series,
    edges_df: pd.DataFrame,
    split_config: dict,
    selected_configs: dict,
    split_config_path: str | Path,
    output_dir: str | Path,
    experiment_name: str = "elliptic-track-b",
    tracking_uri: str = "http://127.0.0.1:5000",
    random_seeds: tuple[int, ...] = TRACK_B_SEEDS,
    windows: tuple[str, ...] = TRACK_B_WINDOWS,
    device: str | None = None,
) -> dict:
    """Execute/resume all Track B configurations and consolidate MLflow results."""

    expected_features = {"Local", "All", "Local+Hand"}
    if set(feature_sets) != expected_features:
        raise ValueError(
            "feature_sets must contain exactly Local, All and Local+Hand."
        )
    if not random_seeds or len(set(random_seeds)) != len(random_seeds):
        raise ValueError("random_seeds must be non-empty and unique.")
    if set(windows) - set(TRACK_B_WINDOWS):
        raise ValueError(f"Unknown Track B windows: {set(windows) - set(TRACK_B_WINDOWS)}")

    required_configs = {
        ("xgboost", "Local"),
        ("xgboost", "Local+Hand"),
        ("graphsage", "Local"),
        ("graphsage", "All"),
    }
    if not required_configs.issubset(selected_configs):
        raise ValueError("Selected configurations do not cover all four Track B models.")

    split_config_path = Path(split_config_path)
    if not split_config_path.is_file():
        raise FileNotFoundError(split_config_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    completed = _completed_run_keys(client, experiment_id)

    total_expected = len(windows) * len(TRACK_B_CONFIGS) * len(random_seeds)
    skipped = 0
    executed = 0

    print(
        f"Track B: {len(windows)} windows × {len(TRACK_B_CONFIGS)} "
        f"configurations × {len(random_seeds)} seeds = {total_expected} runs"
    )
    print(f"Already completed and reusable: {len(completed)}")

    for split_name in windows:
        split_item = next(
            item for item in split_config["track_b"]
            if item["name"] == split_name
        )
        fit_start, fit_end = split_item["fit"]
        val_start, val_end = split_item["validation"]
        test_start, test_end = split_item["test"]
        cutoff = int(fit_end + 4)

        print(
            f"\n{'=' * 72}\n{split_name}: "
            f"fit {fit_start}-{fit_end}, validation {val_start}-{val_end}, "
            f"test {test_start}-{test_end}\n{'=' * 72}"
        )

        # Build one transductive graph structure per window. It includes
        # fit + validation + test-period nodes; only fit labels enter loss.
        graph_structure = build_gnn_graph_structure(
            labels=labels,
            timesteps=timesteps,
            edges_df=edges_df,
            split_config=split_config,
            protocol="track_b",
            track_b_name=split_name,
        )

        # Prepare tabular data only for the two XGBoost feature sets.
        tabular_data = {
            name: _prepare_tabular_split(
                features=feature_sets[name],
                labels=labels,
                timesteps=timesteps,
                split_config=split_config,
                split_name=split_name,
            )
            for name in ("Local", "Local+Hand")
        }

        for config_spec in TRACK_B_CONFIGS:
            model_name = config_spec["model"]
            feature_set = config_spec["feature_set"]
            kind = config_spec["kind"]
            selected = selected_configs[(model_name, feature_set)]

            if kind == "gnn":
                data = attach_node_features(
                    graph_structure=graph_structure,
                    features=feature_sets[feature_set],
                )
                data.x = data.x.clone()  # avoid non-writable NumPy backing memory
                data.edge_index = data.edge_index.clone()
            else:
                data = tabular_data[feature_set]

            for seed in random_seeds:
                key = (split_name, model_name, feature_set, int(seed))
                if key in completed:
                    skipped += 1
                    print(f"SKIP completed: {key}")
                    continue

                run_name = (
                    f"Track_B__{split_name}__{model_name}__"
                    f"{feature_set.replace('+', '_')}__seed_{seed}"
                )
                graph_protocol = "directed" if kind == "gnn" else "tabular"

                with mlflow.start_run(run_name=run_name):
                    mlflow.set_tags({
                        "track": "B",
                        "split_name": split_name,
                        "model": model_name,
                        "feature_set": feature_set,
                        "graph_protocol": graph_protocol,
                        "evaluation_stage": "validation_and_test",
                        "source_tuning_run_id": selected["tuning_run_id"],
                    })
                    mlflow.log_params({
                        "random_seed": int(seed),
                        "cutoff": cutoff,
                        "fit_timestep_start": fit_start,
                        "fit_timestep_end": fit_end,
                        "validation_timestep_start": val_start,
                        "validation_timestep_end": val_end,
                        "test_timestep_start": test_start,
                        "test_timestep_end": test_end,
                        "n_features": int(
                            feature_sets[feature_set].shape[1]
                        ),
                        "source_tuning_run_id": selected["tuning_run_id"],
                        "source_tuning_experiment": selected["tuning_experiment"],
                        "source_tuning_validation_ap": selected["tuning_validation_ap"],
                    })

                    if kind == "tabular":
                        params = selected["params"]
                        mlflow.log_params({
                            "random_state": int(seed),
                            "scale_pos_weight": selected["scale_pos_weight"],
                            **{f"model__{k}": v for k, v in params.items()},
                        })

                        result = train_and_evaluate(
                            model_name="xgboost",
                            X_train=data["X_fit"],
                            y_train=data["y_fit"],
                            X_validation=data["X_validation"],
                            y_validation=data["y_validation"],
                            random_state=int(seed),
                            scale_pos_weight=selected["scale_pos_weight"],
                            model_params=params,
                        )
                        validation_scores = result["validation_probabilities"]
                        validation_labels = data["y_validation"].to_numpy()
                        test_scores = result["model"].predict_proba(
                            data["X_test"]
                        )[:, 1]
                        test_labels = data["y_test"].to_numpy()
                        test_ids = data["test_ids"]
                        validation_ids = data["validation_ids"]
                        test_timesteps = timesteps.loc[test_ids].to_numpy()
                        validation_timesteps = timesteps.loc[validation_ids].to_numpy()
                        best_epoch = None
                        runtime = None
                        peak_memory = None

                    else:
                        params = selected["params"]
                        mlflow.log_params({
                            "hidden_channels": params["hidden_channels"],
                            "dropout": params["dropout"],
                            "learning_rate": params["learning_rate"],
                            "weight_decay": params["weight_decay"],
                            "max_epochs": params["max_epochs"],
                            "patience": params["patience"],
                            "min_delta": params["min_delta"],
                            "optimizer": selected["optimizer"],
                            "class_weighting": selected["class_weighting"],
                            "num_graph_nodes": graph_structure["num_nodes"],
                            "num_graph_edges": graph_structure["num_edges"],
                            "device_requested": device or (
                                "cuda" if torch.cuda.is_available() else "cpu"
                            ),
                        })

                        result = train_gnn_model(
                            data=data,
                            model_name="graphsage",
                            hidden_channels=params["hidden_channels"],
                            dropout=params["dropout"],
                            learning_rate=params["learning_rate"],
                            weight_decay=params["weight_decay"],
                            max_epochs=params["max_epochs"],
                            patience=params["patience"],
                            min_delta=params["min_delta"],
                            seed=int(seed),
                            device=device,
                        )

                        validation_scores = result["validation_probabilities"]
                        validation_labels = result["validation_labels"]

                        model_device = next(result["model"].parameters()).device
                        result["model"].eval()
                        with torch.no_grad():
                            logits = result["model"](
                                data.x.to(model_device),
                                data.edge_index.to(model_device),
                            )
                            test_scores = torch.softmax(logits, dim=1)[
                                data.test_mask.to(model_device), 1
                            ].detach().cpu().numpy()

                        test_mask_cpu = data.test_mask.cpu().numpy()
                        test_labels = data.y.cpu().numpy()[test_mask_cpu]
                        test_ids = graph_structure["node_ids"][test_mask_cpu]
                        val_mask_cpu = data.val_mask.cpu().numpy()
                        validation_ids = graph_structure["node_ids"][val_mask_cpu]
                        test_timesteps = timesteps.loc[test_ids].to_numpy()
                        validation_timesteps = timesteps.loc[validation_ids].to_numpy()
                        best_epoch = result["best_epoch"]
                        runtime = result["training_runtime_seconds"]
                        peak_memory = result["peak_gpu_memory_mb"]

                        with tempfile.TemporaryDirectory() as tmp:
                            checkpoint = Path(tmp) / "model_state.pt"
                            torch.save(result["model"].state_dict(), checkpoint)
                            mlflow.log_artifact(str(checkpoint), artifact_path="model")

                    validation_ap = float(
                        average_precision_score(validation_labels, validation_scores)
                    )
                    validation_auc = float(
                        roc_auc_score(validation_labels, validation_scores)
                    )
                    threshold_result = select_validation_threshold(
                        validation_labels,
                        validation_scores,
                    )
                    test_metrics = evaluate_test_predictions(
                        y_true=test_labels,
                        y_scores=test_scores,
                        threshold=threshold_result["threshold"],
                    )

                    logged_metrics = {
                        "validation_average_precision": validation_ap,
                        "validation_roc_auc": validation_auc,
                        "validation_selected_threshold": threshold_result["threshold"],
                        "validation_selected_f1": threshold_result["validation_f1"],
                        "test_average_precision": test_metrics["average_precision"],
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
                        budget_pct = int(item["review_budget"] * 100)
                        logged_metrics.update({
                            f"test_{budget_pct}pct_precision_at_k": item["precision_at_k"],
                            f"test_{budget_pct}pct_recall_at_k": item["recall_at_k"],
                            f"test_{budget_pct}pct_n_reviewed": item["n_reviewed"],
                            f"test_{budget_pct}pct_true_positives": item["true_positives"],
                            f"test_{budget_pct}pct_false_positives": item["false_positives"],
                            f"test_{budget_pct}pct_cutoff_probability": item["cutoff_probability"],
                        })
                    mlflow.log_metrics(logged_metrics)
                    if best_epoch is not None:
                        mlflow.log_metric("best_epoch", best_epoch)
                    if runtime is not None:
                        mlflow.log_metric("training_runtime_seconds", runtime)
                    if peak_memory is not None:
                        mlflow.log_metric("peak_gpu_memory_mb", peak_memory)

                    validation_predictions = pd.DataFrame({
                        "txId": pd.Index(validation_ids).astype(str),
                        "timestep": validation_timesteps,
                        "y_true": validation_labels,
                        "y_score": validation_scores,
                    })
                    test_predictions = pd.DataFrame({
                        "txId": pd.Index(test_ids).astype(str),
                        "timestep": test_timesteps,
                        "y_true": test_labels,
                        "y_score": test_scores,
                        "y_pred": (
                            test_scores >= threshold_result["threshold"]
                        ).astype(int),
                    })
                    _log_prediction_artifact(
                        validation_predictions,
                        "validation_predictions.csv",
                    )
                    _log_prediction_artifact(
                        test_predictions,
                        "test_predictions.csv",
                    )
                    mlflow.log_artifact(str(split_config_path), artifact_path="config")

                    executed += 1
                    print(
                        f"Completed {key} | "
                        f"val AP={validation_ap:.5f} | "
                        f"test AP={test_metrics['average_precision']:.5f} | "
                        f"test ROC-AUC={test_metrics['roc_auc']:.5f}"
                    )

                if kind == "gnn":
                    del result
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if kind == "gnn":
                del data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del graph_structure
        gc.collect()

    run_level, aggregate = _aggregate_mlflow_results(
        client=client,
        experiment_id=experiment_id,
        output_dir=output_dir,
    )

    print("\nTRACK B EXECUTION SUMMARY")
    print(f"Expected runs for requested windows: {total_expected}")
    print(f"New runs executed this session: {executed}")
    print(f"Previously completed runs skipped: {skipped}")
    print(f"Completed Track B runs in MLflow: {len(run_level)}")
    print(f"Run-level results: {output_dir / 'track_b_run_level_results.csv'}")
    print(f"Aggregated results: {output_dir / 'track_b_aggregated_results.csv'}")

    return {
        "run_level_results": run_level,
        "aggregated_results": aggregate,
        "experiment_id": experiment_id,
        "new_runs_executed": executed,
        "previous_runs_skipped": skipped,
        "output_dir": output_dir,
    }