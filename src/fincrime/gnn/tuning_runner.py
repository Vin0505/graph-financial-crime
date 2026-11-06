"""MLflow-integrated runner for GCN and GraphSAGE tuning experiments."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import csv
import gc

import mlflow
import torch

from fincrime.gnn.data import (
    attach_node_features,
    build_gnn_graph_structure,
)
from fincrime.gnn.training import train_gnn_model
from fincrime.gnn.tuning import (
    load_gnn_tuning_config,
    resolve_gnn_trial,
)


def run_gnn_tuning(
    feature_sets: Mapping[str, Any],
    labels,
    timesteps,
    edges_df,
    split_config: dict,
    config_paths: Mapping[str, str | Path],
    experiment_name: str = "elliptic-gnn-tuning",
    tracking_uri: str = "http://127.0.0.1:5000",
    summary_path: str | Path = "reports/gnn_tuning_summary.csv",
) -> dict:
    """
    Run six trials for each GNN-feature-set combination.

    Expected feature sets:
        Local: 93 local features
        All: Local + 72 supplied neighbour aggregates

    Each trial is logged as an independent MLflow run.
    Hyperparameters are selected using tuning-window validation AP.
    """

    if not isinstance(feature_sets, Mapping):
        raise TypeError("feature_sets must be a mapping.")

    required_feature_sets = {"Local", "All"}

    if set(feature_sets) != required_feature_sets:
        raise ValueError(
            "feature_sets must contain exactly 'Local' and 'All'."
        )

    if not isinstance(config_paths, Mapping):
        raise TypeError("config_paths must be a mapping.")

    required_models = {"gcn", "graphsage"}

    if set(config_paths) != required_models:
        raise ValueError(
            "config_paths must contain exactly 'gcn' and 'graphsage'."
        )

    loaded_configs = {
        model_name: load_gnn_tuning_config(
            config_path=config_path,
            expected_model=model_name,
        )
        for model_name, config_path in config_paths.items()
    }

    config_paths = {
        model_name: Path(path)
        for model_name, path in config_paths.items()
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("\nBuilding the shared tuning graph structure...")

    graph_structure = build_gnn_graph_structure(
        labels=labels,
        timesteps=timesteps,
        edges_df=edges_df,
        split_config=split_config,
        protocol="tuning",
    )

    all_results = {}
    summary_rows = []

    for model_name in ("gcn", "graphsage"):

        model_config = loaded_configs[model_name]
        config_path = config_paths[model_name]

        for feature_set_name in ("Local", "All"):

            features = feature_sets[feature_set_name]

            print(f"\n{'=' * 70}")
            print(f"Model: {model_name.upper()} | Feature set: {feature_set_name}")
            print(f"{'=' * 70}")

            # The graph topology and masks are shared; only x changes.
            data = attach_node_features(
                graph_structure=graph_structure,
                features=features,
            )

            feature_set_results = []

            for trial in model_config["tuning"]["trials"]:

                resolved = resolve_gnn_trial(
                    config=model_config,
                    trial=trial,
                )

                trial_name = resolved["trial_name"]

                print(f"\nRunning {model_name} | {feature_set_name} | {trial_name}")

                with mlflow.start_run(
                    run_name=(
                        f"{model_name}_{feature_set_name.lower()}__{trial_name}"
                    )
                ):

                    mlflow.set_tags({
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "trial_name": trial_name,
                        "selection_metric": "average_precision",
                        "split": "tuning",
                        "graph_protocol": "directed",
                    })

                    mlflow.log_params({
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "trial_name": trial_name,
                        "hidden_channels": resolved["hidden_channels"],
                        "dropout": resolved["dropout"],
                        "optimizer": resolved["optimizer"],
                        "learning_rate": resolved["learning_rate"],
                        "weight_decay": resolved["weight_decay"],
                        "max_epochs": resolved["max_epochs"],
                        "patience": resolved["patience"],
                        "min_delta": resolved["min_delta"],
                        "class_weighting": resolved["class_weighting"],
                        "n_features": data.x.shape[1],
                        "n_nodes": data.num_nodes,
                        "n_edges": data.edge_index.shape[1],
                        "device_requested": (
                            "cuda" if torch.cuda.is_available() else "cpu"
                        ),
                    })

                    result = train_gnn_model(
                        data=data,
                        model_name=model_name,
                        hidden_channels=resolved["hidden_channels"],
                        dropout=resolved["dropout"],
                        learning_rate=resolved["learning_rate"],
                        weight_decay=resolved["weight_decay"],
                        max_epochs=resolved["max_epochs"],
                        patience=resolved["patience"],
                        min_delta=resolved["min_delta"],
                    )

                    metrics = {
                        "validation_average_precision": result["best_val_ap"],
                        "best_epoch": result["best_epoch"],
                        "training_runtime_seconds": result[
                            "training_runtime_seconds"
                        ],
                    }

                    if result["peak_gpu_memory_mb"] is not None:
                        metrics["peak_gpu_memory_mb"] = result[
                            "peak_gpu_memory_mb"
                        ]

                    mlflow.log_metrics(metrics)

                    mlflow.log_params({
                        "seed": result["seed"],
                        "device": result["device"],
                        "positive_training_count": int(
                            data.y[data.train_mask].eq(1).sum().item()
                        ),
                        "negative_training_count": int(
                            data.y[data.train_mask].eq(0).sum().item()
                        ),
                    })

                    mlflow.log_artifact(
                        str(config_path),
                        artifact_path="config",
                    )

                    run_id = mlflow.active_run().info.run_id

                    trial_result = {
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "trial_name": trial_name,
                        "run_id": run_id,
                        "average_precision": result["best_val_ap"],
                        "best_epoch": result["best_epoch"],
                        "training_runtime_seconds": result[
                            "training_runtime_seconds"
                        ],
                        "peak_gpu_memory_mb": result[
                            "peak_gpu_memory_mb"
                        ],
                        "seed": result["seed"],
                        "device": result["device"],
                        "resolved_parameters": resolved,
                    }

                    # Keep the model only when it becomes the best trial
                    # for this model-feature-set combination.
                    if (
                        not feature_set_results
                        or trial_result["average_precision"]
                        > max(
                            item["average_precision"]
                            for item in feature_set_results
                        )
                    ):
                        trial_result["model"] = result["model"]
                        trial_result["history"] = result["history"]

                    feature_set_results.append(trial_result)

                    summary_rows.append({
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "trial_name": trial_name,
                        "run_id": run_id,
                        "validation_average_precision": result["best_val_ap"],
                        "best_epoch": result["best_epoch"],
                        "training_runtime_seconds": result[
                            "training_runtime_seconds"
                        ],
                        "peak_gpu_memory_mb": result[
                            "peak_gpu_memory_mb"
                        ],
                        "seed": result["seed"],
                        "device": result["device"],
                        "hidden_channels": resolved["hidden_channels"],
                        "dropout": resolved["dropout"],
                        "learning_rate": resolved["learning_rate"],
                        "weight_decay": resolved["weight_decay"],
                    })

                    print(
                        f"  Validation AP: {result['best_val_ap']:.6f}"
                    )
                    print(f"  Best epoch: {result['best_epoch']}")
                    print(
                        "  Runtime: "
                        f"{result['training_runtime_seconds']:.2f} seconds"
                    )

                    if result["peak_gpu_memory_mb"] is not None:
                        print(
                            "  Peak GPU memory: "
                            f"{result['peak_gpu_memory_mb']:.2f} MB"
                        )

                    del result
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            best_trial = max(
                feature_set_results,
                key=lambda item: item["average_precision"],
            )

            all_results[(model_name, feature_set_name)] = {
                "trials": feature_set_results,
                "best_trial": best_trial,
                "best_model": best_trial["model"],
                "best_run_id": best_trial["run_id"],
            }

            print(f"\nBest {model_name.upper()} trial on {feature_set_name}:")
            print(f"  Trial: {best_trial['trial_name']}")
            print(f"  Validation AP: {best_trial['average_precision']:.6f}")
            print(f"  MLflow run ID: {best_trial['run_id']}")

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "feature_set",
        "trial_name",
        "run_id",
        "validation_average_precision",
        "best_epoch",
        "training_runtime_seconds",
        "peak_gpu_memory_mb",
        "seed",
        "device",
        "hidden_channels",
        "dropout",
        "learning_rate",
        "weight_decay",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nTuning summary saved to: {summary_path}")

    # Select the primary architecture using its best AP across Local and All.
    architecture_results = {}

    for model_name in ("gcn", "graphsage"):
        best_trial = max(
            (
                all_results[(model_name, feature_set)]["best_trial"]
                for feature_set in ("Local", "All")
            ),
            key=lambda item: item["average_precision"],
        )

        architecture_results[model_name] = {
            "best_trial": best_trial,
            "best_validation_ap": best_trial["average_precision"],
            "parameter_count": sum(
                parameter.numel()
                for parameter in best_trial["model"].parameters()
            ),
        }

    gcn_ap = architecture_results["gcn"]["best_validation_ap"]
    sage_ap = architecture_results["graphsage"]["best_validation_ap"]

    if gcn_ap > sage_ap:
        primary_model = "gcn"
    elif sage_ap > gcn_ap:
        primary_model = "graphsage"
    else:
        gcn_parameters = architecture_results["gcn"]["parameter_count"]
        sage_parameters = architecture_results["graphsage"]["parameter_count"]

        primary_model = (
            "gcn" if gcn_parameters <= sage_parameters else "graphsage"
        )

    secondary_model = (
        "graphsage" if primary_model == "gcn" else "gcn"
    )

    print("\nPrimary GNN selection:")
    print(
        f"  GCN best validation AP: {gcn_ap:.6f}"
    )
    print(
        f"  GraphSAGE best validation AP: {sage_ap:.6f}"
    )
    print(f"  Primary GNN: {primary_model.upper()}")
    print(f"  Secondary GNN: {secondary_model.upper()}")

    return {
        "results": all_results,
        "summary_path": summary_path,
        "architecture_results": architecture_results,
        "primary_model": primary_model,
        "secondary_model": secondary_model,
    }