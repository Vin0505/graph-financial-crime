
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "fincrime-infrastructure-smoke-test"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="windows-to-docker-smoke-test",
):
    mlflow.log_param("model", "synthetic_test")
    mlflow.log_param("feature_set", "test")

    mlflow.log_metric("validation_ap", 0.95)
    mlflow.log_metric("validation_roc_auc", 0.98)

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = Path(temp_dir) / "smoke_test.txt"
        artifact_path.write_text(
            "MLflow artifact proxy test successful.",
            encoding="utf-8",
        )

        mlflow.log_artifact(str(artifact_path))

    run_id = mlflow.active_run().info.run_id

# Verify metadata persisted and artifact is accessible.
run = client.get_run(run_id)
artifacts = client.list_artifacts(run_id)

assert run.data.params["model"] == "synthetic_test"
assert run.data.metrics["validation_ap"] == 0.95
assert any(item.path == "smoke_test.txt" for item in artifacts)

print("MLflow smoke test passed!")
print("Tracking URI:", TRACKING_URI)
print("Experiment ID:", experiment_id)
print("Run ID:", run_id)
print("Parameters:", run.data.params)
print("Metrics:", run.data.metrics)
print("Artifacts:", [item.path for item in artifacts])
