import pytest
import yaml

from fincrime.gnn.tuning import (
    load_gnn_tuning_config,
    resolve_gnn_trial,
)


def make_config(model_name="gcn"):
    return {
        "model": {
            "name": model_name,
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 200,
            "patience": 20,
            "min_delta": 0.0,
            "class_weighting": "balanced",
        },
        "trials": [
            {
                "name": f"hidden{hidden}_dropout{dropout}",
                "hidden_channels": hidden,
                "dropout": dropout,
            }
            for hidden in (32, 64)
            for dropout in (0.3, 0.4, 0.5)
        ],
    }


def write_config(tmp_path, config):
    path = tmp_path / "gnn.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_valid_gcn_config(tmp_path):
    path = write_config(tmp_path, make_config("gcn"))

    config = load_gnn_tuning_config(path, expected_model="gcn")

    assert config["model"]["name"] == "gcn"
    assert len(config["tuning"]["trials"]) == 6


def test_load_valid_graphsage_config(tmp_path):
    path = write_config(tmp_path, make_config("graphsage"))

    config = load_gnn_tuning_config(
        path,
        expected_model="graphsage",
    )

    assert config["model"]["name"] == "graphsage"


def test_reject_wrong_model(tmp_path):
    path = write_config(tmp_path, make_config("graphsage"))

    with pytest.raises(ValueError):
        load_gnn_tuning_config(path, expected_model="gcn")


def test_reject_duplicate_trial_names(tmp_path):
    config = make_config()
    config["trials"][1]["name"] = config["trials"][0]["name"]
    path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="unique"):
        load_gnn_tuning_config(path)


def test_reject_missing_trial_combination(tmp_path):
    config = make_config()
    config["trials"][0]["dropout"] = 0.4
    path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="cover every combination"):
        load_gnn_tuning_config(path)


def test_reject_invalid_hidden_channels(tmp_path):
    config = make_config()
    config["trials"][0]["hidden_channels"] = 128
    path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="hidden_channels"):
        load_gnn_tuning_config(path)


def test_reject_invalid_dropout(tmp_path):
    config = make_config()
    config["trials"][0]["dropout"] = 0.2
    path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="dropout"):
        load_gnn_tuning_config(path)


def test_resolve_trial():
    config = make_config("gcn")
    trial = config["trials"][0]

    resolved = resolve_gnn_trial(config, trial)

    assert resolved["model_name"] == "gcn"
    assert resolved["hidden_channels"] == 32
    assert resolved["dropout"] == 0.3
    assert resolved["learning_rate"] == 0.001
    assert resolved["max_epochs"] == 200