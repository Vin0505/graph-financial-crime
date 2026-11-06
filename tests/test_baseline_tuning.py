import pytest

from fincrime.baselines.tuning import (
    load_xgboost_tuning_config,
    resolve_xgboost_trial,
)


@pytest.fixture
def tuning_config(tmp_path):
    config_path = tmp_path / "xgboost.yaml"

    trials = []

    for index, depth in enumerate([3, 5, 7, 3, 5, 5], start=1):
        trials.append({
            "name": f"trial_{index:02d}",
            "max_depth": depth,
            "learning_rate": 0.05,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "scale_pos_weight": (
                "train_ratio" if index >= 4 else 1.0
            ),
        })

    import yaml

    config = {
        "model": {
            "name": "xgboost",
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1,
        },
        "tuning": {
            "selection_metric": "average_precision",
            "trials": trials,
        },
    }

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    return config_path


def test_load_xgboost_tuning_config(tuning_config):
    config = load_xgboost_tuning_config(tuning_config)

    assert config["model"]["name"] == "xgboost"
    assert config["tuning"]["selection_metric"] == "average_precision"
    assert len(config["tuning"]["trials"]) == 6


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_xgboost_tuning_config("missing_xgboost.yaml")


def test_rejects_wrong_model_name(tuning_config):
    import yaml

    config = yaml.safe_load(tuning_config.read_text())
    config["model"]["name"] = "random_forest"

    tuning_config.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="xgboost"):
        load_xgboost_tuning_config(tuning_config)


def test_rejects_duplicate_trial_names(tuning_config):
    import yaml

    config = yaml.safe_load(tuning_config.read_text())
    config["tuning"]["trials"][1]["name"] = (
        config["tuning"]["trials"][0]["name"]
    )

    tuning_config.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="unique"):
        load_xgboost_tuning_config(tuning_config)


def test_rejects_wrong_number_of_trials(tuning_config):
    import yaml

    config = yaml.safe_load(tuning_config.read_text())
    config["tuning"]["trials"].pop()

    tuning_config.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="six"):
        load_xgboost_tuning_config(tuning_config)


def test_resolves_train_ratio_using_training_labels(tuning_config):
    config = load_xgboost_tuning_config(tuning_config)
    trial = config["tuning"]["trials"][3]

    # 8 negative samples and 2 positive samples.
    y_train = [0] * 8 + [1] * 2

    resolved = resolve_xgboost_trial(
        config=config,
        trial=trial,
        y_train=y_train,
    )

    assert resolved["scale_pos_weight"] == 4.0
    assert resolved["positive_training_count"] == 2
    assert resolved["negative_training_count"] == 8


def test_numeric_scale_pos_weight_is_preserved(tuning_config):
    config = load_xgboost_tuning_config(tuning_config)
    trial = config["tuning"]["trials"][0]

    resolved = resolve_xgboost_trial(
        config=config,
        trial=trial,
        y_train=[0, 0, 0, 1],
    )

    assert resolved["scale_pos_weight"] == 1.0


def test_model_and_trial_parameters_are_merged(tuning_config):
    config = load_xgboost_tuning_config(tuning_config)
    trial = config["tuning"]["trials"][0]

    resolved = resolve_xgboost_trial(
        config=config,
        trial=trial,
        y_train=[0, 0, 1],
    )

    params = resolved["model_params"]

    assert params["n_estimators"] == 300
    assert params["max_depth"] == 3
    assert params["learning_rate"] == 0.05
    assert params["objective"] == "binary:logistic"
    assert params["eval_metric"] == "aucpr"

    assert "scale_pos_weight" not in params
    assert "random_state" not in params
    assert "name" not in params


@pytest.mark.parametrize(
    "invalid_labels",
    [
        [],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 2],
        [0, 1, None],
    ],
)
def test_rejects_invalid_training_labels(
    tuning_config,
    invalid_labels,
):
    config = load_xgboost_tuning_config(tuning_config)
    trial = config["tuning"]["trials"][0]

    with pytest.raises(ValueError):
        resolve_xgboost_trial(
            config=config,
            trial=trial,
            y_train=invalid_labels,
        )
