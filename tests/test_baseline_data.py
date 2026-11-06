import pandas as pd
import pytest

from fincrime.baselines.data import prepare_tabular_data


@pytest.fixture
def split_config():
    return {
        "labels": {
            "illicit": "1",
            "licit": "2",
            "unknown": "unknown",
        },
        "tuning": {
            "fit": {
                "start": 1,
                "end": 24,
            },
            "validation": {
                "start": 25,
                "end": 29,
            },
        }
    }


@pytest.fixture
def sample_data():
    index = [101, 102, 103, 104, 105, 106]

    features = pd.DataFrame(
        {
            "local_0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "local_1": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        },
        index=index,
    )

    labels = pd.Series(
        ["1", "2", "unknown", "1", "2", "unknown"],
        index=index,
    )

    timesteps = pd.Series(
        [1, 1, 24, 25, 25, 29],
        index=index,
    )

    return features, labels, timesteps


def test_unknown_labels_are_excluded(
    split_config,
    sample_data,
):
    features, labels, timesteps = sample_data

    result = prepare_tabular_data(
        features,
        labels,
        timesteps,
        split_config,
    )

    assert list(result["X_train"].index) == [101, 102]
    assert list(result["X_validation"].index) == [104, 105]


def test_binary_label_mapping(
    split_config,
    sample_data,
):
    features, labels, timesteps = sample_data

    result = prepare_tabular_data(
        features,
        labels,
        timesteps,
        split_config,
    )

    assert result["y_train"].tolist() == [1, 0]
    assert result["y_validation"].tolist() == [1, 0]


def test_misaligned_indices_raise_error(
    split_config,
    sample_data,
):
    features, labels, timesteps = sample_data

    labels = labels.iloc[::-1]

    with pytest.raises(ValueError, match="indices do not match"):
        prepare_tabular_data(
            features,
            labels,
            timesteps,
            split_config,
        )


def test_missing_features_raise_error(
    split_config,
    sample_data,
):
    features, labels, timesteps = sample_data
    features.iloc[0, 0] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        prepare_tabular_data(
            features,
            labels,
            timesteps,
            split_config,
        )