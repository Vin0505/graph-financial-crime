import pandas as pd
import pytest

from fincrime.features.feature_validation import validate_model_features


def test_valid_features_are_accepted():
    features = pd.DataFrame(
        {
            "local_0": [0.1, 0.2],
            "local_1": [0.3, 0.4],
        }
    )

    assert validate_model_features(features) is None


def test_timestep_column_is_rejected():
    features = pd.DataFrame(
        {
            "local_0": [0.1, 0.2],
            "timestep": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="must not be included"):
        validate_model_features(features)


def test_timestep_check_is_case_insensitive():
    features = pd.DataFrame(
        {
            "local_0": [0.1, 0.2],
            "TimeStep": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="must not be included"):
        validate_model_features(features)


def test_duplicate_feature_columns_are_rejected():
    features = pd.DataFrame(
        [[0.1, 0.2]],
        columns=["local_0", "local_0"],
    )

    with pytest.raises(ValueError, match="must be unique"):
        validate_model_features(features)


def test_non_dataframe_input_is_rejected():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_model_features([[0.1, 0.2]])