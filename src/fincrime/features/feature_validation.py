import pandas as pd


def validate_model_features(
    features: pd.DataFrame,
    timestep_column: str = "timestep",
) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame.")

    if features.columns.duplicated().any():
        raise ValueError("Feature columns must be unique.")

    normalized_columns = {
        str(column).strip().lower()
        for column in features.columns
    }

    if timestep_column.strip().lower() in normalized_columns:
        raise ValueError(
            f"'{timestep_column}' must not be included in model features."
        )