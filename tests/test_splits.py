
from pathlib import Path

import pandas as pd
import pytest

from fincrime.evaluation.splits import (
    load_split_config,
    build_split_masks,
)


@pytest.fixture
def split_config():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "splits.yaml"

    return load_split_config(config_path)


@pytest.fixture
def timesteps():
    return pd.Series(
        range(1, 50),
        index=range(49),
        name="timestep",
    )


def test_tuning_split_boundaries(split_config, timesteps):
    masks = build_split_masks(timesteps, split_config, "tuning")

    assert masks["fit"].sum() == 24
    assert masks["validation"].sum() == 5
    assert masks["test"] is None


def test_track_a_split_boundaries(split_config, timesteps):
    masks = build_split_masks(timesteps, split_config, "track_a")

    assert masks["fit"].sum() == 30
    assert masks["validation"].sum() == 4
    assert masks["test"].sum() == 15


@pytest.mark.parametrize(
    "cutoff, expected_fit, expected_validation, expected_test",
    [
        ("cutoff_34", 30, 4, 4),
        ("cutoff_38", 34, 4, 4),
        ("cutoff_42", 38, 4, 4),
        ("cutoff_46", 42, 4, 3),
    ],
)
def test_track_b_split_boundaries(
    split_config,
    timesteps,
    cutoff,
    expected_fit,
    expected_validation,
    expected_test,
):
    masks = build_split_masks(
        timesteps,
        split_config,
        "track_b",
        track_b_name=cutoff,
    )

    assert masks["fit"].sum() == expected_fit
    assert masks["validation"].sum() == expected_validation
    assert masks["test"].sum() == expected_test


def test_track_a_and_cutoff_34_match(split_config, timesteps):
    track_a = build_split_masks(timesteps, split_config, "track_a")

    cutoff_34 = build_split_masks(
        timesteps,
        split_config,
        "track_b",
        track_b_name="cutoff_34",
    )

    assert track_a["fit"].equals(cutoff_34["fit"])
    assert track_a["validation"].equals(cutoff_34["validation"])
