import pandas as pd
import pytest

from fincrime.evaluation.splits import build_label_masks


@pytest.fixture
def labels():
    return pd.Series(
        [
            "1", "2", "unknown", "1", "unknown",
            "2", "1", "2", "unknown", "1",
        ],
        index=range(10),
        name="class",
    )


@pytest.fixture
def split_masks():
    return {
        "fit": pd.Series(
            [True, True, True, True, False,
             False, False, False, False, False]
        ),
        "validation": pd.Series(
            [False, False, False, False, True,
             True, False, False, False, False]
        ),
        "test": pd.Series(
            [False, False, False, False, False,
             False, True, True, True, True]
        ),
    }


@pytest.fixture
def label_config():
    return {
        "illicit": "1",
        "licit": "2",
        "unknown": "unknown",
    }


def test_unknown_labels_excluded_from_masks(
    labels, split_masks, label_config
):
    masks = build_label_masks(labels, split_masks, label_config)

    assert masks["train_loss"].tolist() == [
        True, True, False, True, False,
        False, False, False, False, False,
    ]

    assert masks["validation_eval"].tolist() == [
        False, False, False, False, False,
        True, False, False, False, False,
    ]

    assert masks["test_eval"].tolist() == [
        False, False, False, False, False,
        False, True, True, False, True,
    ]


def test_unknown_nodes_are_not_removed_from_graph(
    labels, split_masks, label_config
):
    masks = build_label_masks(labels, split_masks, label_config)

    assert len(labels) == 10
    assert len(masks["train_loss"]) == 10


def test_unexpected_labels_are_rejected(
    labels, split_masks, label_config
):
    labels.iloc[0] = "3"

    with pytest.raises(ValueError, match="Unexpected labels"):
        build_label_masks(labels, split_masks, label_config)


def test_missing_labels_are_rejected(
    labels, split_masks, label_config
):
    labels.iloc[0] = None

    with pytest.raises(ValueError, match="missing values"):
        build_label_masks(labels, split_masks, label_config)


def test_index_mismatch_is_rejected(
    labels, split_masks, label_config
):
    split_masks["fit"].index = range(1, 11)

    with pytest.raises(ValueError, match="identical indices"):
        build_label_masks(labels, split_masks, label_config)