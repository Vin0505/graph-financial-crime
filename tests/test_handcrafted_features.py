
import numpy as np
import pandas as pd
import pytest

from fincrime.features.handcrafted_features import (
    build_handcrafted_features,
)


@pytest.fixture
def toy_graph():
    """
    Synthetic directed graph:

        101 ---> 102 ---> 103
                  ^
                  |
                  |
                 104

    All nodes belong to timestep 1.
    """

    local_features = pd.DataFrame(
        {
            "f1": [10.0, 20.0, 30.0, 40.0],
            "f2": [1.0, 2.0, 3.0, 4.0],
        },
        index=pd.Index(
            [101, 102, 103, 104],
            name="txId",
        ),
    )

    edges = pd.DataFrame(
        {
            "txId1": [101, 102, 104],
            "txId2": [102, 103, 102],
        }
    )

    timesteps = pd.Series(
        [1, 1, 1, 1],
        index=local_features.index,
        name="timestep",
    )

    return local_features, edges, timesteps


def test_feature_shapes_and_alignment(toy_graph):
    local_features, edges, timesteps = toy_graph

    features, diagnostics = build_handcrafted_features(
        local_features,
        edges,
        timesteps,
    )

    # 2 degree features + 4 groups of 2 local features.
    assert features.shape == (4, 10)

    assert diagnostics.shape == (4, 2)

    assert features.index.equals(local_features.index)
    assert diagnostics.index.equals(local_features.index)

    assert features.columns.is_unique


def test_incoming_and_outgoing_degrees(toy_graph):
    local_features, edges, timesteps = toy_graph

    features, _ = build_handcrafted_features(
        local_features,
        edges,
        timesteps,
    )

    # Node 102 receives edges from 101 and 104.
    assert features.loc[102, "hand_in_degree"] == 2

    # Node 102 has one outgoing edge to 103.
    assert features.loc[102, "hand_out_degree"] == 1

    # Node 101 has no incoming edges.
    assert features.loc[101, "hand_in_degree"] == 0

    # Node 103 has no outgoing edges.
    assert features.loc[103, "hand_out_degree"] == 0


def test_one_hop_neighbour_means(toy_graph):
    local_features, edges, timesteps = toy_graph

    features, _ = build_handcrafted_features(
        local_features,
        edges,
        timesteps,
    )

    # Incoming neighbours of 102 are 101 and 104.
    # Mean f1 = (10 + 40) / 2 = 25.
    assert features.loc[102, "hand_in1_mean_f1"] == pytest.approx(25.0)

    # Outgoing neighbour of 102 is 103.
    assert features.loc[102, "hand_out1_mean_f1"] == pytest.approx(30.0)


def test_exactly_two_hop_neighbour_means(toy_graph):
    local_features, edges, timesteps = toy_graph

    features, diagnostics = build_handcrafted_features(
        local_features,
        edges,
        timesteps,
    )

    # 101 -> 102 -> 103
    # The exactly-2-hop outgoing neighbour of 101 is 103.
    assert features.loc[101, "hand_out2_mean_f1"] == pytest.approx(30.0)

    # 103 has exactly-2-hop incoming neighbours 101 and 104.
    # Mean f1 = (10 + 40) / 2 = 25.
    assert features.loc[103, "hand_in2_mean_f1"] == pytest.approx(25.0)

    assert diagnostics.loc[101, "out_degree_2hop_unique"] == 1
    assert diagnostics.loc[103, "in_degree_2hop_unique"] == 2


def test_zero_neighbour_means(toy_graph):
    local_features, edges, timesteps = toy_graph

    features, _ = build_handcrafted_features(
        local_features,
        edges,
        timesteps,
    )

    # Node 103 has no outgoing neighbours.
    assert features.loc[103, "hand_out1_mean_f1"] == 0.0

    # Node 101 has no incoming neighbours.
    assert features.loc[101, "hand_in1_mean_f1"] == 0.0


def test_cross_timestep_edges_are_rejected(toy_graph):
    local_features, edges, timesteps = toy_graph

    # Move node 103 to another timestep.
    timesteps.loc[103] = 2

    with pytest.raises(ValueError, match="Cross-timestep edges"):
        build_handcrafted_features(
            local_features,
            edges,
            timesteps,
        )


def test_missing_transaction_mapping_is_rejected(toy_graph):
    local_features, edges, timesteps = toy_graph

    # Add an edge referencing a transaction not in local_features.
    invalid_edges = pd.concat(
        [
            edges,
            pd.DataFrame(
                {
                    "txId1": [101],
                    "txId2": [999],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="absent from local_features"):
        build_handcrafted_features(
            local_features,
            invalid_edges,
            timesteps,
        )


def test_labels_are_not_part_of_feature_builder(toy_graph):
    # The function signature accepts only local features, edges and timesteps.
    import inspect

    parameters = inspect.signature(
        build_handcrafted_features
    ).parameters

    assert "labels" not in parameters
    assert "classes" not in parameters


def test_non_finite_local_features_are_rejected(toy_graph):
    local_features, edges, timesteps = toy_graph

    local_features.loc[101, "f1"] = np.inf

    with pytest.raises(ValueError, match="missing or infinite"):
        build_handcrafted_features(
            local_features,
            edges,
            timesteps,
        )
