"""Spatial extraction tested without a physics engine or an evo import."""

from dataclasses import replace

import numpy as np
import pytest

from spatial_neat import (
    ESConfig,
    ESDeveloper,
    ExpressionLimitError,
    NetworkDefinition,
    Node,
    Region,
    Substrate,
    coordinate_features,
)


class Pattern:
    def __init__(self, dimensions=3, width=3.0):
        self.dimensions, self.width = dimensions, width

    def query(self, values):
        # A symmetric localized band in the target and source coordinates.
        values = np.asarray(values)
        d = self.dimensions
        return np.exp(-self.width * np.sum(values[..., : 2 * d] ** 2, axis=-1))[..., None]


class Constant:
    def query(self, values):
        return np.array([0.5])


def anchors(dimensions=3, regions=None):
    position = (0.0,) * dimensions
    return Substrate(
        (Node("sensor", position, "input"),),
        (Node("action", position, "output"),),
        regions or (Region("core", (-1.0,) * dimensions, (1.0,) * dimensions),),
    )


def config(**options):
    return ESConfig(
        initial_depth=2,
        max_depth=2,
        division_threshold=0.001,
        variance_threshold=0.001,
        band_threshold=0.001,
        **options,
    )


@pytest.mark.parametrize("dimensions", [2, 3])
def test_discovery_is_deterministic_and_has_functional_hidden_nodes(dimensions):
    developer = ESDeveloper(config())
    first = developer.develop(Pattern(dimensions), anchors(dimensions))
    second = developer.develop(Pattern(dimensions), anchors(dimensions))
    assert first == second
    assert first.network.hidden_count > 0
    assert first.queries > 0
    for node in first.network.nodes:
        assert len(node.position) == dimensions
    hidden = {n.key for n in first.network.nodes if n.role == "hidden"}
    assert all(any(e.target == key for e in first.network.edges) for key in hidden)
    assert all(any(e.source == key for e in first.network.edges) for key in hidden)


def test_uniform_pattern_does_not_force_hidden_neurons():
    result = ESDeveloper(config()).develop(Constant(), anchors())
    assert result.network.hidden_count == 0
    assert result.network.edges == ()
    direct = ESDeveloper(config(direct_links=True)).develop(Constant(), anchors())
    assert [(e.source, e.target, e.weight) for e in direct.network.edges] == [
        ("sensor", "action", 0.5)
    ]


def test_pattern_changes_density_not_a_fixed_neuron_count():
    developer = ESDeveloper(config())
    empty = developer.develop(Constant(), anchors())
    detailed = developer.develop(Pattern(), anchors())
    assert detailed.network.hidden_count > empty.network.hidden_count
    assert detailed.queries > empty.queries


def test_adaptive_tree_has_multiple_depths_and_density_depends_on_pattern_width():
    adaptive = ESConfig(
        initial_depth=2,
        max_depth=3,
        division_threshold=0.001,
        variance_threshold=0.001,
        band_threshold=0.001,
        iterations=0,
        max_hidden_nodes=2048,
    )
    developer = ESDeveloper(adaptive)
    detailed = developer.develop(Pattern(width=3), anchors())
    depths = {len(n.key.rsplit(":", 1)[1]) for n in detailed.network.nodes if n.role == "hidden"}
    assert depths == {2, 3}  # not merely a uniform fixed grid with a new name
    broad = developer.develop(Pattern(width=1), anchors())
    narrow = developer.develop(Pattern(width=8), anchors())
    assert broad.network.hidden_count != narrow.network.hidden_count


def test_3d_features_use_both_z_coordinates():
    query = coordinate_features((0, 0, 0.25), (0, 0, -0.5))
    np.testing.assert_array_equal(query, [0, 0, 0.25, 0, 0, -0.5, 0.75, 1])


def test_actual_3d_discovery_responds_to_z_pattern():
    class ZPattern:
        def query(self, values):
            return np.array([np.exp(-4 * (values[2] ** 2 + values[5] ** 2))])

    result = ESDeveloper(config(iterations=0)).develop(ZPattern(), anchors())
    hidden = [n for n in result.network.nodes if n.role == "hidden"]
    assert hidden
    assert {p.position[2] for p in hidden} == {-0.75, 0.75}
    # With no z dependence the same constant xy slices discover nothing.
    assert ESDeveloper(config()).develop(Constant(), anchors()).network.hidden_count == 0


def test_recurrent_connections_are_real_not_an_upward_only_filter():
    recurrent_config = replace(config(), variance_threshold=0.0)
    network = ESDeveloper(recurrent_config).develop(Pattern(), anchors()).network
    hidden = {n.key for n in network.nodes if n.role == "hidden"}
    edges = {e.key for e in network.edges}
    assert any(a != b and (b, a) in edges for a, b in edges if a in hidden and b in hidden)
    assert any(a == b for a, b in edges)
    no_self = ESDeveloper(replace(recurrent_config, allow_self_connections=False)).develop(
        Pattern(), anchors()
    )
    assert all(e.source != e.target for e in no_self.network.edges)


def test_filter_and_reachability_remove_disconnected_hidden_candidates():
    developer = ESDeveloper(config(), connection_filter=lambda a, b: b.role != "output")
    result = developer.develop(Pattern(), anchors())
    assert result.candidates > 0
    assert result.network.hidden_count == 0
    assert result.network.edges == ()


def test_modular_regions_have_noncolliding_stable_ids():
    regions = (Region("left", (-1, -1, -1), (1, 1, 1)), Region("right", (-1, -1, -1), (1, 1, 1)))
    developer = ESDeveloper(config(iterations=0))
    core = developer.develop(Pattern(), anchors(regions=regions[:1])).network
    both = developer.develop(Pattern(), anchors(regions=regions)).network
    assert both.hidden_count == 2 * core.hidden_count
    assert {n.key for n in core.nodes} <= {n.key for n in both.nodes}


@pytest.mark.parametrize(
    "field, budget",
    [
        ("max_queries", 2),
        ("max_hidden_nodes", 1),
        ("max_edges", 1),
    ],
)
def test_limits_raise_without_silent_truncation_and_developer_can_be_reused(field, budget):
    developer = ESDeveloper(replace(config(), **{field: budget}))
    with pytest.raises(ExpressionLimitError):
        developer.develop(Pattern(), anchors())
    with pytest.raises(ExpressionLimitError):
        developer.develop(Pattern(), anchors())  # no retained partial search state
    normal = ESDeveloper(config()).develop(Pattern(), anchors())
    assert normal.network.hidden_count


@pytest.mark.parametrize("query", [np.array([np.nan]), np.zeros((1, 1)), np.array([])])
def test_invalid_cppn_outputs_rejected(query):
    class Invalid:
        def query(self, values):
            return query

    with pytest.raises(ValueError):
        ESDeveloper(config()).develop(Invalid(), anchors())


@pytest.mark.parametrize(
    "options",
    [
        {"max_depth": 0},
        {"initial_depth": 3, "max_depth": 2},
        {"iterations": -1},
        {"max_queries": 0},
        {"weight_scale": float("nan")},
        {"max_depth": 2.5},
        {"variance_threshold": -1},
    ],
)
def test_invalid_config_rejected(options):
    with pytest.raises(ValueError):
        ESConfig(**options)


def test_network_json_roundtrip():
    result = ESDeveloper(config()).develop(Pattern(), anchors())
    assert NetworkDefinition.from_dict(result.network.to_dict()) == result.network
