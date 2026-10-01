"""Independent equation checks and cache invalidation at every public boundary."""

from copy import deepcopy

import numpy as np
import pytest

from spatial_neat import Development, Edge, GrowingRNN, Maturation, NetworkDefinition, Node


def reference_step(snapshot, inputs):
    """The old synchronous dictionary/add.at equation, independent of runtime caches."""
    nodes = snapshot["network"]["nodes"]
    dynamic = [n for n in nodes if n["role"] != "input"]
    targets = {n["key"]: i for i, n in enumerate(dynamic)}
    values = dict(zip((n["key"] for n in nodes if n["role"] == "input"), inputs))
    values.update(snapshot["state"])
    total = np.zeros(len(dynamic))
    indices, products = [], []
    for weight in snapshot["weights"]:
        indices.append(targets[weight["target"]])
        products.append(
            values[weight["source"]] * ((weight["base"] + weight["delta"]) * weight["gate"])
        )
    np.add.at(total, np.asarray(indices, dtype=np.intp), products)
    snapshot["state"] = {
        n["key"]: float(
            snapshot["state"][n["key"]]
            + n["leak"] * (np.tanh(total[i]) - snapshot["state"][n["key"]])
        )
        for i, n in enumerate(dynamic)
    }
    gates = {(w["source"], w["target"]): w["gate"] for w in snapshot["weights"]}
    gates = Maturation(snapshot["options"]["maturation_steps"]).advance(gates)
    for w in snapshot["weights"]:
        w["gate"] = gates[w["source"], w["target"]]
    return {n["key"]: snapshot["state"][n["key"]] for n in nodes if n["role"] == "output"}


def small():
    return NetworkDefinition(
        (
            Node("out", (0, 1), "output"),
            Node("in", (0, 0), "input"),
            Node("h", (1, 0), "hidden", leak=0.3),
        ),
        (Edge("h", "out", 0.4), Edge("in", "h", 0.5), Edge("h", "h", 0.2)),
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("maturation", [0, 1, 7])
@pytest.mark.parametrize("limit", [None, 0.98])
def test_randomized_arrays_match_dictionary_equation(seed, maturation, limit):
    rng = np.random.default_rng(seed)
    dimensions = 2 + seed % 2
    nodes = [
        Node(f"n{i}", tuple(rng.uniform(-1, 1, dimensions)), role, leak=float(rng.uniform(0.01, 1)))
        for i, role in enumerate(("output", "input", "hidden", "output", "hidden", "input"))
    ]
    rng.shuffle(nodes)
    edges = [
        Edge(a.key, b.key, float(rng.uniform(-1, 1)))
        for a in nodes
        for b in nodes
        if b.role != "input" and rng.random() < 0.65
    ]
    rng.shuffle(edges)
    brain = GrowingRNN(
        NetworkDefinition(tuple(nodes), tuple(edges)),
        recurrent_limit=limit,
        maturation_steps=maturation,
    )
    for edge in edges:
        brain.set_delta(*edge.key, float(rng.uniform(-0.1, 0.1)))
    snapshot = brain.snapshot()
    snapshot["state"] = {key: float(rng.uniform(-0.8, 0.8)) for key in brain.state}
    for w in snapshot["weights"]:
        w["gate"] = float(rng.uniform(0, 1))
    brain.restore(snapshot)
    reference = deepcopy(snapshot)
    for i in range(32):
        values = rng.uniform(-1, 1, len(brain.input_keys))
        expected = reference_step(reference, values)
        outputs = brain.step(dict(zip(brain.input_keys, values)) if i % 2 else values)
        np.testing.assert_allclose(list(outputs.values()), list(expected.values()), atol=1e-14)
        np.testing.assert_allclose(
            list(brain.state.values()), list(reference["state"].values()), atol=1e-14
        )
        np.testing.assert_allclose(
            list(brain.gates.values()), [w["gate"] for w in reference["weights"]], atol=1e-14
        )


def test_step_never_queries_edge_keys_or_public_weight_mappings(monkeypatch):
    brain = GrowingRNN(small())

    def fail(*args):
        pytest.fail("hot path must not serialize or look up edge keys")

    monkeypatch.setattr(Edge, "key", property(fail))
    monkeypatch.setattr(GrowingRNN, "snapshot", fail)
    for field in ("base_weights", "learned_deltas", "gates"):
        monkeypatch.setattr(GrowingRNN, field, property(fail))
    for _ in range(12):
        assert np.isfinite(list(brain.step([0.5]).values())).all()


def test_restoration_handles_shuffled_snapshot_records_and_rebuilds_weights():
    brain = GrowingRNN(small())
    brain.step([0.5])
    snapshot = brain.snapshot()
    snapshot["weights"].reverse()
    snapshot["state"] = dict(reversed(tuple(snapshot["state"].items())))
    snapshot["weights"][0]["delta"] = 0.1
    snapshot["weights"][0]["gate"] = 0.25
    brain.restore(snapshot)
    reference = brain.snapshot()  # canonical edge order, regardless of record order
    expected = reference_step(reference, [0.7])
    assert brain.step([0.7]) == expected
    clone = GrowingRNN.from_snapshot(brain.snapshot())
    brain.set_delta("in", "h", 1)
    assert clone.learned_deltas["in", "h"] == 0.0
    assert clone.learned_deltas["h", "h"] == 0.1


def test_maturation_and_offset_caches_refresh_on_growth_reset_and_rollback():
    brain = GrowingRNN(small(), maturation_steps=4)
    brain.step([0.5])
    before = brain.snapshot()
    change = Development(brain).grow(
        NetworkDefinition(brain.network.nodes, (Edge("in", "out", 1),))
    )
    brain.set_delta("in", "out", 0.4)
    brain.mature(2)
    reference = brain.snapshot()
    assert brain.step([0.7]) == reference_step(reference, [0.7])
    brain.mature(2)
    assert brain.gates["in", "out"] == 1
    brain.reset()
    reference = brain.snapshot()
    assert brain.step([0.7]) == reference_step(reference, [0.7])
    change.rollback()
    assert brain.snapshot() == before
    assert brain.step([0.7]) == reference_step(deepcopy(before), [0.7])


@pytest.mark.parametrize("ticks", [-1, 0.5, True])
def test_invalid_maturation_is_atomic_even_when_all_edges_are_mature(ticks):
    brain = GrowingRNN(small())
    before = brain.snapshot()
    with pytest.raises(ValueError):
        brain.mature(ticks)
    assert brain.snapshot() == before


def test_overflow_does_not_mature_new_connections_or_poison_next_valid_step():
    brain = GrowingRNN(NetworkDefinition(small().nodes, ()), recurrent_limit=None)
    Development(brain).grow(NetworkDefinition(brain.network.nodes, (Edge("in", "h", 1e308),)))
    brain.mature(1)
    before = brain.snapshot()
    with pytest.raises(ValueError, match="accumulation"):
        brain.step([1e308])
    assert brain.snapshot() == before
    assert brain.step([0]) == reference_step(deepcopy(before), [0])


def test_accessors_and_snapshots_are_detached_from_arrays():
    brain = GrowingRNN(small())
    before = brain.snapshot()
    for mapping in (brain.state, brain.base_weights, brain.learned_deltas, brain.gates):
        for key in mapping:
            mapping[key] = 99
    detached = brain.snapshot()
    detached["weights"][0]["gate"] = 0
    detached["state"]["h"] = 1
    assert brain.snapshot() == before


def test_empty_edges_and_snapshot_with_zero_maturation_partial_gate():
    brain = GrowingRNN(NetworkDefinition(small().nodes, ()), maturation_steps=0)
    assert brain.step([0.5]) == {"out": 0}
    brain = GrowingRNN(small(), maturation_steps=0)
    snapshot = brain.snapshot()
    for w in snapshot["weights"]:
        w["gate"] = 0.25
    brain.restore(snapshot)
    assert brain.step([0.5]) == reference_step(snapshot, [0.5])
    assert all(v == 1 for v in brain.gates.values())
