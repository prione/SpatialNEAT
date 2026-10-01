import copy
import json

import numpy as np
import pytest

from spatial_neat import Edge, GrowingRNN, NetworkDefinition, Node


def network(extra=False):
    nodes = [
        Node("sensor", (0, 0, -1), "input"),
        Node("memory", (0, 0, 0), "hidden", 0.3),
        Node("action", (0, 0, 1), "output"),
    ]
    edges = [
        Edge("sensor", "memory", 0.5),
        Edge("memory", "memory", 0.4),
        Edge("memory", "action", 0.5),
    ]
    if extra:
        nodes.append(Node("new", (0.25, 0, 0), "hidden", 0.2))
        edges.extend(
            [
                Edge("sensor", "new", 0.3),
                Edge("new", "memory", 1.0),
                Edge("memory", "new", 0.2),
                Edge("new", "action", 0.3),
            ]
        )
    return NetworkDefinition(tuple(nodes), tuple(edges))


def test_synchronous_recurrence_and_reset():
    brain = GrowingRNN(network())
    assert brain.step([1])["action"] == 0  # output sees previous hidden state
    assert brain.state["memory"] == pytest.approx(0.3 * np.tanh(0.5))
    assert brain.step([0])["action"] > 0
    assert brain.step([0])["action"] > 0
    brain.reset()
    assert brain.step([0])["action"] == 0


def test_growth_preserves_state_base_learning_and_first_step_behavior():
    brain = GrowingRNN(network(), maturation_steps=4)
    brain.step([1])
    brain.set_delta("memory", "action", 0.1)
    reference = GrowingRNN.from_snapshot(brain.snapshot())
    state, base, learned = brain.state, brain.base_weights, brain.learned_deltas
    report = brain.grow(network(extra=True))
    assert report.added_nodes == ("new",)
    assert len(report.added_edges) == 4
    for key, value in state.items():
        assert brain.state[key] == value
    for key, value in base.items():
        assert brain.base_weights[key] == value
        assert brain.learned_deltas[key] == learned[key]
    assert brain.state["new"] == 0
    assert brain.step([1])["action"] == reference.step([1])["action"]
    assert brain.gates["new", "memory"] == 0.25
    for _ in range(3):
        brain.step([0])
    assert all(g == 1 for g in brain.gates.values())
    assert brain.grow(network(extra=True)).added_nodes == ()


def test_reexpression_omissions_and_changed_weights_do_not_erase_old_network():
    brain = GrowingRNN(network())
    candidate = NetworkDefinition(network().nodes, (Edge("sensor", "memory", 100),))
    brain.grow(candidate)
    assert brain.base_weights["sensor", "memory"] == 0.5
    assert len(brain.network.edges) == 3


def test_growth_failure_is_transactional():
    brain = GrowingRNN(network(), max_nodes=3)
    before = brain.snapshot()
    with pytest.raises(ValueError, match="budget"):
        brain.grow(network(extra=True))
    assert brain.snapshot() == before
    altered = list(network().nodes)
    altered[1] = Node("memory", (0.1, 0, 0), "hidden", 0.3)
    with pytest.raises(ValueError, match="changed"):
        brain.grow(NetworkDefinition(tuple(altered), network().edges))
    assert brain.snapshot() == before


def test_recurrent_budget_preserves_old_effective_weights_on_growth():
    brain = GrowingRNN(network())
    brain.set_delta("memory", "memory", 0.4)
    brain.grow(network(extra=True))
    assert brain.base_weights["memory", "memory"] == 0.4
    assert brain.learned_deltas["memory", "memory"] == 0.4
    assert brain.base_weights["new", "memory"] == pytest.approx(0.18)
    accepted = brain.set_delta("new", "memory", 100)
    assert accepted == pytest.approx(0)
    assert brain.set_delta("sensor", "memory", 2) == 2  # not a recurrent edge


def test_initial_recurrent_row_normalization():
    definition = NetworkDefinition(
        network().nodes, (Edge("memory", "memory", 10), Edge("action", "memory", -10))
    )
    brain = GrowingRNN(definition)
    assert sum(map(abs, brain.base_weights.values())) == pytest.approx(0.98)


def test_json_checkpoint_exact_continuation_including_maturity_and_learning():
    brain = GrowingRNN(network())
    brain.step([1])
    brain.set_delta("memory", "action", 0.05)
    brain.grow(network(extra=True))
    brain.step([0.5])
    saved = json.loads(json.dumps(brain.snapshot(), allow_nan=False))
    restored = GrowingRNN.from_snapshot(saved)
    assert restored.snapshot() == saved
    for value in (0.2, 0.3, -0.2, 0.1):
        assert brain.step([value]) == restored.step([value])
    assert brain.snapshot() == restored.snapshot()


@pytest.mark.parametrize(
    "alter",
    [
        lambda d: d["state"].update(memory=float("nan")),
        lambda d: d["state"].update(memory=2),
        lambda d: d["weights"][0].update(gate=2),
        lambda d: d["weights"][0].update(delta=float("inf")),
        lambda d: d["weights"].append(d["weights"][0]),
        lambda d: d["weights"][1].update(base=5),
        lambda d: d.update(schema_version=2),
    ],
)
def test_invalid_snapshots_rejected(alter):
    data = copy.deepcopy(GrowingRNN(network()).snapshot())
    alter(data)
    with pytest.raises(ValueError):
        GrowingRNN.from_snapshot(data)


@pytest.mark.parametrize("inputs", [[float("nan")], [], {"wrong": 1}, [1, 2]])
def test_invalid_inputs_do_not_advance_state(inputs):
    brain = GrowingRNN(network())
    before = brain.snapshot()
    with pytest.raises(ValueError):
        brain.step(inputs)
    assert brain.snapshot() == before


def test_sparse_accumulation_overflow_does_not_advance_state():
    definition = NetworkDefinition(network().nodes, (Edge("sensor", "memory", 1e308),))
    brain = GrowingRNN(definition)
    before = brain.snapshot()
    with pytest.raises(ValueError, match="accumulation"):
        brain.step([1e308])
    assert brain.snapshot() == before


def test_state_accessors_do_not_allow_mutating_runtime():
    brain = GrowingRNN(network())
    brain.state["memory"] = 1
    brain.base_weights["sensor", "memory"] = 10
    assert brain.state["memory"] == 0
    assert brain.base_weights["sensor", "memory"] == 0.5


def test_optional_stability_and_immediate_maturation():
    brain = GrowingRNN(network(), recurrent_limit=None, maturation_steps=0)
    brain.grow(network(extra=True))
    assert brain.base_weights["new", "memory"] == 1
    assert all(g == 1 for g in brain.gates.values())
    assert brain.set_delta("memory", "memory", 10) == 10
