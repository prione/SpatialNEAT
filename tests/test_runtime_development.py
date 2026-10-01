"""Runtime development operations, migration semantics and transaction boundaries."""

import json
from dataclasses import replace

import numpy as np
import pytest

from growth_hyperneat import (
    Development,
    Edge,
    GrowingRNN,
    Maturation,
    MigrationPolicy,
    NetworkDefinition,
    Node,
    StateTransfer,
    WeightTransfer,
    topology_health,
)


def definition():
    return NetworkDefinition(
        (
            Node("i", (0, 0, -1), "input"),
            Node("a", (-0.25, 0, 0), "hidden", 0.3),
            Node("b", (0.25, 0, 0), "hidden", 0.3),
            Node("o", (0, 0, 1), "output"),
        ),
        (
            Edge("i", "a", 0.4),
            Edge("i", "b", 0.8),
            Edge("a", "a", 0.1),
            Edge("b", "a", 0.2),
            Edge("a", "b", 0.3),
            Edge("b", "b", 0.2),
            Edge("a", "o", 0.2),
            Edge("b", "o", 0.3),
        ),
    )


def seeded():
    brain = GrowingRNN(definition())
    brain.step([0.8])
    brain.step([0.5])
    brain.set_delta("i", "a", 0.05)
    brain.set_delta("b", "o", 0.1)
    return brain, Development(brain)


def test_growth_maturation_and_pruning_are_separate_operations():
    brain, development = seeded()
    old = brain.snapshot()
    candidate = NetworkDefinition(
        definition().nodes + (Node("new", (0, 0, 0), "hidden"),),
        definition().edges + (Edge("i", "new", 0.5),),
    )
    change = development.grow(candidate)
    assert change.report.added_nodes == ("new",)
    assert brain.gates["i", "new"] == 0
    state = brain.state
    development.mature(ticks=2)
    assert brain.gates["i", "new"] == 0.25
    assert brain.state == state
    change.rollback()  # explicitly rewinds maturity as well
    assert brain.snapshot() == old


def test_pruning_removes_incident_edges_and_only_corresponding_state_learning():
    brain, development = seeded()
    old = brain.snapshot()
    change = development.prune(nodes=("b",))
    assert change.report.removed_nodes == ("b",)
    assert set(brain.state) == {"a", "o"}
    assert brain.state["a"] == old["state"]["a"]
    assert brain.learned_deltas["i", "a"] == 0.05
    assert all("b" not in key for key in brain.base_weights)
    assert change.report.health.unreachable_outputs == ()
    brain.step([0.2])
    change.rollback()  # restores pre-prune temporal/learning state, not just topology
    assert brain.snapshot() == old
    with pytest.raises(RuntimeError):
        change.rollback()


def test_edge_only_prune_keeps_neurons_and_reports_inactive_nodes():
    brain, development = seeded()
    change = development.prune(edges=(("a", "o"), ("b", "o")))
    assert brain.network.nodes == definition().nodes
    assert change.report.health.unreachable_outputs == ("o",)
    assert set(change.report.health.inactive_hidden) == {"a", "b"}


def test_optional_output_path_guard_rejects_disconnection_without_mutation():
    brain, development = seeded()
    before = brain.snapshot()
    with pytest.raises(ValueError, match="disconnects"):
        development.prune(
            edges=(("a", "o"), ("b", "o")), policy=MigrationPolicy(require_output_paths=True)
        )
    assert brain.snapshot() == before


@pytest.mark.parametrize("nodes, edges", [(("missing",), ()), ((), (("i", "o"),))])
def test_unknown_pruning_target_is_not_silently_ignored(nodes, edges):
    brain, development = seeded()
    before = brain.snapshot()
    with pytest.raises(ValueError, match="does not exist"):
        development.prune(nodes, edges)
    assert brain.snapshot() == before


def test_move_with_preserved_weights_preserves_computation_exactly():
    brain, development = seeded()
    reference = GrowingRNN.from_snapshot(brain.snapshot())
    change = development.move({"a": (0.75, 0.5, -0.5)})
    assert change.report.moved_nodes == ("a",)
    assert brain.state == reference.state
    assert brain.base_weights == reference.base_weights
    assert brain.learned_deltas == reference.learned_deltas
    for value in [0.1, 0.2, 0.4, -0.2]:
        assert brain.step([value]) == reference.step([value])


def test_move_with_cppn_changes_only_incident_base_weights_and_retains_learning():
    class Coordinates:
        def query(self, query):
            return np.array([0.05 * query[2] + 0.1 * query[5]])

    brain, development = seeded()
    old_base, old_delta, old_state = brain.base_weights, brain.learned_deltas, brain.state
    change = development.move({"a": (0, 0, 0.5)}, cppn=Coordinates())
    assert set(change.report.reexpressed_edges) == {key for key in old_base if "a" in key}
    assert brain.base_weights["i", "a"] == pytest.approx(0.0)
    assert brain.base_weights["a", "o"] == pytest.approx(0.125)
    assert brain.base_weights["b", "o"] == old_base["b", "o"]
    assert brain.learned_deltas == old_delta
    assert brain.state == old_state


@pytest.mark.parametrize(
    "positions", [{"missing": (0, 0, 0)}, {"a": (0, 0)}, {"a": (float("nan"), 0, 0)}]
)
def test_move_failures_leave_runtime_unchanged(positions):
    brain, development = seeded()
    before = brain.snapshot()
    with pytest.raises(ValueError):
        development.move(positions)
    assert brain.snapshot() == before


def test_invalid_cppn_during_move_does_not_commit_partial_updates():
    class Bad:
        def query(self, query):
            return np.array([np.nan])

    brain, development = seeded()
    before = brain.snapshot()
    with pytest.raises(ValueError):
        development.move({"a": (0, 0, 0)}, cppn=Bad())
    assert brain.snapshot() == before


def test_preview_evaluation_is_isolated_and_commit_does_not_adopt_trial_time():
    brain, development = seeded()
    before = brain.snapshot()
    candidate = NetworkDefinition(
        tuple(replace(n, position=(0.5, 0, 0)) if n.key == "a" else n for n in brain.network.nodes),
        brain.network.edges,
    )
    prepared = development.preview(candidate)
    trial = prepared.trial()
    trial.step([0.5])
    trial.set_delta("i", "a", 0.2)
    assert brain.snapshot() == before
    prepared.after_snapshot["state"]["a"] = 1  # public snapshot accessor is a copy
    change = development.commit(prepared)
    assert brain.state == before["state"]
    assert brain.learned_deltas["i", "a"] == 0.05
    change.rollback()
    assert brain.snapshot() == before


@pytest.mark.parametrize(
    "advance",
    [
        lambda b: b.step([0.5]),
        lambda b: b.set_delta("i", "a", 0.2),
        lambda b: b.restore(b.snapshot()),
    ],
)
def test_stale_preview_cannot_overwrite_new_activity_learning_or_restoration(advance):
    brain, development = seeded()
    prepared = development.preview(brain.network)
    advance(brain)
    before = brain.snapshot()
    with pytest.raises(RuntimeError):
        development.commit(prepared)
    assert brain.snapshot() == before


def test_preview_cannot_be_committed_into_another_identical_runtime():
    brain, development = seeded()
    prepared = development.preview(brain.network)
    other = Development(GrowingRNN.from_snapshot(brain.snapshot()))
    with pytest.raises(RuntimeError):
        other.commit(prepared)


def test_rollback_is_invalidated_by_later_changes_and_old_growth_api():
    brain, development = seeded()
    first = development.move({"a": (0, 0, 0)})
    second = development.move({"a": (0.5, 0, 0)})
    with pytest.raises(RuntimeError):
        first.rollback()
    second.rollback()
    with pytest.raises(RuntimeError):
        first.rollback()  # not a stack of old stale handles
    change = development.move({"a": (0.25, 0, 0)})
    brain.grow(brain.network)
    with pytest.raises(RuntimeError):
        change.rollback()


def test_reconfigure_is_exact_not_additive_and_can_retune_leak():
    brain, development = seeded()
    candidate = NetworkDefinition(
        tuple(
            replace(n, leak=0.6) if n.key == "a" else n for n in definition().nodes if n.key != "b"
        ),
        tuple(e for e in definition().edges if "b" not in e.key),
    )
    change = brain.reconfigure(candidate)
    assert change.report.removed_nodes == ("b",)
    assert change.report.retuned_nodes == ("a",)
    assert len(brain.network.nodes) == 3


def test_port_changes_are_explicit_and_role_changes_always_require_new_id():
    brain, development = seeded()
    candidate = NetworkDefinition(
        definition().nodes + (Node("i2", (1, 0, -1), "input"),), definition().edges
    )
    with pytest.raises(ValueError, match="allow_port_changes"):
        development.reconfigure(candidate)
    development.reconfigure(candidate, MigrationPolicy(allow_port_changes=True))
    assert brain.input_keys == ("i", "i2")
    brain.step({"i": 0.2, "i2": 0.3})
    changed = NetworkDefinition(
        tuple(replace(n, role="output") if n.key == "a" else n for n in brain.network.nodes),
        brain.network.edges,
    )
    with pytest.raises(ValueError, match="role"):
        development.reconfigure(changed, MigrationPolicy(allow_port_changes=True))


def test_port_reordering_requires_explicit_permission():
    brain = GrowingRNN(
        NetworkDefinition(
            (Node("i", (0, 0), "input"), Node("i2", (1, 0), "input"), Node("o", (0, 1), "output")),
            (),
        )
    )
    candidate = NetworkDefinition(
        (brain.network.nodes[1], brain.network.nodes[0], brain.network.nodes[2]), ()
    )
    with pytest.raises(ValueError, match="allow_port_changes"):
        brain.reconfigure(candidate)


def test_reexpress_stability_normalizes_changed_bases_but_preserves_learned_offsets():
    brain, development = seeded()
    before = brain.base_weights
    delta = brain.learned_deltas
    candidate = NetworkDefinition(
        brain.network.nodes,
        tuple(replace(e, weight=10) if e.target == "a" else e for e in brain.network.edges),
    )
    keys = tuple(e.key for e in candidate.edges if e.target == "a")
    change = development.reconfigure(candidate, MigrationPolicy(reexpress_edges=keys))
    assert change.report.normalized_edges
    assert brain.learned_deltas == delta
    assert brain.base_weights["b", "o"] == before["b", "o"]
    assert sum(
        abs(brain.base_weights[k] + brain.learned_deltas[k]) for k in keys if k[0] != "i"
    ) == pytest.approx(0.98)


def test_discarding_learning_is_explicit_and_stability_is_rechecked():
    brain, development = seeded()
    development.reconfigure(brain.network, MigrationPolicy(preserve_learning=False))
    assert all(v == 0 for v in brain.learned_deltas.values())


def test_state_and_weight_transfers_support_explicit_id_migration():
    brain, development = seeded()
    before = brain.snapshot()
    candidate = NetworkDefinition(
        tuple(replace(n, key="renamed") if n.key == "a" else n for n in brain.network.nodes),
        tuple(
            Edge(
                "renamed" if e.source == "a" else e.source,
                "renamed" if e.target == "a" else e.target,
                e.weight,
            )
            for e in brain.network.edges
        ),
    )
    weight_transfers = tuple(
        WeightTransfer(
            e.source,
            e.target,
            (
                (
                    "a" if e.source == "renamed" else e.source,
                    "a" if e.target == "renamed" else e.target,
                    1.0,
                ),
            ),
        )
        for e in candidate.edges
        if "renamed" in e.key
    )
    development.reconfigure(
        candidate,
        MigrationPolicy(
            states=(StateTransfer("renamed", (("a", 1.0),)),),
            weights=weight_transfers,
        ),
    )
    assert brain.state["renamed"] == before["state"]["a"]
    assert brain.learned_deltas["i", "renamed"] == 0.05


def test_maturity_policy_can_be_used_independently():
    original = {("i", "h"): 0.0, ("h", "o"): 0.75}
    assert Maturation(4).advance(original, 2) == {("i", "h"): 0.5, ("h", "o"): 1.0}
    assert original["i", "h"] == 0.0
    assert Maturation(0).advance(original, 0) == original
    assert all(v == 1 for v in Maturation(0).advance(original).values())
    with pytest.raises(ValueError):
        Maturation().advance(original, -1)
    with pytest.raises(ValueError):
        Maturation().advance({"x": float("nan")})


def test_snapshot_roundtrip_and_exact_continuation_after_prune_and_move():
    brain, development = seeded()
    development.prune(nodes=("b",))
    development.move({"a": (0.5, 0.5, 0.5)})
    saved = json.loads(json.dumps(brain.snapshot(), allow_nan=False))
    restored = GrowingRNN.from_snapshot(saved)
    assert saved["schema_version"] == 1  # old 0.1 snapshots remain readable
    for value in [0.2, 0.5, -0.2]:
        assert brain.step([value]) == restored.step([value])
    assert brain.snapshot() == restored.snapshot()


def test_output_path_diagnostics_are_structural_not_a_function_guarantee():
    brain = GrowingRNN(
        NetworkDefinition(
            (Node("i", (0, 0), "input"), Node("o", (0, 1), "output")),
            (Edge("i", "o", 0),),
        )
    )
    assert topology_health(brain.network).unreachable_outputs == ()
    assert brain.step([1])["o"] == 0
