from dataclasses import replace

import pytest
from test_runtime_development import definition, seeded

from spatial_neat import (
    Development,
    Edge,
    GrowingRNN,
    MigrationPolicy,
    NetworkDefinition,
    StateTransfer,
    WeightTransfer,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MigrationPolicy(weight_mode="unknown"),
        lambda: MigrationPolicy(preserve_learning="yes"),
        lambda: MigrationPolicy(reexpress_edges=(("a", "b"), ("a", "b"))),
        lambda: MigrationPolicy(
            states=(StateTransfer("a", (("a", 1),)), StateTransfer("a", (("a", 1),)))
        ),
        lambda: MigrationPolicy(
            weight_mode="reexpress", weights=(WeightTransfer("a", "b", (("a", "b", 1),)),)
        ),
        lambda: StateTransfer("a", ()),
        lambda: StateTransfer("a", (("a", 0.4), ("b", 0.4))),
        lambda: StateTransfer("a", (("a", -0.5), ("b", 1.5))),
        lambda: StateTransfer("a", (("a", float("nan")),)),
        lambda: StateTransfer("a", (("a", 0.5), ("a", 0.5))),
        lambda: WeightTransfer("a", "b", ()),
        lambda: WeightTransfer("a", "b", (("a", "b", -1),)),
        lambda: WeightTransfer("a", "b", (("a", "b", float("inf")),)),
    ],
)
def test_invalid_policies_and_coefficients_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "policy",
    [
        MigrationPolicy(states=(StateTransfer("missing", (("a", 1),)),)),
        MigrationPolicy(states=(StateTransfer("i", (("a", 1),)),)),
        MigrationPolicy(states=(StateTransfer("a", (("o", 1),)),)),
        MigrationPolicy(states=(StateTransfer("a", (("missing", 1),)),)),
        MigrationPolicy(reexpress_edges=(("i", "o"),)),
        MigrationPolicy(weights=(WeightTransfer("i", "o", (("i", "a", 1),)),)),
        MigrationPolicy(weights=(WeightTransfer("i", "a", (("missing", "a", 1),)),)),
    ],
)
def test_bad_migration_references_rejected_transactionally(policy):
    brain, development = seeded()
    before = brain.snapshot()
    with pytest.raises(ValueError):
        development.reconfigure(brain.network, policy)
    assert brain.snapshot() == before


def test_projected_weight_overflow_does_not_mutate_runtime():
    brain = GrowingRNN(
        NetworkDefinition(definition().nodes, (Edge("i", "a", 1e308), Edge("i", "b", 1e308)))
    )
    development = Development(brain)
    before = brain.snapshot()
    policy = MigrationPolicy(weights=(WeightTransfer("i", "a", (("i", "a", 1), ("i", "b", 1))),))
    with pytest.raises(ValueError, match="finite"):
        development.reconfigure(brain.network, policy)
    assert brain.snapshot() == before


def test_preview_runtime_budget_failure_does_not_mutate_live_network():
    brain = GrowingRNN(definition(), max_nodes=4, max_edges=8)
    before = brain.snapshot()
    candidate = NetworkDefinition(
        brain.network.nodes + (replace(brain.network.nodes[1], key="new"),), brain.network.edges
    )
    with pytest.raises(ValueError, match="budget"):
        Development(brain).preview(candidate)
    assert brain.snapshot() == before


def test_failed_restore_keeps_runtime_and_existing_rollback_point():
    brain, development = seeded()
    before = brain.snapshot()
    change = development.move({"a": (0, 0, 0)})
    moved = brain.snapshot()
    bad = brain.snapshot()
    bad["state"]["a"] = float("nan")
    with pytest.raises(ValueError):
        brain.restore(bad)
    assert brain.snapshot() == moved
    change.rollback()
    assert brain.snapshot() == before
