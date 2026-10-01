import json

import numpy as np
import pytest
from test_runtime_development import definition, seeded

from growth_hyperneat import Development, GrowingRNN, Node


def test_optional_merge_projects_states_incoming_and_outgoing_weights():
    brain, development = seeded()
    before = brain.snapshot()
    prepared = development.preview_merge(
        ("a", "b"), Node("merged", (0, 0, 0), "hidden", 0.3), coefficients=(0.25, 0.75)
    )
    trial = prepared.trial()
    assert brain.snapshot() == before
    assert trial.state["merged"] == pytest.approx(0.25 * brain.state["a"] + 0.75 * brain.state["b"])
    assert trial.base_weights["i", "merged"] == pytest.approx(0.25 * 0.4 + 0.75 * 0.8)
    assert trial.learned_deltas["i", "merged"] == pytest.approx(0.25 * 0.05)
    assert trial.base_weights["merged", "o"] == pytest.approx(0.2 + 0.3)
    assert trial.learned_deltas["merged", "o"] == pytest.approx(0.1)
    assert trial.base_weights["merged", "merged"] == pytest.approx(
        0.25 * (0.1 + 0.2) + 0.75 * (0.3 + 0.2)
    )
    change = development.commit(prepared)
    assert change.report.removed_nodes == ("a", "b")
    assert change.report.added_nodes == ("merged",)
    assert change.report.transferred_nodes == ("merged",)
    assert brain.network.hidden_count == 1
    assert brain.step([0.2]) == trial.step([0.2])
    change.rollback()
    assert brain.snapshot() == before


def test_merge_can_keep_one_source_identity_without_colliding():
    brain, development = seeded()
    change = development.merge(("a", "b"), Node("a", (0, 0, 0), "hidden", 0.3))
    assert change.report.removed_nodes == ("b",)
    assert set(brain.state) == {"a", "o"}


def test_merge_does_not_claim_nonlinear_function_equivalence():
    brain = GrowingRNN(definition(), recurrent_limit=None)
    development = Development(brain)
    development.merge(("a", "b"), Node("m", (0, 0, 0), "hidden", 0.3))
    brain.step([1])
    reference_average = 0.5 * 0.3 * (np.tanh(0.4) + np.tanh(0.8))
    assert abs(brain.state["m"] - reference_average) > 1e-5


def test_merge_of_mixed_maturity_folds_current_gates_into_projected_weights():
    brain, development = seeded()
    # This source snapshot is a supported way to restore a paused mature/growing network.
    saved = brain.snapshot()
    for weight in saved["weights"]:
        if (weight["source"], weight["target"]) == ("b", "o"):
            weight["gate"] = 0.5
    brain.restore(saved)
    development.merge(("a", "b"), Node("m", (0, 0, 0), "hidden", 0.3))
    assert brain.base_weights["m", "o"] == pytest.approx(0.2 + 0.5 * 0.3)
    assert brain.learned_deltas["m", "o"] == pytest.approx(0.5 * 0.1)
    assert brain.gates["m", "o"] == 1.0


@pytest.mark.parametrize(
    "sources, target, coefficients",
    [
        (("a",), Node("m", (0, 0, 0), "hidden"), None),
        (("a", "a"), Node("m", (0, 0, 0), "hidden"), None),
        (("a", "missing"), Node("m", (0, 0, 0), "hidden"), None),
        (("a", "i"), Node("m", (0, 0, 0), "hidden"), None),
        (("a", "b"), Node("o", (0, 0, 0), "hidden"), None),
        (("a", "b"), Node("m", (0, 0), "hidden"), None),
        (("a", "b"), Node("m", (0, 0, 0), "output"), None),
        (("a", "b"), Node("m", (0, 0, 0), "hidden"), (1,)),
        (("a", "b"), Node("m", (0, 0, 0), "hidden"), (0.5, -0.5)),
        (("a", "b"), Node("m", (0, 0, 0), "hidden"), (0.8, 0.8)),
    ],
)
def test_invalid_merge_rejected_without_modifying_live_network(sources, target, coefficients):
    brain, development = seeded()
    before = brain.snapshot()
    with pytest.raises(ValueError):
        development.merge(sources, target, coefficients)
    assert brain.snapshot() == before


def test_compressed_snapshot_exactly_resumes_and_can_grow_again():
    brain, development = seeded()
    development.merge(("a", "b"), Node("m", (0, 0, 0), "hidden", 0.3))
    restored = GrowingRNN.from_snapshot(json.loads(json.dumps(brain.snapshot())))
    for value in (0.1, 0.2, 0.3):
        assert brain.step([value]) == restored.step([value])
    # A subsequent explicit grow invalidates the old merge undo point and keeps
    # compressed knowledge, even though the original discovery IDs return.
    brain.grow(definition())
    assert {"m", "a", "b"} <= set(brain.state)
