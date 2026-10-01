"""Expression and growth exercised together, not only with hand-built networks."""

from dataclasses import replace

import numpy as np

from spatial_neat import ESConfig, ESDeveloper, GrowingRNN, Node, Region, Substrate


class Pattern:
    def query(self, values):
        return np.array([np.exp(-3 * np.sum(np.asarray(values)[:6] ** 2))])


def test_discovered_network_grows_a_new_region_without_losing_memory_or_learning():
    core = Region("core", (-1, -1, -1), (1, 1, 1))
    substrate = Substrate(
        (Node("i", (0, 0, 0), "input"),), (Node("o", (0, 0, 0), "output"),), (core,)
    )
    developer = ESDeveloper(
        ESConfig(max_depth=2, variance_threshold=0, band_threshold=0.001, max_edges=20000)
    )
    initial = developer.develop(Pattern(), substrate)
    brain = GrowingRNN(initial.network)
    brain.step([0.8])
    outgoing = next(e for e in initial.network.edges if e.target == "o")
    accepted = brain.set_delta(outgoing.source, outgoing.target, -0.001)
    assert accepted != 0
    original = brain.snapshot()
    baseline = GrowingRNN.from_snapshot(original)
    grown = replace(substrate, regions=(core, Region("new", (-1, -1, -1), (1, 1, 1))))
    candidate = developer.develop(Pattern(), grown)
    report = brain.grow(candidate.network)
    assert report.added_nodes
    assert brain.network.hidden_count > initial.network.hidden_count
    assert all(brain.state[k] == value for k, value in original["state"].items())
    assert brain.learned_deltas[outgoing.key] == accepted
    assert brain.step([0.3]) == baseline.step([0.3])
    restored = GrowingRNN.from_snapshot(brain.snapshot())
    for _ in range(10):
        assert brain.step([0.1]) == restored.step([0.1])
