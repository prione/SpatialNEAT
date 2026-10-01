"""Deterministic geometric discovery, neural growth and exact state restoration."""

import json
from dataclasses import replace

import numpy as np

from growth_hyperneat import ESConfig, ESDeveloper, GrowingRNN, Node, Region, Substrate


class ExampleCPPN:
    """Analytical coordinate pattern; substitute NeatCPPN for evolved patterns."""

    def query(self, values):
        values = np.asarray(values)
        return np.exp(-3 * np.sum(values[..., :6] ** 2, axis=-1))[..., None]


def main():
    core = Region("core", (-1, -1, -1), (1, 1, 1))
    substrate = Substrate(
        (Node("sensor", (0, 0, 0), "input"),),
        (Node("action", (0, 0, 0), "output"),),
        (core,),
    )
    developer = ESDeveloper(ESConfig(max_depth=2, variance_threshold=0, band_threshold=0.001))
    cppn = ExampleCPPN()
    first = developer.develop(cppn, substrate)
    brain = GrowingRNN(first.network)
    for _ in range(4):
        brain.step({"sensor": 0.5})
    old_state, old_base = brain.state, brain.base_weights
    # In a real application this decision belongs to its growth policy, not ES.
    grown = replace(substrate, regions=(core, Region("new-module", (-1, -1, -1), (1, 1, 1))))
    candidate = developer.develop(cppn, grown)
    report = brain.grow(candidate.network)
    assert all(brain.state[k] == v for k, v in old_state.items())
    assert all(brain.base_weights[k] == v for k, v in old_base.items())
    saved = json.loads(json.dumps(brain.snapshot(), allow_nan=False))
    restored = GrowingRNN.from_snapshot(saved)
    assert brain.step([0.2]) == restored.step([0.2])
    print(
        json.dumps(
            {
                "before_hidden": first.network.hidden_count,
                "after_hidden": brain.network.hidden_count,
                "added_nodes": len(report.added_nodes),
                "complexity": brain.complexity,
                "exact_resume": brain.snapshot() == restored.snapshot(),
            }
        )
    )


if __name__ == "__main__":
    main()
