"""Growth, maturity, pruning, movement and optional approximate compression.

This example verifies lifecycle mechanics, not autonomous developmental learning.
"""

import json

from growth_hyperneat import Development, Edge, GrowingRNN, NetworkDefinition, Node


def main():
    network = NetworkDefinition(
        (
            Node("i", (0, 0, -1), "input"),
            Node("a", (-0.25, 0, 0), "hidden", 0.3),
            Node("b", (0.25, 0, 0), "hidden", 0.3),
            Node("o", (0, 0, 1), "output"),
        ),
        (
            Edge("i", "a", 0.4),
            Edge("i", "b", 0.8),
            Edge("a", "a", 0.2),
            Edge("b", "b", 0.2),
            Edge("a", "o", 0.2),
            Edge("b", "o", 0.3),
        ),
    )
    brain = GrowingRNN(network)
    development = Development(brain)
    brain.step([0.5])
    brain.set_delta("i", "a", 0.05)
    grown = NetworkDefinition(
        network.nodes + (Node("extra", (0, 0.5, 0), "hidden"),),
        network.edges + (Edge("i", "extra", 0.2),),
    )
    development.grow(grown)
    development.mature(ticks=4)
    gate = brain.gates["i", "extra"]
    development.prune(nodes=("extra",))
    development.move({"a": (-0.5, 0, 0)})  # same identity, state and weights
    before_merge = brain.snapshot()
    prepared = development.preview_merge(("a", "b"), Node("merged", (0, 0, 0), "hidden", 0.3))
    trial = prepared.trial()
    trial.step([0.2])  # application can evaluate old skills on this isolated copy
    change = development.commit(prepared)
    compressed_hidden = brain.network.hidden_count
    brain.step([0.3])
    change.rollback()  # demonstration of rejecting a compression trial
    print(
        json.dumps(
            {
                "maturity_after_4_ticks": gate,
                "compressed_hidden": compressed_hidden,
                "restored_hidden": brain.network.hidden_count,
                "rollback_exact": brain.snapshot() == before_merge,
            }
        )
    )


if __name__ == "__main__":
    main()
