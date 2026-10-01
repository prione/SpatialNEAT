"""Optional approximate hidden-node compression, separate from growth."""

from dataclasses import replace
from math import isfinite

from .migration import MigrationPolicy, StateTransfer, WeightTransfer
from .model import Edge, NetworkDefinition, Node


def merge_hidden(snapshot, sources, target: Node, coefficients=None):
    """Return a topology and explicit migration plan for a linear projection.

    State is a convex average. Incoming rows are averaged, outgoing columns
    summed: W_new = R W_old P. This does NOT preserve a nonlinear tanh RNN's
    function in general. No similarity criterion or automatic merge decision
    is imposed; validate/retrain the candidate before accepting it.
    """
    sources = tuple(sources)
    if len(sources) < 2 or len(sources) != len(set(sources)):
        raise ValueError("compression needs at least two distinct hidden nodes")
    network = NetworkDefinition.from_dict(snapshot["network"])
    nodes = {n.key: n for n in network.nodes}
    if target.role != "hidden" or any(k not in nodes or nodes[k].role != "hidden" for k in sources):
        raise ValueError("only hidden nodes can be merged")
    if len(target.position) != len(network.nodes[0].position):
        raise ValueError("merge target dimensionality differs")
    if target.key in nodes and target.key not in sources:
        raise ValueError("merge destination collides with an unrelated node")
    coefficients = (
        tuple(float(c) for c in coefficients)
        if coefficients is not None
        else ((1 / len(sources),) * len(sources))
    )
    if len(coefficients) != len(sources):
        raise ValueError("one coefficient per source is required")
    state_transfer = StateTransfer(target.key, tuple(zip(sources, coefficients)))
    alpha = dict(state_transfer.sources)

    def mapped(key):
        return target.key if key in alpha else key

    projected = {}
    for edge in network.edges:
        key = mapped(edge.source), mapped(edge.target)
        projected.setdefault(key, []).append(
            (edge.source, edge.target, alpha.get(edge.target, 1.0))
        )
    old_weights = {(w["source"], w["target"]): w for w in snapshot["weights"]}
    edges, transfers = [], []
    for (source, destination), contributors in projected.items():
        # Raw topology weights are metadata. Runtime migration projects base,
        # learned deltas and gates from the actual snapshot, not this field.
        weight = sum(old_weights[a, b]["base"] * c for a, b, c in contributors)
        if not isfinite(weight):
            raise ValueError("compression weight overflow")
        edges.append(Edge(source, destination, weight))
        if any(a in alpha or b in alpha for a, b, _ in contributors):
            transfers.append(WeightTransfer(source, destination, tuple(contributors)))
    kept = []
    inserted = False
    for node in network.nodes:
        if node.key in alpha:
            if not inserted:
                kept.append(target)
                inserted = True
        else:
            kept.append(node)
    candidate = NetworkDefinition(tuple(kept), tuple(edges))
    policy = replace(MigrationPolicy(), states=(state_transfer,), weights=tuple(transfers))
    return candidate, policy
