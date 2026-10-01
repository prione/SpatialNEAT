"""Compiled, immutable sparse indices; independent of mutable execution state."""

from dataclasses import dataclass

import numpy as np

from .model import NetworkDefinition


def _indices(values):
    result = np.asarray(values, dtype=np.intp)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class RuntimeLayout:
    input_keys: tuple
    output_keys: tuple
    dynamic_keys: tuple
    edge_keys: tuple
    edge_indices: dict
    input_indices: np.ndarray
    dynamic_indices: np.ndarray
    output_indices: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    leaks: np.ndarray
    recurrent_rows: tuple
    roles: dict

    @classmethod
    def compile(cls, network: NetworkDefinition):
        positions = {node.key: i for i, node in enumerate(network.nodes)}
        roles = {node.key: node.role for node in network.nodes}
        inputs = tuple(n.key for n in network.nodes if n.role == "input")
        outputs = tuple(n.key for n in network.nodes if n.role == "output")
        dynamic = tuple(n.key for n in network.nodes if n.role != "input")
        destinations = {key: i for i, key in enumerate(dynamic)}
        edges = tuple(e.key for e in network.edges)
        rows = [[] for _ in dynamic]
        for i, (source, target) in enumerate(edges):
            if roles[source] != "input":
                rows[destinations[target]].append(i)
        leaks = np.array([n.leak for n in network.nodes if n.role != "input"], dtype=float)
        leaks.flags.writeable = False
        return cls(
            inputs,
            outputs,
            dynamic,
            edges,
            {key: i for i, key in enumerate(edges)},
            _indices([positions[k] for k in inputs]),
            _indices([positions[k] for k in dynamic]),
            _indices([destinations[k] for k in outputs]),
            _indices([positions[a] for a, _ in edges]),
            _indices([destinations[b] for _, b in edges]),
            leaks,
            tuple(_indices(row) for row in rows),
            roles,
        )
