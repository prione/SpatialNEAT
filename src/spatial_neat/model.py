"""Immutable, simulation-independent geometry and sparse network definitions."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Node:
    """Stable logical identity; coordinates are not the runtime identity."""

    key: str
    position: tuple[float, ...]
    role: str
    leak: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "position", tuple(float(v) for v in self.position))
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("node key must be a nonempty string")
        if self.role not in ("input", "hidden", "output"):
            raise ValueError("invalid node role")
        if len(self.position) not in (2, 3) or not all(map(isfinite, self.position)):
            raise ValueError("positions must be finite 2D or 3D coordinates")
        if not isfinite(self.leak) or not 0 < self.leak <= 1:
            raise ValueError("leak must be in (0, 1]")


@dataclass(frozen=True)
class Region:
    """A bounded hidden-node search region, optionally one of several modules."""

    key: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    leak: float = 0.2

    def __post_init__(self):
        object.__setattr__(self, "lower", tuple(float(v) for v in self.lower))
        object.__setattr__(self, "upper", tuple(float(v) for v in self.upper))
        Node(self.key, self.lower, "hidden", self.leak)
        if len(self.upper) != len(self.lower) or not all(map(isfinite, self.upper)):
            raise ValueError("invalid region dimensions")
        if any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError("region bounds must have positive extent")


@dataclass(frozen=True)
class Substrate:
    """Fixed input/output anchors, with variable-density hidden search regions."""

    inputs: tuple[Node, ...]
    outputs: tuple[Node, ...]
    regions: tuple[Region, ...]

    def __post_init__(self):
        for field in ("inputs", "outputs", "regions"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.inputs or not self.outputs or not self.regions:
            raise ValueError("at least one input, output and search region is required")
        if any(n.role != "input" for n in self.inputs):
            raise ValueError("input anchors must have input role")
        if any(n.role != "output" for n in self.outputs):
            raise ValueError("output anchors must have output role")
        _unique([n.key for n in self.inputs + self.outputs], "anchor")
        _unique([r.key for r in self.regions], "region")
        dimensions = {len(n.position) for n in self.inputs + self.outputs}
        dimensions.update(len(r.lower) for r in self.regions)
        if len(dimensions) != 1:
            raise ValueError("all coordinates must have the same dimensionality")

    @property
    def dimensions(self):
        return len(self.inputs[0].position)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    weight: float

    def __post_init__(self):
        if not isfinite(self.weight):
            raise ValueError("edge weight must be finite")

    @property
    def key(self):
        return self.source, self.target


@dataclass(frozen=True)
class NetworkDefinition:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        _unique([n.key for n in self.nodes], "node")
        _unique([e.key for e in self.edges], "edge")
        nodes = {n.key: n for n in self.nodes}
        if not nodes or len({len(n.position) for n in self.nodes}) != 1:
            raise ValueError("network must have nodes of consistent dimensionality")
        if not any(n.role == "input" for n in self.nodes):
            raise ValueError("network needs an input")
        if not any(n.role == "output" for n in self.nodes):
            raise ValueError("network needs an output")
        for edge in self.edges:
            if edge.source not in nodes or edge.target not in nodes:
                raise ValueError("edge refers to an unknown node")
            if nodes[edge.target].role == "input":
                raise ValueError("input nodes cannot receive edges")

    @property
    def hidden_count(self):
        return sum(n.role == "hidden" for n in self.nodes)

    def to_dict(self):
        return {
            "nodes": [
                {"key": n.key, "position": list(n.position), "role": n.role, "leak": n.leak}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "weight": e.weight} for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(tuple(Node(**n) for n in data["nodes"]), tuple(Edge(**e) for e in data["edges"]))


def _unique(keys, name):
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate {name} key")
