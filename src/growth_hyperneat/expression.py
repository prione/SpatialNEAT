"""Iterated ES-HyperNEAT spatial extraction, generalized to quadtrees/octrees.

Based on Risi & Stanley (GECCO 2011), sections 3.1 and 3.2. Recurrent
connections, modular regions, budgets and direct links are explicit extensions.
This is an independent implementation, not a copy of the bundled Pureples code.
"""

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Callable, Protocol

import numpy as np

from .model import Edge, NetworkDefinition, Node, Substrate


class CPPN(Protocol):
    def query(self, values: np.ndarray) -> np.ndarray:
        """Return (..., outputs) for stateless (..., features) coordinate queries."""


def coordinate_features(source, target):
    """[source coordinates, target coordinates, Euclidean distance, bias]."""
    return np.r_[source, target, np.linalg.norm(np.subtract(source, target)), 1.0]


class ExpressionLimitError(RuntimeError):
    """Discovery aborted without returning a silently truncated network."""


@dataclass(frozen=True)
class ESConfig:
    initial_depth: int = 2
    max_depth: int = 3
    division_threshold: float = 0.03
    variance_threshold: float = 0.03
    band_threshold: float = 0.05
    weight_threshold: float = 0.01
    weight_scale: float = 1.0
    iterations: int = 1
    allow_self_connections: bool = True
    direct_links: bool = False
    max_hidden_nodes: int = 256
    max_edges: int = 8192
    max_queries: int = 200000

    def __post_init__(self):
        integer_fields = (
            "initial_depth",
            "max_depth",
            "iterations",
            "max_hidden_nodes",
            "max_edges",
            "max_queries",
        )
        if any(type(getattr(self, f)) is not int for f in integer_fields):
            raise ValueError("depth, iteration and budget settings must be integers")
        if not 1 <= self.initial_depth <= self.max_depth <= 12 or self.iterations < 0:
            raise ValueError("invalid depth or iteration setting")
        for name in (
            "division_threshold",
            "variance_threshold",
            "band_threshold",
            "weight_threshold",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("thresholds must be finite and nonnegative")
        if not isfinite(self.weight_scale) or self.weight_scale <= 0:
            raise ValueError("weight_scale must be finite and positive")
        if min(self.max_hidden_nodes, self.max_edges, self.max_queries) < 1:
            raise ValueError("budgets must be positive")


@dataclass(frozen=True)
class ExpressionResult:
    network: NetworkDefinition
    queries: int
    candidates: int


@dataclass
class _Cell:
    center: np.ndarray
    half: np.ndarray
    depth: int
    path: str
    value: float
    children: tuple = ()
    variance: float = 0.0


class ESDeveloper:
    """Reusable developer. Each call has independent caches and search state.

    A connection_filter may express domain-specific locality without making the
    library depend on bodies, tasks or a physics engine. It must be deterministic.
    """

    def __init__(
        self,
        config: ESConfig | None = None,
        features: Callable = coordinate_features,
        connection_filter: Callable[[Node, Node], bool] | None = None,
    ):
        self.config = config or ESConfig()
        self.features = features
        self.connection_filter = connection_filter

    def develop(self, cppn: CPPN, substrate: Substrate) -> ExpressionResult:
        return _Discovery(self, cppn, substrate).run()


class _Discovery:
    def __init__(self, developer, cppn, substrate):
        self.config = developer.config
        self.features = developer.features
        self.filter = developer.connection_filter
        self.cppn, self.substrate = cppn, substrate
        self.cache = {}
        self.hidden = {}
        self.edges = {}
        self.anchor_keys = {n.key for n in substrate.inputs + substrate.outputs}

    def sample(self, source, target):
        key = tuple(source), tuple(target)
        if key not in self.cache:
            if len(self.cache) >= self.config.max_queries:
                raise ExpressionLimitError("CPPN query budget exceeded")
            features = np.asarray(self.features(source, target), dtype=float)
            if features.ndim != 1 or not np.isfinite(features).all():
                raise ValueError("feature encoder must return a finite vector")
            result = np.asarray(self.cppn.query(features), dtype=float)
            if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
                raise ValueError("CPPN must return a finite nonempty output vector")
            # Bound arbitrary CPPN activations, without distorting [-1, 1] outputs.
            self.cache[key] = float(np.clip(result[0], -1, 1))
        return self.cache[key]

    def tree(self, anchor, region, outgoing):
        def evaluate(position):
            return (
                self.sample(anchor.position, position)
                if outgoing
                else self.sample(position, anchor.position)
            )

        lower, upper = np.asarray(region.lower), np.asarray(region.upper)
        root = _Cell((lower + upper) / 2, (upper - lower) / 2, 0, "", 0.0)

        def divide(cell):
            half = cell.half / 2
            children = []
            for i, signs in enumerate(product((-1, 1), repeat=len(half))):
                center = cell.center + half * signs
                children.append(
                    _Cell(center, half, cell.depth + 1, cell.path + str(i), evaluate(center))
                )
            cell.children = tuple(children)
            local_variance = float(np.var([child.value for child in children]))
            if cell.depth + 1 < self.config.initial_depth or (
                cell.depth + 1 < self.config.max_depth
                and local_variance > self.config.division_threshold
            ):
                for child in children:
                    divide(child)
            leaves = []

            def collect(point):
                if point.children:
                    for child in point.children:
                        collect(child)
                else:
                    leaves.append(point.value)

            collect(cell)
            cell.variance = float(np.var(leaves))

        divide(root)
        return root, evaluate

    def extract(self, anchor, region, outgoing):
        root, evaluate = self.tree(anchor, region, outgoing)

        def prune(cell):
            for child in cell.children:
                if child.children and child.variance > self.config.variance_threshold:
                    yield from prune(child)
                    continue
                band = 0.0
                for axis in range(len(child.center)):
                    offset = np.zeros(len(child.center))
                    offset[axis] = 2 * child.half[axis]
                    # Like the original band test, neighbors may lie outside the
                    # region; candidates themselves always stay inside its bounds.
                    left = abs(child.value - evaluate(child.center - offset))
                    right = abs(child.value - evaluate(child.center + offset))
                    band = max(band, min(left, right))
                if band > self.config.band_threshold:
                    # Region + octree path is stable across deterministic re-expression.
                    key = f"hidden:{len(region.key)}:{region.key}:{child.path}"
                    node = Node(key, tuple(child.center), "hidden", region.leak)
                    source, target = (anchor, node) if outgoing else (node, anchor)
                    if self.accept(source, target, child.value):
                        yield node, source, target, child.value

        yield from prune(root)

    def accept(self, source, target, weight):
        return (
            abs(weight) > self.config.weight_threshold
            and (source.key != target.key or self.config.allow_self_connections)
            and (self.filter is None or self.filter(source, target))
        )

    def add(self, node, source, target, weight):
        if node.key in self.anchor_keys:
            raise ValueError("generated hidden key collides with anchor key")
        if node.key not in self.hidden:
            if len(self.hidden) >= self.config.max_hidden_nodes:
                raise ExpressionLimitError("hidden candidate budget exceeded")
            self.hidden[node.key] = node
        key = source.key, target.key
        if key not in self.edges and len(self.edges) >= self.config.max_edges:
            raise ExpressionLimitError("edge budget exceeded")
        self.edges[key] = Edge(*key, weight * self.config.weight_scale)

    def run(self):
        frontier = list(self.substrate.inputs)
        explored = set()
        for _ in range(self.config.iterations + 1):
            discovered = set()
            for anchor in frontier:
                explored.add(anchor.key)
                for region in self.substrate.regions:
                    for node, source, target, weight in self.extract(anchor, region, True):
                        self.add(node, source, target, weight)
                        discovered.add(node.key)
            frontier = [self.hidden[key] for key in sorted(discovered - explored)]
            if not frontier:
                break
        for anchor in self.substrate.outputs:
            for region in self.substrate.regions:
                for node, source, target, weight in self.extract(anchor, region, False):
                    self.add(node, source, target, weight)
        if self.config.direct_links:
            for source in self.substrate.inputs:
                for target in self.substrate.outputs:
                    value = self.sample(source.position, target.position)
                    if self.accept(source, target, value):
                        if len(self.edges) >= self.config.max_edges:
                            raise ExpressionLimitError("edge budget exceeded")
                        self.edges[source.key, target.key] = Edge(
                            source.key, target.key, value * self.config.weight_scale
                        )

        def reachable(start, reverse=False):
            adjacency = {}
            for source, target in self.edges:
                a, b = (target, source) if reverse else (source, target)
                adjacency.setdefault(a, set()).add(b)
            start = tuple(start)
            seen, stack = set(start), list(start)
            while stack:
                for key in sorted(adjacency.get(stack.pop(), ())):
                    if key not in seen:
                        seen.add(key)
                        stack.append(key)
            return seen

        live = reachable(n.key for n in self.substrate.inputs) & reachable(
            (n.key for n in self.substrate.outputs), reverse=True
        )
        hidden = tuple(self.hidden[key] for key in sorted(live & self.hidden.keys()))
        nodes = self.substrate.inputs + hidden + self.substrate.outputs
        edges = tuple(
            self.edges[key] for key in sorted(self.edges) if key[0] in live and key[1] in live
        )
        return ExpressionResult(NetworkDefinition(nodes, edges), len(self.cache), len(self.hidden))
