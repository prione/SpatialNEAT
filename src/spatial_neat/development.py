"""Development operations, isolated previews, validated commits and explicit rollback."""

from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping

import numpy as np

from .compression import merge_hidden
from .expression import coordinate_features
from .migration import DevelopmentReport, MigrationPolicy, migrate_snapshot
from .model import Edge, NetworkDefinition, Node
from .runtime import GrowingRNN


@dataclass(frozen=True)
class PreparedChange:
    """A validated proposal. trial() gives an independent RNN for evaluation."""

    report: DevelopmentReport
    _brain: GrowingRNN
    _token: object
    _before: dict
    _after: dict

    @property
    def before_snapshot(self):
        return deepcopy(self._before)

    @property
    def after_snapshot(self):
        return deepcopy(self._after)

    def trial(self):
        return GrowingRNN.from_snapshot(self._after)


class DevelopmentChange:
    """Undo handle. Rollback rewinds all trial-time state/learning after the commit.

    It is invalidated by another topology change or explicit snapshot restoration.
    External optimizer state is not owned by this library and must be rolled back
    separately by the application.
    """

    def __init__(self, brain, before, report):
        self.report = report
        self._brain, self._before = brain, before
        self._token = brain._structure_token
        self._active = True

    def rollback(self):
        if not self._active or self._brain._structure_token is not self._token:
            raise RuntimeError("change is no longer the latest rollback point")
        self._brain.restore(self._before)
        self._active = False


class Development:
    """Domain-neutral lifecycle operations; applications decide when and why.

    Exact topology replacement is explicit in reconfigure(). grow() is additive.
    Pruning/moving/merging never infer importance from activities or rewards.
    """

    def __init__(self, brain: GrowingRNN):
        self.brain = brain

    def preview(self, candidate: NetworkDefinition, policy: MigrationPolicy | None = None):
        before = self.brain.snapshot()
        after, report = migrate_snapshot(before, candidate, policy or MigrationPolicy())
        # Validate runtime budgets, finite values and stability before any commit.
        GrowingRNN.from_snapshot(after)
        return PreparedChange(report, self.brain, self.brain._structure_token, before, after)

    def commit(self, prepared: PreparedChange) -> DevelopmentChange:
        if prepared._brain is not self.brain or prepared._token is not self.brain._structure_token:
            raise RuntimeError("proposal belongs to a different or changed runtime")
        if self.brain.snapshot() != prepared._before:
            raise RuntimeError("runtime advanced since preview; prepare a fresh change")
        staged = prepared.trial()
        self.brain._adopt(staged)
        return DevelopmentChange(self.brain, prepared.before_snapshot, prepared.report)

    def reconfigure(self, candidate: NetworkDefinition, policy: MigrationPolicy | None = None):
        return self.commit(self.preview(candidate, policy))

    def grow(self, candidate: NetworkDefinition) -> DevelopmentChange:
        known = {n.key: n for n in self.brain.network.nodes}
        for node in candidate.nodes:
            if node.key in known and node != known[node.key]:
                raise ValueError(f"existing node changed: {node.key}")
        edges = {e.key for e in self.brain.network.edges}
        merged = NetworkDefinition(
            self.brain.network.nodes + tuple(n for n in candidate.nodes if n.key not in known),
            self.brain.network.edges + tuple(e for e in candidate.edges if e.key not in edges),
        )
        # Explicit grow may add ports, but it never removes/reorders established ports.
        return self.reconfigure(merged, MigrationPolicy(allow_port_changes=True))

    def mature(self, ticks=1):
        """Advance connection gates without advancing neural state."""
        self.brain.mature(ticks)

    def prune(self, nodes=(), edges=(), policy: MigrationPolicy | None = None):
        nodes, edges = set(nodes), {tuple(e) for e in edges}
        known = {n.key: n for n in self.brain.network.nodes}
        available = {e.key for e in self.brain.network.edges}
        if not nodes <= known.keys() or not edges <= available:
            raise ValueError("pruning target does not exist")
        candidate = NetworkDefinition(
            tuple(n for n in self.brain.network.nodes if n.key not in nodes),
            tuple(
                e
                for e in self.brain.network.edges
                if e.key not in edges and e.source not in nodes and e.target not in nodes
            ),
        )
        return self.reconfigure(candidate, policy)

    def move(
        self,
        positions: Mapping[str, tuple[float, ...]],
        cppn=None,
        features=coordinate_features,
        weight_scale=1.0,
        preserve_learning=True,
        require_output_paths=False,
    ):
        """Relocate IDs. With a CPPN, re-express only incident edges, not topology.

        Without a CPPN the computation is unchanged. Re-expression changes
        weights immediately (and may normalize them); it is not a smooth-growth
        guarantee. Use preview/reconfigure for custom relocation policies.
        """
        known = {n.key for n in self.brain.network.nodes}
        if not positions.keys() <= known:
            raise ValueError("moving target does not exist")
        if not isfinite(weight_scale) or weight_scale <= 0:
            raise ValueError("weight_scale must be finite and positive")
        nodes = tuple(
            replace(n, position=positions[n.key]) if n.key in positions else n
            for n in self.brain.network.nodes
        )
        moved = {n.key: n for n in nodes}
        edges, regenerated = [], []
        for edge in self.brain.network.edges:
            if cppn is not None and (edge.source in positions or edge.target in positions):
                query = np.asarray(
                    features(moved[edge.source].position, moved[edge.target].position), dtype=float
                )
                if query.ndim != 1 or not np.isfinite(query).all():
                    raise ValueError("feature encoder must return a finite vector")
                result = np.asarray(cppn.query(query), dtype=float)
                if result.ndim != 1 or not result.size or not np.isfinite(result).all():
                    raise ValueError("CPPN must return a finite output vector")
                edge = Edge(
                    edge.source, edge.target, float(np.clip(result[0], -1, 1)) * weight_scale
                )
                regenerated.append(edge.key)
            edges.append(edge)
        policy = MigrationPolicy(
            preserve_learning=preserve_learning,
            require_output_paths=require_output_paths,
            reexpress_edges=tuple(regenerated),
        )
        return self.reconfigure(NetworkDefinition(nodes, tuple(edges)), policy)

    def preview_merge(self, sources, target: Node, coefficients=None, require_output_paths=False):
        candidate, policy = merge_hidden(self.brain.snapshot(), sources, target, coefficients)
        return self.preview(candidate, replace(policy, require_output_paths=require_output_paths))

    def merge(self, sources, target: Node, coefficients=None, require_output_paths=False):
        """Approximate compression; callers should prefer preview_merge + evaluation."""
        return self.commit(self.preview_merge(sources, target, coefficients, require_output_paths))
