"""Sparse synchronous leaky RNN; lifecycle operations live in development."""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from .maturation import Maturation
from .model import NetworkDefinition


@dataclass(frozen=True)
class GrowthReport:
    added_nodes: tuple[str, ...]
    added_edges: tuple[tuple[str, str], ...]


class GrowingRNN:
    """All non-input neurons update from the previous timestep, including outputs.

    A conservative absolute recurrent row-sum bound controls tanh dynamics.
    Setting recurrent_limit=None disables it. Growth never changes established
    base weights or learned deltas; only new weights use remaining row budget.
    Biases are represented by an explicit constant-valued input anchor if needed.
    """

    def __init__(
        self,
        network: NetworkDefinition,
        recurrent_limit: float | None = 0.98,
        maturation_steps: int = 8,
        max_nodes: int = 4096,
        max_edges: int = 65536,
    ):
        if recurrent_limit is not None and (
            not isfinite(recurrent_limit) or not 0 < recurrent_limit < 1
        ):
            raise ValueError("recurrent_limit must be in (0, 1) or None")
        if type(maturation_steps) is not int or maturation_steps < 0:
            raise ValueError("maturation_steps must be a nonnegative integer")
        if type(max_nodes) is not int or type(max_edges) is not int:
            raise ValueError("runtime budgets must be integers")
        if min(max_nodes, max_edges) < 1:
            raise ValueError("runtime budgets must be positive")
        self.recurrent_limit = recurrent_limit
        self.maturation_steps = maturation_steps
        self.max_nodes, self.max_edges = max_nodes, max_edges
        self._check_budget(network)
        self.network = network
        self._base = {e.key: float(e.weight) for e in network.edges}
        self._delta = dict.fromkeys(self._base, 0.0)
        self._gates = dict.fromkeys(self._base, 1.0)
        self._normalize_new(network, set(self._base), self._base, self._delta)
        self._state = {n.key: 0.0 for n in network.nodes if n.role != "input"}
        self._structure_token = object()
        self._index()

    @property
    def state(self):
        return dict(self._state)

    @property
    def base_weights(self):
        return dict(self._base)

    @property
    def learned_deltas(self):
        return dict(self._delta)

    @property
    def gates(self):
        return dict(self._gates)

    @property
    def input_keys(self):
        return tuple(n.key for n in self.network.nodes if n.role == "input")

    @property
    def output_keys(self):
        return tuple(n.key for n in self.network.nodes if n.role == "output")

    @property
    def complexity(self):
        """Raw structural counts, including dormant/growing edges, for fitness costs."""
        return {
            "nodes": len(self.network.nodes),
            "hidden": self.network.hidden_count,
            "edges": len(self.network.edges),
        }

    def _check_budget(self, network):
        if len(network.nodes) > self.max_nodes or len(network.edges) > self.max_edges:
            raise ValueError("runtime topology budget exceeded")

    def _normalize_new(self, network, new, base, delta):
        if self.recurrent_limit is None:
            return
        roles = {n.key: n.role for n in network.nodes}
        rows = {}
        for edge in network.edges:
            if roles[edge.source] != "input":
                rows.setdefault(edge.target, []).append(edge.key)
        for keys in rows.values():
            established = sum(abs(base[k] + delta[k]) for k in keys if k not in new)
            amount = sum(abs(base[k]) for k in keys if k in new)
            available = max(0.0, self.recurrent_limit - established)
            if amount > available:
                for key in keys:
                    if key in new:
                        base[key] *= available / amount

    def _index(self):
        nodes = self.network.nodes
        positions = {node.key: i for i, node in enumerate(nodes)}
        self._input_indices = np.array([positions[key] for key in self.input_keys], dtype=int)
        self._dynamic_keys = tuple(n.key for n in nodes if n.role != "input")
        self._dynamic_indices = np.array([positions[k] for k in self._dynamic_keys], dtype=int)
        dynamic = {key: i for i, key in enumerate(self._dynamic_keys)}
        self._sources = np.array([positions[e.source] for e in self.network.edges], dtype=int)
        self._targets = np.array([dynamic[e.target] for e in self.network.edges], dtype=int)
        self._leaks = np.array([nodes[positions[k]].leak for k in self._dynamic_keys])
        self._roles = {n.key: n.role for n in nodes}

    def step(self, inputs) -> dict[str, float]:
        if isinstance(inputs, Mapping):
            if set(inputs) != set(self.input_keys):
                raise ValueError("input mapping must contain exactly all input keys")
            inputs = [inputs[key] for key in self.input_keys]
        values = np.asarray(inputs, dtype=float)
        if values.shape != (len(self.input_keys),) or not np.isfinite(values).all():
            raise ValueError("inputs must be a finite vector matching input_keys")
        all_values = np.zeros(len(self.network.nodes))
        all_values[self._input_indices] = values
        previous = np.array([self._state[k] for k in self._dynamic_keys])
        all_values[self._dynamic_indices] = previous
        weights = np.array(
            [
                (self._base[e.key] + self._delta[e.key]) * self._gates[e.key]
                for e in self.network.edges
            ]
        )
        total = np.zeros(len(self._dynamic_keys))
        with np.errstate(over="ignore", invalid="ignore"):
            np.add.at(total, self._targets, all_values[self._sources] * weights)
        if not np.isfinite(total).all():
            raise ValueError("non-finite neural accumulation; state was not advanced")
        updated = previous + self._leaks * (np.tanh(total) - previous)
        gates = Maturation(self.maturation_steps).advance(self._gates)
        self._state = dict(zip(self._dynamic_keys, map(float, updated)))
        self._gates = gates
        return {key: self._state[key] for key in self.output_keys}

    def mature(self, ticks=1):
        """Advance connection maturity only. step() already advances one tick."""
        self._gates = Maturation(self.maturation_steps).advance(self._gates, ticks)

    def set_delta(self, source: str, target: str, delta: float) -> float:
        """Set a learned weight offset, clipped to the remaining recurrent budget.

        The library does not prescribe a reward/gradient learning rule. Callers
        own their traces/optimizers; this API retains learned offsets across growth.
        """
        key = source, target
        if key not in self._base:
            raise KeyError(key)
        if not isfinite(delta) or not isfinite(self._base[key] + delta):
            raise ValueError("learned weight must be finite")
        if self.recurrent_limit is not None and self._roles[source] != "input":
            used = sum(
                abs(self._base[k] + self._delta[k])
                for k in self._base
                if k != key and k[1] == target and self._roles[k[0]] != "input"
            )
            available = max(0.0, self.recurrent_limit - used)
            effective = float(np.clip(self._base[key] + delta, -available, available))
            delta = effective - self._base[key]
        self._delta[key] = float(delta)
        return float(delta)

    def grow(self, candidate: NetworkDefinition) -> GrowthReport:
        """Compatibility shorthand; use Development.grow for a rollback handle."""
        from .development import Development

        report = Development(self).grow(candidate).report
        return GrowthReport(report.added_nodes, report.added_edges)

    def reconfigure(self, candidate: NetworkDefinition, policy=None):
        """Explicit exact topology change. Returns a DevelopmentChange undo handle."""
        from .development import Development

        return Development(self).reconfigure(candidate, policy)

    def restore(self, snapshot):
        """Validate then restore in-place; invalidates outstanding change handles."""
        self._adopt(type(self).from_snapshot(snapshot))

    def _adopt(self, staged):
        # The staged runtime owns independent dicts/arrays and a fresh structure
        # token. No partially migrated state is exposed by a failed preparation.
        self.__dict__.update(staged.__dict__)

    def reset(self):
        """Reset temporal state only; learned offsets and growth maturity are retained."""
        self._state = dict.fromkeys(self._state, 0.0)

    def snapshot(self):
        return {
            "schema_version": 1,
            "network": self.network.to_dict(),
            "options": {
                "recurrent_limit": self.recurrent_limit,
                "maturation_steps": self.maturation_steps,
                "max_nodes": self.max_nodes,
                "max_edges": self.max_edges,
            },
            "state": dict(self._state),
            "weights": [
                {
                    "source": e.source,
                    "target": e.target,
                    "base": self._base[e.key],
                    "delta": self._delta[e.key],
                    "gate": self._gates[e.key],
                }
                for e in self.network.edges
            ],
        }

    @classmethod
    def from_snapshot(cls, data):
        if data["schema_version"] != 1:
            raise ValueError("unsupported RNN snapshot schema")
        brain = cls(NetworkDefinition.from_dict(data["network"]), **data["options"])
        state = data["state"]
        if set(state) != set(brain._state) or any(
            not isfinite(v) or abs(v) > 1 for v in state.values()
        ):
            raise ValueError("invalid saved neural state")
        weights = data["weights"]
        keys = [(w["source"], w["target"]) for w in weights]
        if len(keys) != len(set(keys)) or set(keys) != set(brain._base):
            raise ValueError("saved weights do not match topology")
        base, delta, gates = {}, {}, {}
        for key, item in zip(keys, weights):
            if not all(isfinite(item[k]) for k in ("base", "delta", "gate")):
                raise ValueError("non-finite saved weights")
            if not 0 <= item["gate"] <= 1 or not isfinite(item["base"] + item["delta"]):
                raise ValueError("invalid saved gate or effective weight")
            base[key], delta[key], gates[key] = item["base"], item["delta"], item["gate"]
        if brain.recurrent_limit is not None:
            for target in brain._dynamic_keys:
                amount = sum(
                    abs(base[k] + delta[k])
                    for k in keys
                    if k[1] == target and brain._roles[k[0]] != "input"
                )
                if amount > brain.recurrent_limit + 1e-12:
                    raise ValueError("saved recurrent weights exceed stability budget")
        brain._base, brain._delta, brain._gates = base, delta, gates
        brain._state = {key: float(value) for key, value in state.items()}
        return brain
