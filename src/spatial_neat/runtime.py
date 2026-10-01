"""Array-backed sparse synchronous RNN; lifecycle operations live in development."""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from .maturation import Maturation
from .model import NetworkDefinition
from .runtime_layout import RuntimeLayout


@dataclass(frozen=True)
class GrowthReport:
    added_nodes: tuple[str, ...]
    added_edges: tuple[tuple[str, str], ...]


class GrowingRNN:
    """All non-input neurons update from the previous timestep, including outputs.

    Arrays own runtime state, weights and maturity. Stable logical keys remain at
    public API/snapshot boundaries; structural changes compile a fresh layout.
    The recurrent L1 bound constrains tanh dynamics without imposing learning rules.
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
        Maturation(maturation_steps)
        if type(max_nodes) is not int or type(max_edges) is not int:
            raise ValueError("runtime budgets must be integers")
        if min(max_nodes, max_edges) < 1:
            raise ValueError("runtime budgets must be positive")
        self.recurrent_limit = recurrent_limit
        self.maturation_steps = maturation_steps
        self.max_nodes, self.max_edges = max_nodes, max_edges
        self._check_budget(network)
        self.network = network
        self._layout = RuntimeLayout.compile(network)
        self._base = np.array([e.weight for e in network.edges], dtype=float)
        self._delta = np.zeros(len(network.edges))
        self._gates = np.ones(len(network.edges))
        self._state = np.zeros(len(self._layout.dynamic_keys))
        self._values = np.empty(len(network.nodes))
        if recurrent_limit is not None:
            for row in self._layout.recurrent_rows:
                amount = sum(abs(float(self._base[i])) for i in row)
                if amount > recurrent_limit:
                    self._base[row] *= recurrent_limit / amount
        self._refresh_weights()
        self._structure_token = object()

    def _refresh_weights(self):
        self._effective = self._base + self._delta
        self._weighted = self._effective * self._gates
        self._maturing = np.flatnonzero(self._gates < 1.0)

    @staticmethod
    def _mapping(keys, values):
        return dict(zip(keys, map(float, values)))

    @property
    def state(self):
        return self._mapping(self._layout.dynamic_keys, self._state)

    @property
    def base_weights(self):
        return self._mapping(self._layout.edge_keys, self._base)

    @property
    def learned_deltas(self):
        return self._mapping(self._layout.edge_keys, self._delta)

    @property
    def gates(self):
        return self._mapping(self._layout.edge_keys, self._gates)

    @property
    def input_keys(self):
        return self._layout.input_keys

    @property
    def output_keys(self):
        return self._layout.output_keys

    @property
    def complexity(self):
        """Structural counts include dormant/growing edges."""
        return {
            "nodes": len(self.network.nodes),
            "hidden": self.network.hidden_count,
            "edges": len(self.network.edges),
        }

    def _check_budget(self, network):
        if len(network.nodes) > self.max_nodes or len(network.edges) > self.max_edges:
            raise ValueError("runtime topology budget exceeded")

    def _commit_maturation(self, proposed):
        if proposed is not None:
            gates, remaining = proposed
            active = self._maturing
            self._gates[active] = gates
            self._weighted[active] = self._effective[active] * gates
            self._maturing = remaining

    def step(self, inputs) -> dict[str, float]:
        layout = self._layout
        if isinstance(inputs, Mapping):
            if set(inputs) != set(layout.input_keys):
                raise ValueError("input mapping must contain exactly all input keys")
            inputs = [inputs[key] for key in layout.input_keys]
        values = np.asarray(inputs, dtype=float)
        if values.shape != (len(layout.input_keys),) or not np.isfinite(values).all():
            raise ValueError("inputs must be a finite vector matching input_keys")
        self._values[layout.input_indices] = values
        self._values[layout.dynamic_indices] = self._state
        with np.errstate(over="ignore", invalid="ignore"):
            total = np.bincount(
                layout.targets,
                weights=self._values[layout.sources] * self._weighted,
                minlength=len(layout.dynamic_keys),
            )
        if not np.isfinite(total).all():
            raise ValueError("non-finite neural accumulation; state was not advanced")
        updated = self._state + layout.leaks * (np.tanh(total) - self._state)
        gates = Maturation(self.maturation_steps)._advance_validated(self._gates, self._maturing)
        # Failed accumulation/maturation leaves both public state and gates unchanged.
        self._state = updated
        self._commit_maturation(gates)
        return self._mapping(layout.output_keys, self._state[layout.output_indices])

    def mature(self, ticks=1):
        """Advance connection maturity only. step() already advances one tick."""
        proposed = Maturation(self.maturation_steps)._advance_validated(
            self._gates, self._maturing, ticks
        )
        self._commit_maturation(proposed)

    def set_delta(self, source: str, target: str, delta: float) -> float:
        """Set an offset, clipped to the remaining recurrent row budget."""
        layout = self._layout
        index = layout.edge_indices[source, target]
        base = float(self._base[index])
        if not isfinite(delta) or not isfinite(base + delta):
            raise ValueError("learned weight must be finite")
        if self.recurrent_limit is not None and layout.roles[source] != "input":
            row = layout.recurrent_rows[layout.targets[index]]
            used = sum(abs(float(self._effective[i])) for i in row if i != index)
            available = max(0.0, self.recurrent_limit - used)
            effective = float(np.clip(base + delta, -available, available))
            delta = effective - base
        self._delta[index] = float(delta)
        self._effective[index] = base + float(delta)
        self._weighted[index] = self._effective[index] * self._gates[index]
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
        # Staged arrays/buffers are independent, with a new structural token.
        self.__dict__.update(staged.__dict__)

    def reset(self):
        """Reset temporal state only; retain learned offsets and maturity."""
        self._state.fill(0.0)

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
            "state": self.state,
            "weights": [
                {
                    "source": a,
                    "target": b,
                    "base": float(self._base[i]),
                    "delta": float(self._delta[i]),
                    "gate": float(self._gates[i]),
                }
                for i, (a, b) in enumerate(self._layout.edge_keys)
            ],
        }

    @classmethod
    def from_snapshot(cls, data):
        if data["schema_version"] != 1:
            raise ValueError("unsupported RNN snapshot schema")
        brain = cls(NetworkDefinition.from_dict(data["network"]), **data["options"])
        state = data["state"]
        if set(state) != set(brain._layout.dynamic_keys) or any(
            not isfinite(v) or abs(v) > 1 for v in state.values()
        ):
            raise ValueError("invalid saved neural state")
        weights = data["weights"]
        keys = [(w["source"], w["target"]) for w in weights]
        if len(keys) != len(set(keys)) or set(keys) != set(brain._layout.edge_keys):
            raise ValueError("saved weights do not match topology")
        for key, item in zip(keys, weights):
            if not all(isfinite(item[k]) for k in ("base", "delta", "gate")):
                raise ValueError("non-finite saved weights")
            if not 0 <= item["gate"] <= 1 or not isfinite(item["base"] + item["delta"]):
                raise ValueError("invalid saved gate or effective weight")
            i = brain._layout.edge_indices[key]
            brain._base[i], brain._delta[i], brain._gates[i] = (
                item["base"],
                item["delta"],
                item["gate"],
            )
        brain._refresh_weights()
        if brain.recurrent_limit is not None:
            for row in brain._layout.recurrent_rows:
                amount = sum(abs(float(brain._effective[i])) for i in row)
                if amount > brain.recurrent_limit + 1e-12:
                    raise ValueError("saved recurrent weights exceed stability budget")
        brain._state = np.array([state[key] for key in brain._layout.dynamic_keys], dtype=float)
        return brain
