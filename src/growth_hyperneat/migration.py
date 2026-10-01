"""Pure topology/state migration: no simulation, fitness or growth decision rules."""

from dataclasses import dataclass
from math import fsum, isfinite

from .model import NetworkDefinition


@dataclass(frozen=True)
class StateTransfer:
    """Destination state is a convex combination of same-role source states."""

    target: str
    sources: tuple[tuple[str, float], ...]

    def __post_init__(self):
        object.__setattr__(self, "sources", tuple((k, float(c)) for k, c in self.sources))
        keys, coefficients = zip(*self.sources) if self.sources else ((), ())
        if not self.target or not keys or len(keys) != len(set(keys)):
            raise ValueError("state transfer needs unique source IDs and a destination")
        if any(not isfinite(c) or c < 0 for c in coefficients):
            raise ValueError("state coefficients must be finite and nonnegative")
        if abs(_sum(coefficients) - 1.0) > 1e-12:
            raise ValueError("state coefficients must sum to one")


@dataclass(frozen=True)
class WeightTransfer:
    """Project old edges onto one new edge, retaining base and learned components.

    If old gates differ, their current values are folded into the projected
    weights. That edge starts fully mature; future old maturation is not retained.
    """

    source: str
    target: str
    contributors: tuple[tuple[str, str, float], ...]

    def __post_init__(self):
        object.__setattr__(
            self, "contributors", tuple((a, b, float(c)) for a, b, c in self.contributors)
        )
        keys = [(a, b) for a, b, _ in self.contributors]
        if not self.source or not self.target or not keys or len(keys) != len(set(keys)):
            raise ValueError("weight transfer needs unique contributors and endpoints")
        if any(not isfinite(c) or c < 0 for _, _, c in self.contributors):
            raise ValueError("weight coefficients must be finite and nonnegative")

    @property
    def key(self):
        return self.source, self.target


@dataclass(frozen=True)
class MigrationPolicy:
    weight_mode: str = "preserve"
    preserve_learning: bool = True
    allow_port_changes: bool = False
    require_output_paths: bool = False
    states: tuple[StateTransfer, ...] = ()
    weights: tuple[WeightTransfer, ...] = ()
    reexpress_edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        for name in ("states", "weights"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "reexpress_edges", tuple(tuple(k) for k in self.reexpress_edges))
        if self.weight_mode not in ("preserve", "reexpress"):
            raise ValueError("weight_mode must be preserve or reexpress")
        for name in ("preserve_learning", "allow_port_changes", "require_output_paths"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for keys in (
            [s.target for s in self.states],
            [w.key for w in self.weights],
            self.reexpress_edges,
        ):
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate migration destination")
        if self.weights and (self.weight_mode != "preserve" or self.reexpress_edges):
            raise ValueError("weight projection and re-expression are separate policies")


@dataclass(frozen=True)
class TopologyHealth:
    unreachable_outputs: tuple[str, ...]
    inactive_hidden: tuple[str, ...]


def topology_health(network: NetworkDefinition) -> TopologyHealth:
    """Potential structural paths, including dormant/zero edges; not a skill test."""
    forward, backward = {}, {}
    for edge in network.edges:
        forward.setdefault(edge.source, set()).add(edge.target)
        backward.setdefault(edge.target, set()).add(edge.source)

    def reach(starts, adjacency):
        stack = list(starts)
        seen = set(stack)
        while stack:
            for key in adjacency.get(stack.pop(), ()):
                if key not in seen:
                    seen.add(key)
                    stack.append(key)
        return seen

    incoming = reach((n.key for n in network.nodes if n.role == "input"), forward)
    outgoing = reach((n.key for n in network.nodes if n.role == "output"), backward)
    return TopologyHealth(
        tuple(n.key for n in network.nodes if n.role == "output" and n.key not in incoming),
        tuple(
            n.key for n in network.nodes if n.role == "hidden" and n.key not in incoming & outgoing
        ),
    )


@dataclass(frozen=True)
class DevelopmentReport:
    added_nodes: tuple[str, ...]
    removed_nodes: tuple[str, ...]
    moved_nodes: tuple[str, ...]
    retuned_nodes: tuple[str, ...]
    added_edges: tuple[tuple[str, str], ...]
    removed_edges: tuple[tuple[str, str], ...]
    reexpressed_edges: tuple[tuple[str, str], ...]
    normalized_edges: tuple[tuple[str, str], ...]
    transferred_nodes: tuple[str, ...]
    transferred_edges: tuple[tuple[str, str], ...]
    health: TopologyHealth


def migrate_snapshot(before, candidate: NetworkDefinition, policy: MigrationPolicy):
    """Build a fresh proposed runtime snapshot; never modify the source snapshot."""
    old = NetworkDefinition.from_dict(before["network"])
    previous = {n.key: n for n in old.nodes}
    current = {n.key: n for n in candidate.nodes}
    for key in current.keys() & previous.keys():
        if current[key].role != previous[key].role:
            raise ValueError("existing node role cannot change; use a distinct ID")
    if len(old.nodes[0].position) != len(candidate.nodes[0].position):
        raise ValueError("development cannot change coordinate dimensionality")
    if not policy.allow_port_changes:
        for role in ("input", "output"):
            if tuple(n.key for n in old.nodes if n.role == role) != tuple(
                n.key for n in candidate.nodes if n.role == role
            ):
                raise ValueError("input/output changes require allow_port_changes=True")
    health = topology_health(candidate)
    if policy.require_output_paths and health.unreachable_outputs:
        raise ValueError("candidate disconnects an output from all inputs")

    state = {n.key: before["state"].get(n.key, 0.0) for n in candidate.nodes if n.role != "input"}
    for transfer in policy.states:
        if transfer.target not in state:
            raise ValueError("state transfer destination is not a dynamic node")
        role = current[transfer.target].role
        if any(
            key not in before["state"] or previous[key].role != role for key, _ in transfer.sources
        ):
            raise ValueError("state transfers require existing same-role sources")
        state[transfer.target] = max(
            -1.0, min(1.0, _sum(before["state"][k] * c for k, c in transfer.sources))
        )

    old_weights = {(w["source"], w["target"]): w for w in before["weights"]}
    edge_keys = {e.key for e in candidate.edges}
    transfers = {w.key: w for w in policy.weights}
    if not transfers.keys() <= edge_keys or not set(policy.reexpress_edges) <= edge_keys:
        raise ValueError("weight migration destination does not exist")
    weights, mutable, reexpressed = {}, set(), set()
    for edge in candidate.edges:
        key = edge.key
        if key in transfers:
            contributors = transfers[key].contributors
            if any((a, b) not in old_weights for a, b, _ in contributors):
                raise ValueError("weight contributor does not exist")
            gates = {old_weights[a, b]["gate"] for a, b, _ in contributors}
            gate = next(iter(gates)) if len(gates) == 1 else 1.0
            projected = {}
            for component in ("base", "delta"):
                projected[component] = _sum(
                    old_weights[a, b][component]
                    * c
                    * (1.0 if len(gates) == 1 else old_weights[a, b]["gate"])
                    for a, b, c in contributors
                )
            weights[key] = {**projected, "gate": gate}
            mutable.add(key)
        elif key in old_weights:
            weights[key] = {f: old_weights[key][f] for f in ("base", "delta", "gate")}
        else:
            weights[key] = {
                "base": float(edge.weight),
                "delta": 0.0,
                "gate": 0.0 if before["options"]["maturation_steps"] else 1.0,
            }
            mutable.add(key)
        if policy.weight_mode == "reexpress" or key in policy.reexpress_edges:
            weights[key]["base"] = float(edge.weight)
            mutable.add(key)
            if key in old_weights:
                reexpressed.add(key)
        if not policy.preserve_learning:
            weights[key]["delta"] = 0.0
            if key in old_weights:
                mutable.add(key)

    normalized = []
    limit = before["options"]["recurrent_limit"]
    if limit is not None:
        rows = {}
        for edge in candidate.edges:
            if current[edge.source].role != "input":
                rows.setdefault(edge.target, []).append(edge.key)
        for keys in rows.values():
            effective = {k: weights[k]["base"] + weights[k]["delta"] for k in keys}
            kept = _sum(abs(effective[k]) for k in keys if k not in mutable)
            amount = _sum(abs(effective[k]) for k in keys if k in mutable)
            available = max(0.0, limit - kept)
            if amount > available:
                for key in keys:
                    if key in mutable:
                        # Retain the learned offset; only the proposed base changes.
                        weights[key]["base"] = (
                            effective[key] * available / amount - weights[key]["delta"]
                        )
                        normalized.append(key)

    after = {
        "schema_version": 1,
        "network": candidate.to_dict(),
        "options": dict(before["options"]),
        "state": state,
        "weights": [
            {"source": e.source, "target": e.target, **weights[e.key]} for e in candidate.edges
        ],
    }
    report = DevelopmentReport(
        tuple(n.key for n in candidate.nodes if n.key not in previous),
        tuple(n.key for n in old.nodes if n.key not in current),
        tuple(
            n.key
            for n in candidate.nodes
            if n.key in previous and n.position != previous[n.key].position
        ),
        tuple(
            n.key for n in candidate.nodes if n.key in previous and n.leak != previous[n.key].leak
        ),
        tuple(e.key for e in candidate.edges if e.key not in old_weights),
        tuple(e.key for e in old.edges if e.key not in edge_keys),
        tuple(sorted(reexpressed)),
        tuple(sorted(normalized)),
        tuple(s.target for s in policy.states),
        tuple(w.key for w in policy.weights),
        health,
    )
    return after, report


def _sum(values):
    try:
        result = fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError("migration arithmetic must remain finite") from exc
    if not isfinite(result):
        raise ValueError("migration arithmetic must remain finite")
    return result
