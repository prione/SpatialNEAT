"""Connection maturation is a time policy, not a decision to grow or learn."""

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class Maturation:
    steps: int = 8

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("maturation steps must be a nonnegative integer")

    def advance(self, gates, ticks=1):
        """Return a new gate mapping; zero ticks always leaves maturity unchanged."""
        if type(ticks) is not int or ticks < 0:
            raise ValueError("maturation ticks must be a nonnegative integer")
        if any(not isfinite(v) or not 0 <= v <= 1 for v in gates.values()):
            raise ValueError("maturity gates must be finite and in [0, 1]")
        if not ticks:
            return dict(gates)
        if not self.steps:
            return dict.fromkeys(gates, 1.0)
        return {key: min(1.0, value + ticks / self.steps) for key, value in gates.items()}

    def _advance_validated(self, gates, active, ticks=1):
        """Runtime-only array path; gates were validated at creation/restoration.

        Return proposed values before mutation. Fully mature connections need no
        scan. The public mapping API above continues to validate arbitrary data.
        """
        if type(ticks) is not int or ticks < 0:
            raise ValueError("maturation ticks must be a nonnegative integer")
        if not ticks or not active.size:
            return None
        values = (
            np.minimum(1.0, gates[active] + ticks / self.steps)
            if self.steps
            else np.ones(active.size)
        )
        return values, active[values < 1.0]
