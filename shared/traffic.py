"""Correlated load process across episodes.

Replaces i.i.d. per-episode randomness with a structured trajectory so that
the DKB's age-weighting is meaningful and the system can exhibit learning
within recognisable load regimes.
"""
import numpy as np

from shared.config import (
    LOAD_INIT, LOAD_RW_SIGMA, LOAD_REGIME_SHIFT_P, LOAD_THRESHOLDS,
)


class LoadProcess:
    """Global load value x ∈ [0, 1] shared across both simulators.

    Dynamics (per episode tick):
      - With probability LOAD_REGIME_SHIFT_P: abrupt jump to Uniform(0, 1).
      - Otherwise: Gaussian random walk clipped to [0, 1].

    Qualitative bands (LOAD_THRESHOLDS = (lo, hi)):
      x < lo   → "low"
      lo ≤ x ≤ hi → "moderate"
      x > hi   → "high"
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.x: float = LOAD_INIT
        self._rng = rng

    def step(self) -> float:
        """Advance one episode.  Returns new x in [0, 1]."""
        if self._rng.random() < LOAD_REGIME_SHIFT_P:
            self.x = float(self._rng.uniform(0.0, 1.0))
        else:
            self.x = float(
                np.clip(self.x + self._rng.normal(0.0, LOAD_RW_SIGMA), 0.0, 1.0)
            )
        return self.x

    def qualitative(self) -> str:
        """Map current x to a load-level label."""
        lo, hi = LOAD_THRESHOLDS
        if self.x < lo:
            return "low"
        if self.x > hi:
            return "high"
        return "moderate"

    @property
    def value(self) -> float:
        return self.x
