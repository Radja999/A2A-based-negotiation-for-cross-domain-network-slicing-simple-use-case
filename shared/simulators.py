"""Per-domain physics simulators.

Each class models only its own domain's physics.  They are never shared across
agents — RANSimulator lives inside the RAN-agent's tool closure, EdgeSimulator
inside the Edge-agent's.  Latency is additive: E2E = L_ran + L_edge.
"""
import numpy as np

from shared.config import (
    RAN_K, RAN_BW_BOUNDS, RAN_BW_AVAIL_RANGE,
    EDGE_C, EDGE_F_BOUNDS, EDGE_F_AVAIL_RANGE,
)

# Map load level to which third of the availability range to sample from.
# "high" load  → lower third  (tight resources)
# "moderate"   → middle third
# "low"        → upper third  (generous resources)
_BAND_INDEX = {"low": 2, "moderate": 1, "high": 0}


class RANSimulator:
    """RAN domain physics.

    Control knob : bandwidth B (MHz)
    Latency model: L_ran   = RAN_K / B          (ms)
    Cost model   : E_ran   = (B / 20) * 10      (Watts)
    """

    def __init__(self) -> None:
        self.bw_available_max: float = RAN_BW_AVAIL_RANGE[1]
        self.load_level: str = "moderate"

    # ------------------------------------------------------------------
    def reset_episode(self, rng: np.random.Generator, load_level: str) -> None:
        """Sample a new per-episode available-max bandwidth.

        High load → sample from the lower third of RAN_BW_AVAIL_RANGE.
        Low  load → sample from the upper third.
        """
        lo, hi = RAN_BW_AVAIL_RANGE
        band_size = (hi - lo) / 3.0
        base = lo + _BAND_INDEX[load_level] * band_size
        self.bw_available_max = float(
            np.clip(rng.uniform(base, base + band_size), lo, hi)
        )
        self.load_level = load_level

    # ------------------------------------------------------------------
    def latency_for_bw(self, B: float) -> float:
        """L_ran = RAN_K / B  (ms).  Returns inf for B ≤ 0."""
        return RAN_K / B if B > 0 else float("inf")

    def cost_for_bw(self, B: float) -> float:
        """Energy consumption: E = (B / 20) * 10  (Watts)."""
        return (B / 20.0) * 10.0

    def min_latency(self) -> float:
        """Lowest achievable latency at current available-max bandwidth."""
        return self.latency_for_bw(self.bw_available_max)

    def bw_for_latency(self, L: float) -> float:
        """Inverse: B that achieves exactly L ms.

        Clamped to [RAN_BW_BOUNDS[0], bw_available_max].
        If L ≤ 0, returns available max (cheapest to request more time).
        """
        if L <= 0:
            return self.bw_available_max
        B_raw = RAN_K / L
        return float(np.clip(B_raw, RAN_BW_BOUNDS[0], self.bw_available_max))

    def get_state(self) -> dict:
        """Private state snapshot (never forwarded to peers)."""
        return {
            "load_level":           self.load_level,
            "bw_available_max_mhz": round(self.bw_available_max, 3),
            "min_latency_ms":       round(self.min_latency(), 3),
            "bw_bounds_mhz":        RAN_BW_BOUNDS,
            "bw_avail_range_mhz":   RAN_BW_AVAIL_RANGE,
        }


class EdgeSimulator:
    """Edge domain physics (symmetric to RAN).

    Control knob : CPU frequency f (GHz)
    Latency model: L_edge  = EDGE_C / f          (ms)
    Cost model   : C_edge  = f                   (GHz — frequency IS the cost)
    """

    def __init__(self) -> None:
        self.f_available_max: float = EDGE_F_AVAIL_RANGE[1]
        self.load_level: str = "moderate"

    # ------------------------------------------------------------------
    def reset_episode(self, rng: np.random.Generator, load_level: str) -> None:
        """Sample a new per-episode available-max CPU frequency.

        High load → lower third of EDGE_F_AVAIL_RANGE.
        Low  load → upper third.
        """
        lo, hi = EDGE_F_AVAIL_RANGE
        band_size = (hi - lo) / 3.0
        base = lo + _BAND_INDEX[load_level] * band_size
        self.f_available_max = float(
            np.clip(rng.uniform(base, base + band_size), lo, hi)
        )
        self.load_level = load_level

    # ------------------------------------------------------------------
    def latency_for_freq(self, f: float) -> float:
        """L_edge = EDGE_C / f  (ms).  Returns inf for f ≤ 0."""
        return EDGE_C / f if f > 0 else float("inf")

    def cost_for_freq(self, f: float) -> float:
        """Cost = f (GHz).  Allocated frequency is the cost."""
        return float(f)

    def min_latency(self) -> float:
        """Lowest achievable latency at current available-max frequency."""
        return self.latency_for_freq(self.f_available_max)

    def freq_for_latency(self, L: float) -> float:
        """Inverse: f that achieves exactly L ms.

        Clamped to [EDGE_F_BOUNDS[0], f_available_max].
        If L ≤ 0, returns available max.
        """
        if L <= 0:
            return self.f_available_max
        f_raw = EDGE_C / L
        return float(np.clip(f_raw, EDGE_F_BOUNDS[0], self.f_available_max))

    def get_state(self) -> dict:
        """Private state snapshot (never forwarded to peers)."""
        return {
            "load_level":            self.load_level,
            "freq_available_max_ghz": round(self.f_available_max, 3),
            "min_latency_ms":        round(self.min_latency(), 3),
            "freq_bounds_ghz":       EDGE_F_BOUNDS,
            "freq_avail_range_ghz":  EDGE_F_AVAIL_RANGE,
        }
