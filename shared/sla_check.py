"""Per-domain SLA pre-checks and cost-minimising local optimisation.

Two layers:
  1. *_precheck   — one-shot check: does a specific resource value meet the
                    assigned latency share (with SLA_SAFETY headroom)?
  2. optimize_*_config — finds the MINIMUM resource value that passes precheck,
                    i.e. the cheapest feasible config.  Uses bisection on the
                    monotone ok(B) / ok(f) function so it is exact for any
                    latency model that is monotone in its knob.

No LLM involvement here.  The LLM uses the result to make the *strategic*
decision (accept vs counter) by comparing returned cost to DKB historical
medians (Section 11 of the blueprint).
"""
from shared.config import RAN_BW_BOUNDS, EDGE_F_BOUNDS, SLA_SAFETY

_BISECT_ITERS = 52   # halves interval 52 times → ~machine-precision convergence


# ─────────────────────────── RAN ────────────────────────────────────────────

def ran_precheck(ransim, B: float, latency_share_ms: float) -> dict:
    """Does bandwidth B meet the latency share with SLA_SAFETY headroom?

    Args:
        ransim           : RANSimulator (provides latency/cost models)
        B                : candidate bandwidth (MHz)
        latency_share_ms : assigned RAN latency budget (ms)

    Returns dict with keys:
        ok             – True iff L_ran <= SLA_SAFETY * latency_share_ms
        ran_latency_ms – predicted RAN latency at B
        energy_w       – energy cost at B (Watts)
    """
    L = ransim.latency_for_bw(B)
    return {
        "ok":             bool(L <= SLA_SAFETY * latency_share_ms),
        "ran_latency_ms": L,
        "energy_w":       ransim.cost_for_bw(B),
    }


def optimize_ran_config(ransim, latency_share_ms: float) -> dict:
    """Find the MINIMUM bandwidth that satisfies the share+margin (cheapest).

    Searches [RAN_BW_BOUNDS[0], bw_available_max] by bisection.
    ok(B) is monotone increasing in B, so the minimum feasible B is the
    unique crossing point from False → True.

    Success return:
        {"feasible": True,
         "bandwidth_mhz":            <float>,
         "predicted_ran_latency_ms": <float>,
         "energy_w":                 <float>}

    Failure return (even max-available BW cannot meet share+margin):
        {"feasible": False,
         "reason":   "even max bandwidth cannot meet share"}
    """
    # Fast infeasibility gate: if max available BW fails, nothing else can help.
    if not ran_precheck(ransim, ransim.bw_available_max, latency_share_ms)["ok"]:
        return {
            "feasible": False,
            "reason":   "even max bandwidth cannot meet share",
        }

    # Bisect for the minimum B in [hard_floor, bw_available_max] where ok=True.
    # Invariant: ok(lo)=False (or lo is the hard floor), ok(hi)=True.
    lo = float(RAN_BW_BOUNDS[0])
    hi = float(ransim.bw_available_max)

    for _ in range(_BISECT_ITERS):
        mid = (lo + hi) * 0.5
        if ran_precheck(ransim, mid, latency_share_ms)["ok"]:
            hi = mid    # mid is feasible — can we go even cheaper?
        else:
            lo = mid    # mid not feasible — need more bandwidth

    # hi is now the infimum of the feasible set; floor to the hard lower bound.
    B_opt = max(hi, float(RAN_BW_BOUNDS[0]))
    # hi always satisfies precheck (bisection invariant). Do NOT round B_opt —
    # rounding down can push it below the exact minimum, breaking precheck.
    info  = ran_precheck(ransim, B_opt, latency_share_ms)
    return {
        "feasible":                 True,
        "bandwidth_mhz":            B_opt,                          # raw float64
        "predicted_ran_latency_ms": round(info["ran_latency_ms"], 4),
        "energy_w":                 round(info["energy_w"], 4),
    }


# ─────────────────────────── Edge ───────────────────────────────────────────

def edge_precheck(edgesim, f: float, latency_share_ms: float) -> dict:
    """Does CPU frequency f meet the latency share with SLA_SAFETY headroom?

    Args:
        edgesim          : EdgeSimulator
        f                : candidate CPU frequency (GHz)
        latency_share_ms : assigned Edge latency budget (ms)

    Returns dict with keys:
        ok              – True iff L_edge <= SLA_SAFETY * latency_share_ms
        edge_latency_ms – predicted Edge latency at f
        freq_cost       – cost (= f; allocated frequency IS the cost)
    """
    L = edgesim.latency_for_freq(f)
    return {
        "ok":              bool(L <= SLA_SAFETY * latency_share_ms),
        "edge_latency_ms": L,
        "freq_cost":       edgesim.cost_for_freq(f),
    }


def optimize_edge_config(edgesim, latency_share_ms: float) -> dict:
    """Find the MINIMUM CPU frequency that satisfies the share+margin (cheapest).

    Since cost = f directly, minimum frequency = minimum cost.

    Success return:
        {"feasible": True,
         "cpu_freq_ghz":              <float>,
         "predicted_edge_latency_ms": <float>,
         "freq_cost":                 <float>}

    Failure return:
        {"feasible": False,
         "reason":   "even max frequency cannot meet share"}
    """
    if not edge_precheck(edgesim, edgesim.f_available_max, latency_share_ms)["ok"]:
        return {
            "feasible": False,
            "reason":   "even max frequency cannot meet share",
        }

    lo = float(EDGE_F_BOUNDS[0])
    hi = float(edgesim.f_available_max)

    for _ in range(_BISECT_ITERS):
        mid = (lo + hi) * 0.5
        if edge_precheck(edgesim, mid, latency_share_ms)["ok"]:
            hi = mid
        else:
            lo = mid

    f_opt = max(hi, float(EDGE_F_BOUNDS[0]))
    # Same as RAN: do not round the knob value — rounding down breaks precheck.
    info  = edge_precheck(edgesim, f_opt, latency_share_ms)
    return {
        "feasible":                  True,
        "cpu_freq_ghz":              f_opt,                           # raw float64
        "predicted_edge_latency_ms": round(info["edge_latency_ms"], 4),
        "freq_cost":                 round(info["freq_cost"], 4),
    }
