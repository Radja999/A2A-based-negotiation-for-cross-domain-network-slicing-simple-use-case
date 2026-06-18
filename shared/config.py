# ---- SLA safety margin (Chergui-style) ----------------------------------------
SLA_SAFETY = 0.9          # agent targets <= 0.9 * assigned share (10% headroom)

# ---- RAN domain (bandwidth knob, minimises energy) ----------------------------
# Calibrated so episodes span easy/hard/infeasible for URLLC:
#   low-load sum_min  ≈ 4.7 ms  (easy, ~5 ms slack vs 10 ms budget)
#   mod-load sum_min  ≈ 6.4 ms  (real tension, ~3.6 ms slack)
#   high-load sum_min ≈ 7–11 ms (hard to occasionally infeasible)
RAN_K               = 60.0          # L_ran  = RAN_K / B  (ms·MHz)
RAN_BW_BOUNDS       = (5.0, 50.0)   # MHz hard bounds
RAN_BW_AVAIL_RANGE  = (12.0, 45.0)  # per-episode random max; high load → ~12 MHz

# ---- Edge domain (cpu-freq knob; freq IS the cost) ----------------------------
EDGE_C              = 175.0         # L_edge = EDGE_C / f  (ms·GHz)
EDGE_F_BOUNDS       = (20.0, 60.0)  # GHz hard bounds
EDGE_F_AVAIL_RANGE  = (28.0, 55.0)  # per-episode random max; high load → ~28 GHz

# ---- Traffic / correlated load ------------------------------------------------
LOAD_INIT           = 0.5           # in [0,1]; 0 = idle, 1 = peak
LOAD_RW_SIGMA       = 0.05          # random-walk stddev between episodes
LOAD_REGIME_SHIFT_P = 0.05          # prob. of an abrupt regime jump
LOAD_THRESHOLDS     = (0.33, 0.67)  # split into low / moderate / high bands

# ---- Negotiation control ------------------------------------------------------
SOFT_COUNTER_LIMIT  = 4             # counter-proposals before forced escalation
MAX_ROUND           = 18            # hard GroupChat message cap (safety net)
MAX_SELF_RETRIES    = 3             # per-agent bounded re-reasoning attempts

# ---- DKB retrieval scoring (Chergui-inspired) ---------------------------------
ALPHA_SIM           = 1.0           # Jaccard similarity weight
BETA_AGE            = 1.0           # time-decay weight: exp(-age / AGE_TAU)
AGE_TAU             = 80.0          # decay constant (episodes)
DELTA_INFLECT       = 1.0           # bonus weight for instructive failures
GAMMA_DIVERSITY     = 0.8           # MMR diversity penalty
RETRIEVE_TOP_K      = 5
K_GOOD              = 3             # "good" entries surfaced from top-K
K_BAD               = 2             # "bad" entries surfaced from top-K
SCORE_GOOD_MIN      = 0.6           # threshold to label an episode "good"
SCORE_BAD_MAX       = 0.35          # threshold to label an episode "bad"

# ---- Outcome scoring (episode -> stored score) --------------------------------
W_SLA               = 1.0           # SLA compliance dominates
W_COST              = 0.4           # lower cost is better
W_ROUNDS            = 0.1           # faster convergence is better

# ---- Cost-greediness ----------------------------------------------------------
# Accept only if optimised cost <= COST_GREEDY_FACTOR * DKB historical median.
# Counter for a larger share otherwise (even if SLA is already met).
COST_GREEDY_FACTOR  = 1.20

# ---- Experiment ---------------------------------------------------------------
N_EPISODES_DEV      = 5             # dev / debug runs (rate-limit friendly)
N_EPISODES_REAL     = 60            # full comparative run
N_EPISODES          = N_EPISODES_DEV  # flip to N_EPISODES_REAL once stable

# ---- LLM provider -------------------------------------------------------------
LLM_PROVIDER        = "groq"        # "groq" | "gemini" | "ollama"
LLM_MAX_RETRIES     = 5             # retry on 429
LLM_BACKOFF_BASE_S  = 2.0           # exponential: 2, 4, 8, 16, 32 s
INTER_EPISODE_SLEEP = 2.0           # seconds between episodes (~30 RPM guard)

# ---- A2A arm (additive; do not modify AutoGen constants above) ----------------
A2A_HOST               = "127.0.0.1"
A2A_PORTS              = {"orchestrator": 9000, "ran": 9001, "edge": 9002}
MAX_PEER_ROUNDS        = 6    # counter limit per peer pair (cf. SOFT_COUNTER_LIMIT)
A2A_PORT_WAIT_TIMEOUT_S = 30  # seconds a2a_run waits for a server port on startup
