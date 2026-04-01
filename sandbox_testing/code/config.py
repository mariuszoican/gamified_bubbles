# config.py
# ============================================================
# THE ONLY FILE YOU EDIT WHEN NEW DATA ARRIVES.
# All three pipeline scripts import this with: from config import *
#
# To add a new session:
#   1. Add the session code to SESSION_CODES
#   2. Update DATE if you have a new oTree export
#   3. Re-run: prepare_panels.py → compute_variables.py → hypothesis_tests.py
# ============================================================

# ── DATA EXPORT ───────────────────────────────────────────────────────────────
DATE = "2026-03-12"          # oTree CSV export date (YYYY-MM-DD)

# ── SESSION CODES ─────────────────────────────────────────────────────────────
SESSION_CODES = [
    "0xs4vw9m",              # Session 1 — 2026-03-12 (pilot, gh only)
    # "XXXXXXXX",            # Session 2
    # "XXXXXXXX",            # Session 3
    # "XXXXXXXX",            # Session 4
    # "XXXXXXXX",            # Session 5
    # "XXXXXXXX",            # Session 6
    # "XXXXXXXX",            # Session 7
    # "XXXXXXXX",            # Session 8
]

# ── EXPERIMENT PARAMETERS (fixed by design — do not change) ──────────────────
COMPLETION_PAGE    = "FinalForProlific"
N_PERIODS          = 15          # trading periods per market
N_MARKETS          = 2           # repetitions per group
N_TRADERS          = 8           # participants per market (full sessions)
N_OTREE_ROUNDS     = 30          # total oTree rounds = N_PERIODS * N_MARKETS
DIV_VALUES         = [0, 4, 8, 20]
EXP_DIV            = 8
INITIAL_FV         = 120         # fundamental value at period 1: 8 * 15
BASELINE_TREATMENT = "nh"        # regression baseline: non-gamified human
ALGO_PROBABILITY   = 0.20
BADGE_THRESHOLD    = 8           # min score for trader-type classification
FORECAST_BONUS_TOL = 0.05        # within 5% earns bonus

# ── PATHS ─────────────────────────────────────────────────────────────────────
# NOTE: raw data lives in a subfolder (marius_play_1) inside data/.
# Update DATA_SUBDIR when a new session folder is added.
DATA_SUBDIR = "marius_play_1"
DATA_DIR    = f"../data/{DATA_SUBDIR}"    # raw oTree exports (READ-ONLY)
OUTPUT_DIR  = "../generated_data"         # processed outputs (scripts write here)

# ── CRITICAL: subsession.round_number ENCODES BOTH PERIOD AND MARKET ─────────
# The main oTree table has N_PERIODS * N_MARKETS = 30 rows per participant.
# subsession.round_number goes 1–30:
#   trading_day = ((round_number - 1) % N_PERIODS) + 1      → 1 to 15
#   market_rep  = ceil(round_number / N_PERIODS)             → 1 or 2
#              = 1 if round_number <= N_PERIODS else 2
# These derived columns must be created in prepare_panels.py immediately
# after renaming. DO NOT treat subsession.round_number as market_rep directly.

# ── CLOSING PRICE NOTE ────────────────────────────────────────────────────────
# group.closing_price in the main oTree table is mostly populated (174/180)
# but is NaN for periods with no trades (e.g. period 1 of market 1).
# Primary source for closing prices: compute from MBO (last trade per period).
# Use group.closing_price as a cross-check only.

# ── COLUMN RENAME DICT ────────────────────────────────────────────────────────
# Apply at load time: df = df.rename(columns=RENAME, errors="ignore")
# errors="ignore" → new oTree columns won't break anything.
# NOTE: subsession.round_number is renamed to otree_round (NOT market_rep).
#       market_rep and trading_day are DERIVED from otree_round (see above).
RENAME = {
    # participant level
    "participant.code":                          "participant_code",
    "participant.payoff":                        "payoff_total",
    "participant._is_bot":                       "is_bot",
    "participant._current_page_name":            "current_page",
    "participant.time_started_utc":              "time_started",
    # player level — main trading app
    "player.id_in_group":                        "trader_id",
    "player.trader_uuid":                        "trader_uuid",
    "player.current_cash":                       "cash_end",
    "player.num_shares":                         "shares_end",
    "player.dividend_per_share":                 "div_per_share",
    "player.dividend_cash":                      "div_cash",
    "player.cash_after_dividend":                "cash_after_div",
    "player.assigned_initial_cash":              "initial_cash",
    "player.assigned_initial_shares":            "initial_shares",
    "player.forecast_price_next_day":            "forecast_price",
    "player.forecast_confidence_next_day":       "forecast_confidence",
    "player.realized_next_day_closing_price":    "realized_price",
    "player.forecast_bonus_earned":              "forecast_bonus",
    "player.forecast_bonus_scored":              "forecast_hit",
    # player level — post-exp
    "player.num_correct_answers":                "fin_lit_score",
    "player.num_quiz_questions":                 "n_quiz_questions",
    "player.age":                                "age",
    "player.gender":                             "gender",
    "player.trading_experience":                 "trading_experience",
    "player.course_financial":                   "finance_course",
    "player.payoff_for_quiz":                    "quiz_payoff",
    "player.payable_market":                     "payable_market",
    # player level — intro (self_assesment is here, NOT in post_exp)
    "player.self_assesment":                     "self_assess",   # note: source typo preserved
    # group level
    "group.market_design":                       "market_design",
    "group.group_composition":                   "group_composition",
    "group.treatment":                           "treatment",
    "group.trading_session_uuid":                "session_uuid",
    "group.closing_price":                       "closing_price_otree",  # renamed to avoid confusion
    "group.noise_trader_present":                "algo_present",
    "group.noise_trader_draw":                   "algo_draw",
    "group.num_days":                            "n_periods_group",
    # subsession — renamed to otree_round, NOT market_rep
    "subsession.round_number":                   "otree_round",
    "session.code":                              "session_code",
}

# ── CONTROLS VECTOR ───────────────────────────────────────────────────────────
CONTROLS = [
    "fin_lit_score",
    "self_assess",
    "age",
    "overconfidence",
    "trading_experience",
    "is_rep2",
]

CONTROLS_TO_STANDARDISE = [
    "fin_lit_score",
    "self_assess",
    "age",
    "overconfidence",
]
# NOTE: market_rep / is_rep2 is binary — do not standardise.
