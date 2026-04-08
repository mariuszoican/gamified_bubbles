"""
Build trader- and market-day-level panels from oTree experimental data.

Reads raw oTree exports and the custom MBO (message-book-output) file,
then constructs:
  1. trader_day  – trader × market × trading-day panel with demographics,
                   trade counts, market aggregates, wealth, and trader types
  2. market_day  – market × trading-day panel with treatment dummies,
                   price/trade aggregates, trader-type shares, wealth
                   inequality, and cumulative surge/crash/bubble counts
  3. mp          – (intermediate) market-period price and mispricing stats

Output: trader_day_{DATE}.csv and market_day_{DATE}.csv in ``OUT_DIR``.
"""

import numpy as np
import pandas as pd

# ============================================================
# Parameters
# ============================================================
DATE = "2026-04-08"
SESSIONS = ["uik7aoor"]

RAW_DIR = "../raw_data"
OUT_DIR = "../processed_data"

ROUNDS_PER_REPETITION = 15  # trading days per repetition
DIVIDEND_PER_PERIOD = 8  # constant per-period dividend
BUBBLE_SURGE_SIGMA = 2  # σ threshold for surge/crash/bubble flags
TRADER_TYPE_THRESHOLD = 5  # min flag count to classify trader type

# ============================================================
# 1. Load raw oTree exports
# ============================================================


def load_raw(date: str, sessions: list[str]) -> tuple:
    """Read CSVs and keep only rows from target sessions."""
    intro = pd.read_csv(f"{RAW_DIR}/intro_{date}.csv")
    intro = intro[intro["session.code"].isin(sessions)]

    post_exp = pd.read_csv(f"{RAW_DIR}/post_exp_{date}.csv")
    post_exp = post_exp[post_exp["session.code"].isin(sessions)]

    app = pd.read_csv(f"{RAW_DIR}/trader_bridge_app_{date}.csv")
    app = app[app["session.code"].isin(sessions)]
    app = app[app["participant._current_page_name"] == "FinalForProlific"]

    mbo = pd.read_csv(f"{RAW_DIR}/trader_bridge_app_custom_export_mbo_{date}.csv")

    return intro, post_exp, app, mbo


intro, post_exp, app, mbo = load_raw(DATE, SESSIONS)


# ============================================================
# 2. Build trader × market × day panel
# ============================================================


def fundamental_value(trading_day: pd.Series) -> pd.Series:
    """Remaining expected dividends: D × (T − t)."""
    return DIVIDEND_PER_PERIOD * (ROUNDS_PER_REPETITION + 1 - trading_day)


# --- 2a. Core app data: rename and select --------------------------

APP_COLUMN_MAP = {
    "session.code": "session_code",
    "participant.code": "participant_code",
    "participant.payoff": "payoff",
    "player.trader_uuid": "trader_uuid",
    "player.assigned_initial_cash": "initial_cash",
    "player.forecast_price_next_day": "forecast",
    "player.forecast_confidence_next_day": "forecast_confidence",
    "subsession.round_number": "trading_day",
    "group.noise_trader_present": "algo_present",
    "group.market_design": "gamified",
    "group.group_composition": "hybrid",
    "player.algo_belief_present": "algorithm_belief",
    "player.algo_belief_confidence": "algorithm_belief_confidence",
    "player.num_shares": "num_shares",
    "player.current_cash": "current_cash",
    "group.trading_session_uuid": "market_uuid",
}

TRADER_DAY_COLS = [
    "session_code",
    "participant_code",
    "market_uuid",
    "trader_uuid",
    "gamified",
    "hybrid",
    "algo_present",
    "trading_day",
    "payoff",
    "initial_cash",
    "forecast",
    "forecast_confidence",
    "current_cash",
    "num_shares",
    "algorithm_belief",
]

trader_day = app.rename(columns=APP_COLUMN_MAP)[TRADER_DAY_COLS].copy()

# --- 2b. Post-experiment survey & demographics ---------------------

post_exp = post_exp.merge(
    intro[["participant.code", "player.self_assesment", "player.cq_attempt_count"]],
    on="participant.code",
)

POST_COLUMN_MAP = {
    "participant.code": "participant_code",
    "player.payoff_for_trade": "trade_payoff",
    "player.gender": "gender",
    "player.age": "age",
    "player.course_financial": "finance_course",
    "player.trading_experience": "trading_experience",
    "player.self_assesment": "self_assessment",
    "player.cq_attempt_count": "cq_attempt_count",
}
post_exp = post_exp.rename(columns=POST_COLUMN_MAP)

post_exp["gender_female"] = (post_exp["gender"] == "Female").astype(int)
post_exp["fin_quiz_score"] = (
    post_exp["player.num_correct_answers"] / post_exp["player.num_quiz_questions"]
)
post_exp["high_education"] = (
    post_exp["player.education"]
    .isin(
        [
            "MBA",
            "PhD",
            "master",
            "undergraduate: 1st year",
            "undergraduate: 2nd year",
            "undergraduate: 3rd year",
            "undergraduate: 4th year",
        ]
    )
    .astype(int)
)
post_exp["overconfidence"] = (
    post_exp["self_assessment"] / 10 - post_exp["fin_quiz_score"]
)

DEMOG_COLS = [
    "participant_code",
    "trade_payoff",
    "fin_quiz_score",
    "self_assessment",
    "overconfidence",
    "cq_attempt_count",
    "gender_female",
    "age",
    "finance_course",
    "trading_experience",
    "high_education",
]

trader_day = trader_day.merge(post_exp[DEMOG_COLS], on="participant_code")

# --- 2c. Encode treatment dummies and repetition -------------------

trader_day["gamified"] = (trader_day["gamified"] == "gamified").astype(int)
trader_day["hybrid"] = (trader_day["hybrid"] != "human_only").astype(int)
trader_day["repetition"] = np.where(
    trader_day["trading_day"] <= ROUNDS_PER_REPETITION, 1, 2
)
trader_day["trading_day"] = np.where(
    trader_day["trading_day"] > ROUNDS_PER_REPETITION,
    trader_day["trading_day"] - ROUNDS_PER_REPETITION,
    trader_day["trading_day"],
)
trader_day["algorithm_belief"] = np.where(
    trader_day["hybrid"] == 1,
    (trader_day["algorithm_belief"] == "yes").astype(int),
    np.nan,
)
trader_day["algorithm_belief"] = trader_day.groupby(
    ["market_uuid", "participant_code"]
)["algorithm_belief"].transform("max")

# ============================================================
# 3. Build trade-level and market-period panels
# ============================================================

# --- 3a. Trade-level panel -----------------------------------------

trades = mbo[mbo["record_kind"] == "trade"].copy()
trades = trades.rename(
    columns={
        "bid_trader_uuid": "buyer_uuid",
        "ask_trader_uuid": "seller_uuid",
        "trading_session_uuid": "market_uuid",
        "market_number": "repetition",
    }
)
trades["fundamental_value"] = fundamental_value(trades["trading_day"])
trades["event_ts"] = pd.to_datetime(trades["event_ts"])
trades["diff_time"] = (
    trades.groupby("market_uuid")["event_ts"].diff().dt.total_seconds().shift(-1)
)
trades["mispricing"] = trades["price"] - trades["fundamental_value"]
trades["abs_mispricing"] = trades["mispricing"].abs()

TRADE_COLS = [
    "market_uuid",
    "repetition",
    "trading_day",
    "event_ts",
    "diff_time",
    "buyer_uuid",
    "seller_uuid",
    "aggressor_side",
    "price",
    "fundamental_value",
    "mispricing",
    "abs_mispricing",
]
trades = trades[TRADE_COLS].reset_index(drop=True)

# --- 3b. Market-period aggregates -----------------------------------

mp = (
    trades.groupby(["market_uuid", "repetition", "trading_day"])
    .agg(
        n_trades_market=("price", "count"),
        avg_trade_price=("price", "mean"),
        avg_mispricing=("mispricing", "mean"),
        avg_abs_mispricing=("abs_mispricing", "mean"),
        closing_price=("price", "last"),
        opening_price=("price", "first"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        fundamental_value=("fundamental_value", "first"),
    )
    .reset_index()
)

# fill in zero-trade periods from trader_day skeleton
mp = mp.merge(
    trader_day[["market_uuid", "repetition", "trading_day"]].drop_duplicates(),
    how="outer",
)
mp["n_trades_market"] = mp["n_trades_market"].fillna(0)
mp["fundamental_value"] = fundamental_value(mp["trading_day"])
mp["closing_price"] = mp.groupby("market_uuid")["closing_price"].ffill()

# price lags and returns
mp["price_L1"] = mp.groupby("market_uuid")["closing_price"].shift(1)
mp["price_L2"] = mp.groupby("market_uuid")["closing_price"].shift(2)
mp["price_next"] = mp.groupby("market_uuid")["closing_price"].shift(-1)
mp["return"] = mp.groupby("market_uuid")["closing_price"].pct_change()
mp["abs_mispricing_ratio"] = mp["avg_abs_mispricing"] / mp["fundamental_value"]

# --- 3c. Surge, crash, and bubble flags (Asparouhova 2024; Noussair 2001)


def flag_extremes(series: pd.Series, group_key: str, sigma: float = BUBBLE_SURGE_SIGMA):
    """Return +1 / −1 flags for observations beyond ±σ from group mean."""
    mu = series.groupby(group_key).transform("mean")
    sd = series.groupby(group_key).transform("std")
    high = (series > mu + sigma * sd).astype(int)
    low = (series < mu - sigma * sd).astype(int)
    return high, low


mp["surge"], mp["crash"] = flag_extremes(mp["return"], mp["repetition"])

mp["normalized_mispricing"] = (mp["avg_trade_price"] - mp["fundamental_value"]) / mp[
    "fundamental_value"
]
mp["bubble_period"], _ = flag_extremes(mp["normalized_mispricing"], mp["repetition"])
mp["bubble_start"] = (
    (mp["bubble_period"] == 1)
    & (
        mp.groupby(["market_uuid", "repetition"])["bubble_period"].shift(1).fillna(0)
        == 0
    )
).astype(int)


# ============================================================
# 4. Merge trade counts and market aggregates into trader_day
# ============================================================


def count_trades_by_side(trades: pd.DataFrame) -> pd.DataFrame:
    """Count buys and sells per trader × market × day."""
    buys = (
        trades.groupby(["market_uuid", "trading_day", "buyer_uuid"])
        .size()
        .reset_index(name="n_buys")
        .rename(columns={"buyer_uuid": "trader_uuid"})
    )
    sells = (
        trades.groupby(["market_uuid", "trading_day", "seller_uuid"])
        .size()
        .reset_index(name="n_sells")
        .rename(columns={"seller_uuid": "trader_uuid"})
    )
    out = buys.merge(
        sells, how="outer", on=["market_uuid", "trader_uuid", "trading_day"]
    ).fillna(0)
    out["net_position"] = out["n_buys"] - out["n_sells"]
    return out


n_trades = count_trades_by_side(trades)

trader_day = trader_day.merge(
    n_trades, how="left", on=["market_uuid", "trader_uuid", "trading_day"]
)
for col in ("n_buys", "n_sells", "net_position"):
    trader_day[col] = trader_day[col].fillna(0)

trader_day = trader_day.merge(
    mp, how="left", on=["market_uuid", "repetition", "trading_day"]
)


# ============================================================
# 5. Wealth and inequality
# ============================================================

trader_day["wealth_day"] = (
    trader_day["current_cash"]
    + trader_day["num_shares"] * trader_day["fundamental_value"]
)


def gini(x: pd.Series) -> float:
    """Gini coefficient for a wealth vector."""
    vals = pd.Series(x).dropna().astype(float).values
    if len(vals) == 0:
        return np.nan
    mu = vals.mean()
    if mu == 0 or np.isnan(mu):
        return 0.0 if np.allclose(vals, 0) else np.nan
    return np.abs(vals[:, None] - vals[None, :]).sum() / (2 * len(vals) ** 2 * mu)


trader_day["gini"] = trader_day.groupby(["market_uuid", "trading_day"])[
    "wealth_day"
].transform(gini)


# ============================================================
# 6. Trader-type classification
# ============================================================

TYPE_FLAGS = ["feedback_flag", "speculator_flag", "fundamental_flag"]

temp = trader_day[
    [
        "participant_code",
        "market_uuid",
        "trading_day",
        "num_shares",
        "net_position",
        "closing_price",
        "price_L1",
        "price_L2",
        "price_next",
        "fundamental_value",
    ]
].copy()

# feedback trader: trades in direction of lagged price change
temp["feedback_flag"] = (
    (temp["net_position"] * (temp["price_L1"] - temp["price_L2"])) > 0
).astype(int)

# speculator: trades in direction of next-period price change
temp["speculator_flag"] = (
    (temp["net_position"] * (temp["price_next"] - temp["closing_price"])) > 0
).astype(int)

# fundamentalist: trades against mispricing
temp["fundamental_flag"] = (
    (temp["net_position"] * (temp["closing_price"] - temp["fundamental_value"])) < 0
).astype(int)

# aggregate flags per trader × market and classify
type_counts = temp.groupby(["market_uuid", "participant_code"])[TYPE_FLAGS].sum()


def classify_trader(row: pd.Series) -> pd.Series:
    """Assign trader to the dominant type; split ties equally."""
    if row.max() < TRADER_TYPE_THRESHOLD:
        return pd.Series({f: 0.0 for f in TYPE_FLAGS})
    winners = row[row == row.max()].index
    weight = 1.0 / len(winners)
    return pd.Series({f: weight if f in winners else 0.0 for f in TYPE_FLAGS})


result_types = type_counts.apply(classify_trader, axis=1).reset_index()
result_types["other_flag"] = 1 - result_types[TYPE_FLAGS].sum(axis=1)

trader_day = trader_day.merge(
    result_types, how="left", on=["market_uuid", "participant_code"]
)

# ============================================================
# 7. Market-day panel (one row per market × trading_day)
# ============================================================

# columns constant within a market (treatment assignment)
MARKET_LEVEL_COLS = ["market_uuid", "repetition", "gamified", "hybrid", "algo_present"]

market_id = trader_day[MARKET_LEVEL_COLS].drop_duplicates()

# aggregate trader-level variables to market-day
market_day = (
    trader_day.groupby(["market_uuid", "trading_day"])
    .agg(
        # market composition and demographics
        n_traders=("participant_code", "nunique"),
        avg_age=("age", "mean"),
        share_female=("gender_female", "mean"),
        share_finance_course=("finance_course", "mean"),
        share_trading_experience=("trading_experience", "mean"),
        share_high_education=("high_education", "mean"),
        avg_fin_quiz=("fin_quiz_score", "mean"),
        sd_fin_quiz=("fin_quiz_score", "std"),
        avg_self_assessment=("self_assessment", "mean"),
        avg_overconfidence=("overconfidence", "mean"),
        sd_overconfidence=("overconfidence", "std"),
        avg_cq_attempts=("cq_attempt_count", "mean"),
        share_algo_belief=("algorithm_belief", "mean"),
        # wealth
        avg_wealth=("wealth_day", "mean"),
        sd_wealth=("wealth_day", "std"),
        gini=("gini", "first"),
        # forecasts
        avg_forecast=("forecast", "mean"),
        sd_forecast=("forecast", "std"),
        # trader-type shares (market-level, constant within market)
        share_feedback=("feedback_flag", "mean"),
        share_speculator=("speculator_flag", "mean"),
        share_fundamental=("fundamental_flag", "mean"),
        share_other=("other_flag", "mean"),
    )
    .reset_index()
)

# merge treatment identifiers
market_day = market_day.merge(market_id, on="market_uuid")

# merge price / trade aggregates from mp
market_day = market_day.merge(
    mp, how="left", on=["market_uuid", "repetition", "trading_day"]
)

# cumulative surge / crash / bubble counts within each market
for col in ("surge", "crash", "bubble_period"):
    market_day[f"cum_{col}"] = market_day.groupby("market_uuid")[col].cumsum()
market_day = market_day.sort_values(
    by=["repetition", "trading_day"], ascending=True
).reset_index(drop=True)

# ============================================================
# 8. Save panels
# ============================================================

trader_day.to_csv(f"{OUT_DIR}/trader_day_panel.csv", index=False)
market_day.to_csv(f"{OUT_DIR}/market_day_panel.csv", index=False)

print(
    f"Saved trader_day ({trader_day.shape[0]:,} rows) "
    f"and market_day ({market_day.shape[0]:,} rows) to {OUT_DIR}/."
)

# save payoffs
payoffs = (
    post_exp[["player.email", "player.ucid", "participant.payoff"]]
    .dropna()
    .reset_index(drop=True)
)
exchange_rate = 1 / 500
payoffs["payoff_cad"] = (payoffs["participant.payoff"] * exchange_rate).apply(
    lambda x: round(x, 2)
)
payoffs.to_csv(f"{OUT_DIR}/participant_payments.csv", index=False)
