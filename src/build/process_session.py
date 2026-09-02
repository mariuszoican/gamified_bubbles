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

Output: trader_day_panel.csv, market_day_panel.csv, participant_payments.csv
under data/interim/{session_id}/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths import get_session, interim_dir_for, load_parameters, raw_dir_for

# Horizon for the realized spread / price impact: midquote this many seconds
# after each trade (capped at the end of the trading day). Trading days are
# 60 seconds (trading_day_duration=1 in the oTree config; 99.9% of trades
# occur within 60 s of the day's first order), and limit orders arrive every
# ~2-3 s, so 10 s is long enough for quote adjustment without running into
# the day boundary for most trades.
REALIZED_SPREAD_HORIZON_S = 10.0


def _sign_trades(trades: pd.DataFrame) -> pd.Series:
    """Sign trades Lee–Ready style: quote rule vs the prevailing midpoint,
    tick test for at-mid / unquoted trades. +1 buyer-initiated, -1 seller-
    initiated, NaN when neither rule applies."""
    sign = np.sign(trades["price"] - trades["mid_pre"])
    tick = np.sign(
        trades.groupby(["market_uuid", "trading_day"])["price"].diff()
    ).replace(0, np.nan)
    tick = tick.groupby(
        [trades["market_uuid"], trades["trading_day"]]
    ).ffill()
    sign = sign.replace(0, np.nan).fillna(tick)
    return sign


def build_tick_metrics(
    mbo: pd.DataFrame, mbp: pd.DataFrame, keep_markets: set
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Market-day liquidity / order-flow metrics from the MBO and MBP1 feeds.

    Returns (market_day_metrics, trader_day_orders):
      market_day_metrics – per market × day: time-weighted quoted spread and
        depth, share of two-sided-book time, effective and realized spreads
        (absolute and relative), midquote realized volatility, share volume,
        signed order-flow imbalance, and limit-order submission/cancel counts.
      trader_day_orders  – per trader × market × day: limit orders submitted
        and cancelled.
    """
    mbo = mbo.copy()
    mbp = mbp.copy()
    for df in (mbo, mbp):
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        df.rename(columns={"trading_session_uuid": "market_uuid"}, inplace=True)
    mbo = mbo[mbo["market_uuid"].isin(keep_markets)]
    mbp = mbp[mbp["market_uuid"].isin(keep_markets)].copy()
    mbp["ts"] = pd.to_datetime(mbp["event_ts"], format="ISO8601")

    # IMPORTANT: MBP1's event_seq is a SEPARATE counter from MBO's event_seq;
    # the join key into the MBO stream is source_mbo_event_seq (matching on
    # event_seq introduces look-ahead: the MBP1 counter lags the MBO counter).
    # MBP1 trading_day labels can also be stale (inherited from resting orders
    # around day boundaries), so remap the day from the sourcing MBO event.
    mbp = mbp.merge(
        mbo[["market_uuid", "event_seq", "trading_day"]].rename(
            columns={
                "event_seq": "source_mbo_event_seq",
                "trading_day": "mbo_day",
            }
        ),
        on=["market_uuid", "source_mbo_event_seq"],
        how="left",
    )
    mbp["trading_day"] = (
        mbp["mbo_day"].fillna(mbp["trading_day"]).astype(int)
    )
    mbp = mbp.drop(columns="mbo_day").sort_values(
        ["market_uuid", "source_mbo_event_seq"]
    )

    orders = mbo[mbo["record_kind"] == "order"]
    trades = mbo[mbo["record_kind"] == "trade"].copy()
    trades["ts"] = pd.to_datetime(trades["event_ts"], format="ISO8601")

    # True aggressor side per order: an order whose FIRST recorded event is a
    # fill executed on arrival (marketable); one that first appears as "add"
    # rested passively. (The exported aggressor_side field is empty.)
    first_event = orders.sort_values("event_seq").groupby("order_id").first()
    marketable_ids = set(first_event[first_event["event_type"] == "fill"].index)

    # --- quote-based metrics, time-weighted over two-sided-book spells ---
    quote_rows = []
    for (mu, day), q in mbp.groupby(["market_uuid", "trading_day"]):
        q = q.sort_values("source_mbo_event_seq")
        two = q["best_bid_px"].notna() & q["best_ask_px"].notna()
        dur = (q["ts"].shift(-1) - q["ts"]).dt.total_seconds().clip(lower=0)
        dur.iloc[-1] = 0.0
        day_len, w = dur.sum(), dur[two]
        qs = rqs = depth = np.nan
        if day_len > 0 and w.sum() > 0:
            qs = np.average(q["spread"][two], weights=w)
            rqs = np.average((q["spread"] / q["midpoint"])[two], weights=w)
            depth = np.average(
                (q["best_bid_sz"] + q["best_ask_sz"])[two], weights=w
            )
        mid = q["midpoint"][two]
        r = np.log(mid).diff().dropna()
        # undercutting: limit-order submissions that tighten the quoted spread
        is_add = q["source_order_event_type"] == "add"
        prev_spread = q["spread"].shift(1)
        improving = (
            is_add & (q["spread"] < prev_spread)
            & q["spread"].notna() & prev_spread.notna()
        ).sum()
        n_book_adds = int(is_add.sum())
        quote_rows.append(
            dict(
                market_uuid=mu,
                trading_day=day,
                quoted_spread=qs,
                rel_quoted_spread=rqs,
                depth_best=depth,
                pct_two_sided=w.sum() / day_len if day_len > 0 else np.nan,
                rv_mid=np.sqrt((r**2).sum()) if len(r) else np.nan,
                n_improving_adds=int(improving),
                share_improving_adds=(
                    improving / n_book_adds if n_book_adds > 0 else np.nan
                ),
            )
        )
    quote_metrics = pd.DataFrame(quote_rows)

    # --- trade-based metrics: effective / realized spreads, signed flow ---
    mids = mbp[mbp["best_bid_px"].notna() & mbp["best_ask_px"].notna()]
    mids = mids[
        ["market_uuid", "trading_day", "source_mbo_event_seq", "ts", "midpoint"]
    ]

    def _prevailing(t, m, left_key, right_key, exact):
        return pd.merge_asof(
            t.sort_values(left_key),
            m[[right_key, "midpoint"]]
            .rename(columns={right_key: "join_key"})
            .sort_values("join_key"),
            left_on=left_key,
            right_on="join_key",
            direction="backward",
            allow_exact_matches=exact,
        )["midpoint"].to_numpy()

    trade_rows = []
    for (mu, day), t in trades.groupby(["market_uuid", "trading_day"]):
        t = t.sort_values("event_seq").copy()
        m = mids[(mids["market_uuid"] == mu) & (mids["trading_day"] == day)]
        if len(m):
            # midpoint from the last book snapshot sourced STRICTLY before
            # the trade in the MBO event order, and the prevailing midpoint
            # REALIZED_SPREAD_HORIZON_S later (capped at day end)
            t["mid_pre"] = _prevailing(
                t, m, "event_seq", "source_mbo_event_seq", False
            )
            t["ts_fwd"] = t["ts"] + pd.Timedelta(seconds=REALIZED_SPREAD_HORIZON_S)
            t = t.sort_values("ts_fwd")
            t["mid_post"] = _prevailing(t, m, "ts_fwd", "ts", True)
            t = t.sort_values("event_seq")
        else:
            t["mid_pre"] = np.nan
            t["mid_post"] = np.nan
        trade_rows.append(t)
    trades = pd.concat(trade_rows) if trade_rows else trades.assign(
        mid_pre=np.nan, mid_post=np.nan
    )

    # Sign trades by the TRUE aggressor. On trade records the exported
    # "side" column IS the aggressor side ("bid" = buyer-initiated): the
    # engine derives it from its internal aggressor marker (the separate
    # aggressor_side column is dropped by the exporter's schema). Verified
    # to agree 1:1 with reconstructing the aggressor as the order whose
    # first event is an on-arrival fill. Lee-Ready signing is NOT used: it
    # is biased here because trades often print against stale midpoints in
    # these thin, intermittently one-sided books.
    trades["sign"] = np.select(
        [trades["side"] == "bid", trades["side"] == "ask"],
        [1.0, -1.0],
        default=np.nan,
    )
    trades["sign"] = trades["sign"].fillna(_sign_trades(trades))
    trades["eff_spread"] = 2 * (trades["price"] - trades["mid_pre"]).abs()
    trades["rel_eff_spread"] = trades["eff_spread"] / trades["mid_pre"]
    trades["realized_spread"] = (
        2 * trades["sign"] * (trades["price"] - trades["mid_post"])
    )
    trades["rel_realized_spread"] = trades["realized_spread"] / trades["mid_pre"]
    # price impact: permanent midquote move in the direction of the trade
    # (effective spread = realized spread + price impact)
    trades["price_impact"] = (
        2 * trades["sign"] * (trades["mid_post"] - trades["mid_pre"])
    )
    trades["rel_price_impact"] = trades["price_impact"] / trades["mid_pre"]
    trades["signed_size"] = trades["sign"] * trades["size"]

    trade_metrics = (
        trades.groupby(["market_uuid", "trading_day"])
        .agg(
            volume_shares=("size", "sum"),
            eff_spread=("eff_spread", "mean"),
            rel_eff_spread=("rel_eff_spread", "mean"),
            realized_spread=("realized_spread", "mean"),
            rel_realized_spread=("rel_realized_spread", "mean"),
            price_impact=("price_impact", "mean"),
            rel_price_impact=("rel_price_impact", "mean"),
            signed_volume=("signed_size", "sum"),
            signed_volume_gross=("signed_size", lambda s: s.abs().sum()),
        )
        .reset_index()
    )
    trade_metrics["order_flow_imbalance"] = (
        trade_metrics["signed_volume"] / trade_metrics["signed_volume_gross"]
    )
    trade_metrics = trade_metrics.drop(columns="signed_volume_gross")

    # --- limit-order activity ---
    # An order whose first recorded event is "add" rested in the book
    # (passive limit order); one whose first event is "fill" executed on
    # arrival (marketable order).
    adds = orders[orders["event_type"] == "add"]
    cancels = orders[orders["event_type"] == "cancel"]
    marketable = first_event[first_event["event_type"] == "fill"]
    order_metrics = (
        adds.groupby(["market_uuid", "trading_day"])
        .size()
        .rename("n_limit_orders")
        .reset_index()
        .merge(
            cancels.groupby(["market_uuid", "trading_day"])
            .size()
            .rename("n_cancels")
            .reset_index(),
            on=["market_uuid", "trading_day"],
            how="outer",
        )
        .merge(
            marketable.groupby(["market_uuid", "trading_day"])
            .size()
            .rename("n_marketable_orders")
            .reset_index(),
            on=["market_uuid", "trading_day"],
            how="outer",
        )
        .fillna(0)
    )
    n_submitted = (
        order_metrics["n_limit_orders"] + order_metrics["n_marketable_orders"]
    )
    order_metrics["share_limit_orders"] = np.where(
        n_submitted > 0, order_metrics["n_limit_orders"] / n_submitted, np.nan
    )

    # --- post-trade replenishment: how fast liquidity comes back after a
    # trade consumes the quote (median per market-day)
    replen_rows = []
    for (mu, day), t in trades.groupby(["market_uuid", "trading_day"]):
        a = adds[(adds["market_uuid"] == mu) & (adds["trading_day"] == day)]
        a_ts = pd.to_datetime(a["event_ts"], format="ISO8601")
        a = a.assign(ts=a_ts).sort_values("event_seq")
        q = mbp[(mbp["market_uuid"] == mu) & (mbp["trading_day"] == day)]
        q = q[q["spread"].notna()].sort_values("source_mbo_event_seq")
        gaps, recov = [], []
        for _, r in t.iterrows():
            nxt = a[a["event_seq"] > r["event_seq"]]
            if len(nxt):
                gaps.append((nxt["ts"].iloc[0] - r["ts"]).total_seconds())
            pre = q[q["source_mbo_event_seq"] < r["event_seq"]]
            post = q[q["source_mbo_event_seq"] > r["event_seq"]]
            if len(pre) and len(post):
                ok = post[post["spread"] <= pre["spread"].iloc[-1]]
                if len(ok):
                    recov.append((ok["ts"].iloc[0] - r["ts"]).total_seconds())
        replen_rows.append(
            dict(
                market_uuid=mu,
                trading_day=day,
                time_to_next_order_s=np.median(gaps) if gaps else np.nan,
                spread_recovery_s=np.median(recov) if recov else np.nan,
            )
        )
    replen_metrics = pd.DataFrame(replen_rows)

    market_day_metrics = (
        quote_metrics.merge(
            trade_metrics, on=["market_uuid", "trading_day"], how="outer"
        )
        .merge(order_metrics, on=["market_uuid", "trading_day"], how="outer")
        .merge(replen_metrics, on=["market_uuid", "trading_day"], how="outer")
    )

    trader_day_orders = (
        adds.groupby(["market_uuid", "trading_day", "trader_uuid"])
        .size()
        .rename("n_limit_orders")
        .reset_index()
        .merge(
            cancels.groupby(["market_uuid", "trading_day", "trader_uuid"])
            .size()
            .rename("n_cancels")
            .reset_index(),
            on=["market_uuid", "trading_day", "trader_uuid"],
            how="outer",
        )
        .fillna(0)
    )
    return market_day_metrics, trader_day_orders


def process_session(
    session_id: str | None = None,
    *,
    DATE: str | None = None,
    FOLDER_NAME: str | None = None,
    SESSIONS: list[str] | None = None,
):
    """Build interim panels for one lab session.

    Prefer ``session_id`` (looked up in ``config/sessions.yaml``).
    The DATE / FOLDER_NAME / SESSIONS kwargs are kept for backward
    compatibility with older call sites.
    """
    if session_id is not None:
        session = get_session(session_id)
        DATE = session["export_date"]
        FOLDER_NAME = session["id"]
        SESSIONS = list(session["oTree_codes"])
        raw_path = raw_dir_for(session)
    else:
        if not (DATE and FOLDER_NAME and SESSIONS):
            raise ValueError(
                "Provide session_id, or all of DATE, FOLDER_NAME, and SESSIONS"
            )
        raw_path = Path(FOLDER_NAME)
        if not raw_path.is_absolute():
            from paths import RAW_DIR

            raw_path = RAW_DIR / FOLDER_NAME

    OUT_DIR = interim_dir_for(FOLDER_NAME)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    params = load_parameters()
    ROUNDS_PER_REPETITION = params["rounds_per_repetition"]
    TRAINING_ROUNDS = params["training_rounds"]
    DIVIDEND_PER_PERIOD = params["dividend_per_period"]
    BUBBLE_SURGE_SIGMA = params["bubble_surge_sigma"]
    TRADER_TYPE_THRESHOLD = params["trader_type_threshold"]
    TRADER_GROUP_SIZE = params["trader_group_size"]
    EXCHANGE_RATE = params["exchange_rate"]

    # ============================================================
    # 1. Load raw oTree exports
    # ============================================================

    def load_raw(date: str, sessions: list[str]) -> tuple:
        """Read CSVs and keep only rows from target sessions."""
        intro = pd.read_csv(raw_path / f"intro_{date}.csv")
        intro = intro[intro["session.code"].isin(sessions)]

        participants_full_groups = intro[
            intro["group.realized_group_size"] == TRADER_GROUP_SIZE
        ]["participant.code"].tolist()

        post_exp = pd.read_csv(raw_path / f"post_exp_{date}.csv")
        post_exp = post_exp[post_exp["session.code"].isin(sessions)]

        app = pd.read_csv(raw_path / f"trader_bridge_app_{date}.csv")
        app = app[app["session.code"].isin(sessions)]
        app = app[
            (app["participant._current_page_name"].isin(["FinalForProlific", "Payoff"]))
        ]
        app = app[app["participant.code"].isin(participants_full_groups)]

        mbo = pd.read_csv(
            raw_path / f"trader_bridge_app_custom_export_mbo_{date}.csv"
        )
        mbp = pd.read_csv(
            raw_path / f"trader_bridge_app_custom_export_mbp1_{date}.csv"
        )

        return intro, post_exp, app, mbo, mbp

    intro, post_exp, app, mbo, mbp = load_raw(DATE, SESSIONS)

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
        "player.assigned_initial_shares": "initial_shares",
        "player.dividend_per_share": "dividend_per_share",
        "subsession.round_number": "trading_day",
        "group.market_design": "gamified",
        "group.treatment": "treatment",
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
        "treatment",
        "trading_day",
        "payoff",
        "initial_cash",
        "initial_shares",
        "dividend_per_share",
        "forecast",
        "forecast_confidence",
        "current_cash",
        "num_shares",
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

    trader_day["hedonic"] = np.where(trader_day["treatment"].isin(["ghp", "gh"]), 1, 0)
    trader_day["price_notifications"] = np.where(
        trader_day["treatment"].isin(["ghp", "gp"]), 1, 0
    )

    trader_day["repetition"] = np.where(
        trader_day["trading_day"] <= ROUNDS_PER_REPETITION + TRAINING_ROUNDS, 1, 2
    )
    trader_day["trading_day"] = np.where(
        trader_day["trading_day"] > ROUNDS_PER_REPETITION + TRAINING_ROUNDS,
        trader_day["trading_day"] - ROUNDS_PER_REPETITION - TRAINING_ROUNDS,
        trader_day["trading_day"] - TRAINING_ROUNDS,
    )

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
            "market_number": "session_market_index",  # NOT repetition
        }
    )

    # inject the correct temporal repetition (market_uuid → repetition is 1-to-1 in trader_day)
    market_to_rep = (
        trader_day[["market_uuid", "repetition"]]
        .drop_duplicates()
        .set_index("market_uuid")["repetition"]
    )
    trades["repetition"] = trades["market_uuid"].map(market_to_rep)

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
        "size",
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

    mp = mp.merge(
        trader_day[["market_uuid", "repetition", "trading_day"]].drop_duplicates(),
        on=["market_uuid", "repetition", "trading_day"],
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

    def flag_extremes(
        series: pd.Series, group_key: str, sigma: float = BUBBLE_SURGE_SIGMA
    ):
        """Return +1 / −1 flags for observations beyond ±σ from group mean."""
        mu = series.groupby(group_key).transform("mean")
        sd = series.groupby(group_key).transform("std")
        high = (series > mu + sigma * sd).astype(int)
        low = (series < mu - sigma * sd).astype(int)
        return high, low

    mp["surge"], mp["crash"] = flag_extremes(mp["return"], mp["repetition"])

    mp["normalized_mispricing"] = (
        mp["avg_trade_price"] - mp["fundamental_value"]
    ) / mp["fundamental_value"]
    mp["bubble_period"], _ = flag_extremes(
        mp["normalized_mispricing"], mp["repetition"]
    )
    mp["bubble_start"] = (
        (mp["bubble_period"] == 1)
        & (
            mp.groupby(["market_uuid", "repetition"])["bubble_period"]
            .shift(1)
            .fillna(0)
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

    # limit-order submissions / cancellations per trader-day (MBO order feed)
    tick_market, tick_trader = build_tick_metrics(
        mbo, mbp, set(trader_day["market_uuid"].unique())
    )
    trader_day = trader_day.merge(
        tick_trader, how="left", on=["market_uuid", "trading_day", "trader_uuid"]
    )
    for col in ("n_limit_orders", "n_cancels"):
        trader_day[col] = trader_day[col].fillna(0)

    # intraday directionality |B−S|/(B+S): 1 = one-way flow, 0 = round-tripping
    gross = trader_day["n_buys"] + trader_day["n_sells"]
    trader_day["directionality"] = np.where(
        gross > 0,
        (trader_day["n_buys"] - trader_day["n_sells"]).abs() / gross,
        np.nan,
    )

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

    # --- 5b. Trade-stream reconstruction ------------------------------
    # The oTree player.num_shares snapshot is unreliable: it counts shares
    # escrowed in open sell orders as held, and it is captured when each
    # player's page submits (mid-day, asynchronous across traders), so
    # market-day share sums exceed the 90 outstanding. Rebuild holdings and
    # cash from the MBO trade stream instead, applying the day's REALIZED
    # dividend (draws from {0,4,20,8}; dividend_per_share) to end-of-day
    # reconstructed holdings.
    trades["trade_value"] = trades["price"] * trades["size"]
    flows = (
        trades.groupby(["market_uuid", "trading_day", "buyer_uuid"])
        .agg(q_buy=("size", "sum"), v_buy=("trade_value", "sum"))
        .reset_index()
        .rename(columns={"buyer_uuid": "trader_uuid"})
        .merge(
            trades.groupby(["market_uuid", "trading_day", "seller_uuid"])
            .agg(q_sell=("size", "sum"), v_sell=("trade_value", "sum"))
            .reset_index()
            .rename(columns={"seller_uuid": "trader_uuid"}),
            on=["market_uuid", "trading_day", "trader_uuid"],
            how="outer",
        )
        .fillna(0)
    )
    trader_day = trader_day.merge(
        flows, how="left", on=["market_uuid", "trading_day", "trader_uuid"]
    )
    for col in ("q_buy", "v_buy", "q_sell", "v_sell"):
        trader_day[col] = trader_day[col].fillna(0)

    trader_day = trader_day.sort_values(
        ["market_uuid", "trader_uuid", "trading_day"]
    ).reset_index(drop=True)
    g = trader_day.groupby(["market_uuid", "trader_uuid"])
    trader_day["shares_recon"] = trader_day["initial_shares"] + g[
        "q_buy"
    ].cumsum() - g["q_sell"].cumsum()
    trader_day["div_cash_recon"] = (
        trader_day["dividend_per_share"] * trader_day["shares_recon"]
    )
    g = trader_day.groupby(["market_uuid", "trader_uuid"])
    trader_day["cash_recon"] = (
        trader_day["initial_cash"]
        + g["v_sell"].cumsum()
        - g["v_buy"].cumsum()
        + g["div_cash_recon"].cumsum()
    )
    # end-of-day mark-to-fundamental wealth: cash after the day's dividend
    # plus remaining expected dividends on reconstructed holdings
    trader_day["wealth_day_recon"] = trader_day["cash_recon"] + trader_day[
        "shares_recon"
    ] * DIVIDEND_PER_PERIOD * (ROUNDS_PER_REPETITION - trader_day["trading_day"])
    trader_day = trader_day.drop(
        columns=["q_buy", "v_buy", "q_sell", "v_sell", "div_cash_recon"]
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
    trader_day["gini_recon"] = trader_day.groupby(
        ["market_uuid", "trading_day"]
    )["wealth_day_recon"].transform(gini)

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

    # Market-maker flag, orthogonal to the directional taxonomy above:
    # trades BOTH sides of the book on >=25% of active days and averages at
    # least one round trip per day (>= 2 x rounds trades over the market).
    mm = (
        trader_day.assign(
            active=(trader_day["n_buys"] + trader_day["n_sells"]) > 0,
            two_sided=(trader_day["n_buys"] > 0) & (trader_day["n_sells"] > 0),
            gross=trader_day["n_buys"] + trader_day["n_sells"],
        )
        .groupby(["market_uuid", "participant_code"])
        .agg(active=("active", "sum"), two_sided=("two_sided", "sum"),
             gross=("gross", "sum"))
    )
    mm["market_maker_flag"] = (
        (mm["two_sided"] / mm["active"].replace(0, np.nan) >= 0.25)
        & (mm["gross"] >= 2 * ROUNDS_PER_REPETITION)
    ).fillna(False).astype(float)
    trader_day = trader_day.merge(
        mm[["market_maker_flag"]].reset_index(),
        how="left",
        on=["market_uuid", "participant_code"],
    )

    # ============================================================
    # 7. Market-day panel (one row per market × trading_day)
    # ============================================================

    # columns constant within a market (treatment assignment)
    MARKET_LEVEL_COLS = [
        "market_uuid",
        "repetition",
        "treatment",
        "gamified",
        "hedonic",
        "price_notifications",
    ]

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
            # wealth
            avg_wealth=("wealth_day", "mean"),
            sd_wealth=("wealth_day", "std"),
            gini=("gini", "first"),
            # trade-stream reconstructed wealth (preferred; see 5b)
            avg_wealth_recon=("wealth_day_recon", "mean"),
            sd_wealth_recon=("wealth_day_recon", "std"),
            gini_recon=("gini_recon", "first"),
            # forecasts
            avg_forecast=("forecast", "median"),
            sd_forecast=("forecast", "std"),
            # trader-type shares (market-level, constant within market)
            share_feedback=("feedback_flag", "mean"),
            share_speculator=("speculator_flag", "mean"),
            share_fundamental=("fundamental_flag", "mean"),
            share_other=("other_flag", "mean"),
            share_market_maker=("market_maker_flag", "mean"),
        )
        .reset_index()
    )

    # merge treatment identifiers
    market_day = market_day.merge(market_id, on="market_uuid")

    # merge price / trade aggregates from mp
    market_day = market_day.merge(
        mp, how="left", on=["market_uuid", "repetition", "trading_day"]
    )

    # merge tick-level liquidity / order-flow metrics (MBO + MBP1 feeds)
    market_day = market_day.merge(
        tick_market, how="left", on=["market_uuid", "trading_day"]
    )
    for col in (
        "volume_shares",
        "n_limit_orders",
        "n_cancels",
        "n_marketable_orders",
        "n_improving_adds",
    ):
        market_day[col] = market_day[col].fillna(0)
    market_day["cancel_to_order"] = np.where(
        market_day["n_limit_orders"] > 0,
        market_day["n_cancels"] / market_day["n_limit_orders"],
        np.nan,
    )
    market_day = market_day.merge(
        trader_day.groupby(["market_uuid", "trading_day"])["directionality"]
        .mean()
        .rename("avg_directionality")
        .reset_index(),
        how="left",
        on=["market_uuid", "trading_day"],
    )

    # cumulative surge / crash / bubble counts within each market
    for col in ("surge", "crash", "bubble_period"):
        market_day[f"cum_{col}"] = market_day.groupby("market_uuid")[col].cumsum()
    market_day = market_day.sort_values(
        by=["repetition", "trading_day"], ascending=True
    ).reset_index(drop=True)

    # ============================================================
    # 8. Session id and group labels, save panels
    # ============================================================

    # Stable group labels keyed on the participant set, e.g. "20260520_PM/ng1"
    # (same convention as src/explore), so outlier groups can be excluded
    # downstream using processed data alone.
    gkey = trader_day.groupby("market_uuid")["participant_code"].apply(
        lambda s: "|".join(sorted(s.unique()))
    )
    mkt_grp = market_day[["market_uuid", "treatment"]].drop_duplicates().copy()
    mkt_grp["group_key"] = mkt_grp["market_uuid"].map(gkey)
    labels = (
        mkt_grp[["group_key", "treatment"]]
        .drop_duplicates()
        .sort_values(["treatment", "group_key"])
    )
    labels["group_label"] = (
        f"{FOLDER_NAME}/"
        + labels["treatment"]
        + (labels.groupby("treatment").cumcount() + 1).astype(str)
    )
    uuid_to_label = mkt_grp.merge(labels, on=["group_key", "treatment"]).set_index(
        "market_uuid"
    )["group_label"]
    for panel in (trader_day, market_day):
        panel["session_id"] = str(FOLDER_NAME)
        panel["group_label"] = panel["market_uuid"].map(uuid_to_label)

    trader_day = trader_day[trader_day.trading_day >= 1]
    market_day = market_day[market_day.trading_day >= 1]

    trader_day.to_csv(OUT_DIR / "trader_day_panel.csv", index=False)
    market_day.to_csv(OUT_DIR / "market_day_panel.csv", index=False)

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
    payoffs["payoff_cad"] = (payoffs["participant.payoff"] * EXCHANGE_RATE).apply(
        lambda x: round(x, 2)
    )
    payoffs.to_csv(OUT_DIR / "participant_payments.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build trader-day and market-day panels for one session."
    )
    parser.add_argument(
        "--session",
        help="Session id from config/sessions.yaml (preferred)",
    )
    parser.add_argument("--date", help="Export date stamp YYYY-MM-DD (legacy)")
    parser.add_argument("--folder", help="Raw folder name (legacy)")
    parser.add_argument("--sessions", nargs="+", help="oTree session codes (legacy)")
    args = parser.parse_args()

    if args.session:
        process_session(args.session)
    else:
        process_session(
            DATE=args.date, FOLDER_NAME=args.folder, SESSIONS=args.sessions
        )
