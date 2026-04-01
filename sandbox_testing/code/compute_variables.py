from config import *

import os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


def gini(arr: pd.Series) -> float:
    vals = np.array(arr, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return np.nan
    mean_val = vals.mean()
    if mean_val == 0:
        return np.nan
    return np.sum(np.abs(vals[:, None] - vals[None, :])) / (2 * (n**2) * mean_val)


def _safe_sign(series: pd.Series) -> pd.Series:
    out = np.sign(series.astype(float))
    out[np.isnan(series)] = np.nan
    return out


def _score_match(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    valid = lhs.notna() & rhs.notna() & (lhs != 0) & (rhs != 0)
    return (valid & (lhs == rhs)).astype(int)


def main() -> None:
    # ── SECTION: LOAD ─────────────────────────────────────────────────────────
    panel = pd.read_csv(f"{OUTPUT_DIR}/panel_trader_period.csv")
    mbo = pd.read_csv(f"{DATA_DIR}/trader_bridge_app_custom_export_mbo_{DATE}.csv")
    mbp1 = pd.read_csv(f"{DATA_DIR}/trader_bridge_app_custom_export_mbp1_{DATE}.csv")

    valid_uuids = panel["session_uuid"].dropna().unique()
    mbo = mbo[mbo["trading_session_uuid"].isin(valid_uuids)].copy()
    mbp1 = mbp1[mbp1["trading_session_uuid"].isin(valid_uuids)].copy()
    print(f"[LOAD] panel: {panel.shape}")
    print(f"[LOAD] mbo (filtered): {mbo.shape}")
    print(f"[LOAD] mbp1 (filtered): {mbp1.shape}")

    # CRITICAL FACT CHECK: each trading_session_uuid has exactly one market_number.
    market_num_counts = mbo.groupby("trading_session_uuid")["market_number"].nunique(dropna=True)
    assert (market_num_counts <= 1).all()

    # ── SECTION: FUNDAMENTAL VALUE ────────────────────────────────────────────
    mbo["fund_value"] = EXP_DIV * ((N_PERIODS + 1) - mbo["trading_day"])
    assert mbo["fund_value"].between(0, INITIAL_FV).all()

    # ── SECTION: PERIOD PRICE STATS ───────────────────────────────────────────
    trades = mbo[(mbo["record_kind"] == "trade") & (mbo["event_type"] == "trade")].copy()
    trades["mispricing"] = trades["price"] - trades["fund_value"]
    trades["abs_mispricing"] = trades["mispricing"].abs()

    period_stats = (
        trades.groupby(["trading_session_uuid", "trading_day"], as_index=False)
        .agg(
            avg_price=("price", "mean"),
            avg_mispricing=("mispricing", "mean"),
            abs_mispricing=("abs_mispricing", "mean"),
            n_trades=("price", "size"),
            fund_value=("fund_value", "first"),
        )
    )

    closing = (
        trades.sort_values(["trading_session_uuid", "trading_day", "event_seq"])
        .groupby(["trading_session_uuid", "trading_day"], as_index=False)
        .tail(1)[["trading_session_uuid", "trading_day", "price"]]
        .rename(columns={"price": "closing_price"})
    )
    period_stats = period_stats.merge(closing, on=["trading_session_uuid", "trading_day"], how="left")

    denom = period_stats["fund_value"].replace({0: np.nan})
    period_stats["abs_mp_ratio"] = period_stats["abs_mispricing"] / denom
    period_stats["norm_mispricing"] = (period_stats["avg_price"] - period_stats["fund_value"]) / denom

    # ── SECTION: PERIOD RETURNS & PRICE CHANGES ───────────────────────────────
    period_stats = period_stats.sort_values(["trading_session_uuid", "trading_day"]).copy()
    period_stats["period_return"] = period_stats.groupby("trading_session_uuid")["closing_price"].pct_change()
    period_stats["price_change"] = period_stats.groupby("trading_session_uuid")["closing_price"].diff()

    # Bring market_rep + treatment/meta context from panel.
    period_meta = (
        panel[
            [
                "session_uuid",
                "trading_day",
                "market_rep",
                "market_design",
                "group_composition",
                "treatment",
                "is_gamified",
                "is_hybrid",
                "gamified_x_hybrid",
                "algo_present",
            ]
        ]
        .drop_duplicates(subset=["session_uuid", "trading_day", "market_rep"])
        .rename(columns={"session_uuid": "trading_session_uuid"})
    )
    period_stats = period_stats.merge(period_meta, on=["trading_session_uuid", "trading_day"], how="left")

    # ── SECTION: SURGE / CRASH FLAGS ──────────────────────────────────────────
    grp_sc = period_stats.groupby(["trading_session_uuid", "market_rep"])["period_return"]
    period_stats["ret_mean_market"] = grp_sc.transform("mean")
    period_stats["ret_sd_market"] = grp_sc.transform("std")
    period_stats["is_surge_crash"] = (
        (period_stats["period_return"] - period_stats["ret_mean_market"]).abs()
        > (2 * period_stats["ret_sd_market"])
    ).astype(int)

    # ── SECTION: BUBBLE PERIODS & BUBBLE RUNS ─────────────────────────────────
    grp_bp = period_stats.groupby(["trading_session_uuid", "market_rep"])["norm_mispricing"]
    period_stats["nm_mean_market"] = grp_bp.transform("mean")
    period_stats["nm_sd_market"] = grp_bp.transform("std")
    period_stats["is_bubble_period"] = (
        period_stats["norm_mispricing"] > (period_stats["nm_mean_market"] + 2 * period_stats["nm_sd_market"])
    ).astype(int)

    period_stats = period_stats.sort_values(["trading_session_uuid", "trading_day"]).copy()
    period_stats["bubble_run_id"] = (
        period_stats.groupby("trading_session_uuid")["is_bubble_period"]
        .transform(lambda s: (s != s.shift()).cumsum())
    )
    bubble_runs = (
        period_stats[period_stats["is_bubble_period"] == 1]
        .groupby(["trading_session_uuid", "market_rep"])["bubble_run_id"]
        .nunique()
        .reset_index(name="n_bubble_runs")
    )

    # ── SECTION: DELTA HOLDINGS (position changes per trader per period) ─────
    buys = trades[["trading_session_uuid", "trading_day", "bid_trader_uuid"]].copy()
    buys = buys.rename(columns={"bid_trader_uuid": "trader_uuid"})
    buys["delta"] = 1

    sells = trades[["trading_session_uuid", "trading_day", "ask_trader_uuid"]].copy()
    sells = sells.rename(columns={"ask_trader_uuid": "trader_uuid"})
    sells["delta"] = -1

    all_pos = pd.concat([buys, sells], ignore_index=True)
    delta_holdings = (
        all_pos.groupby(["trading_session_uuid", "trading_day", "trader_uuid"], as_index=False)["delta"].sum()
        .rename(columns={"delta": "delta_holdings"})
    )

    # ── SECTION: TRADER TYPE CLASSIFICATION ───────────────────────────────────
    panel_trader_period = panel.merge(
        delta_holdings.rename(columns={"trading_session_uuid": "session_uuid"}),
        on=["session_uuid", "trading_day", "trader_uuid"],
        how="left",
    )
    panel_trader_period["delta_holdings"] = panel_trader_period["delta_holdings"].fillna(0)

    period_for_trader = period_stats[
        ["trading_session_uuid", "trading_day", "avg_price", "fund_value", "price_change"]
    ].rename(columns={"trading_session_uuid": "session_uuid"})
    panel_trader_period = panel_trader_period.merge(
        period_for_trader, on=["session_uuid", "trading_day"], how="left"
    )

    panel_trader_period = panel_trader_period.sort_values(
        ["session_uuid", "trader_uuid", "trading_day"]
    ).copy()
    session_price = period_stats[["trading_session_uuid", "trading_day", "price_change"]].rename(
        columns={"trading_session_uuid": "session_uuid"}
    )
    session_price = session_price.sort_values(["session_uuid", "trading_day"]).copy()
    session_price["lag_price_change"] = session_price.groupby("session_uuid")["price_change"].shift(1)
    session_price["fwd_price_change"] = session_price.groupby("session_uuid")["price_change"].shift(-1)
    panel_trader_period = panel_trader_period.merge(
        session_price[["session_uuid", "trading_day", "lag_price_change", "fwd_price_change"]],
        on=["session_uuid", "trading_day"],
        how="left",
    )

    sign_delta = _safe_sign(panel_trader_period["delta_holdings"])
    sign_lag = _safe_sign(panel_trader_period["lag_price_change"])
    sign_fwd = _safe_sign(panel_trader_period["fwd_price_change"])
    sign_fund = _safe_sign(panel_trader_period["avg_price"] - panel_trader_period["fund_value"])

    panel_trader_period["feedback_hit"] = _score_match(sign_delta, sign_lag)
    panel_trader_period["speculator_hit"] = _score_match(sign_delta, sign_fwd)
    panel_trader_period["fundamental_hit"] = _score_match(sign_delta, -sign_fund)

    trader_scores = (
        panel_trader_period.groupby(
            ["session_uuid", "market_rep", "participant_code", "trader_uuid"], as_index=False
        )
        .agg(
            feedback_score=("feedback_hit", "sum"),
            speculator_score=("speculator_hit", "sum"),
            fundamental_score=("fundamental_hit", "sum"),
            payoff_total=("payoff_total", "first"),
        )
    )

    # Fractional type shares for ties.
    score_cols = ["feedback_score", "speculator_score", "fundamental_score"]
    score_values = trader_scores[score_cols].to_numpy(dtype=float)
    max_scores = np.nanmax(score_values, axis=1)
    tied = score_values == max_scores[:, None]
    tie_counts = tied.sum(axis=1).astype(float)
    qualifies = max_scores >= BADGE_THRESHOLD

    trader_scores["share_feedback"] = np.where(qualifies & tied[:, 0], 1 / tie_counts, 0.0)
    trader_scores["share_speculator"] = np.where(qualifies & tied[:, 1], 1 / tie_counts, 0.0)
    trader_scores["share_fundamental"] = np.where(qualifies & tied[:, 2], 1 / tie_counts, 0.0)
    trader_scores["share_other"] = np.where(qualifies, 0.0, 1.0)

    trader_scores["trader_type"] = "other"
    trader_scores.loc[(trader_scores["share_feedback"] == 1.0), "trader_type"] = "feedback"
    trader_scores.loc[(trader_scores["share_speculator"] == 1.0), "trader_type"] = "speculator"
    trader_scores.loc[(trader_scores["share_fundamental"] == 1.0), "trader_type"] = "fundamental"

    # ── SECTION: MARKET-LEVEL AGGREGATES ──────────────────────────────────────
    market_base = (
        panel.groupby(["session_uuid", "market_rep"], as_index=False)
        .agg(
            payoff_gini=("payoff_total", gini),
            sd_payoff=("payoff_total", "std"),
            treatment=("treatment", "first"),
            market_design=("market_design", "first"),
            group_composition=("group_composition", "first"),
            is_gamified=("is_gamified", "first"),
            is_hybrid=("is_hybrid", "first"),
            gamified_x_hybrid=("gamified_x_hybrid", "first"),
            algo_present=("algo_present", "first"),
        )
    )

    surge_counts = (
        period_stats.groupby(["trading_session_uuid", "market_rep"], as_index=False)["is_surge_crash"]
        .sum()
        .rename(columns={"trading_session_uuid": "session_uuid", "is_surge_crash": "n_surge_crash"})
    )
    bubble_runs = bubble_runs.rename(columns={"trading_session_uuid": "session_uuid"})
    type_shares = (
        trader_scores.groupby(["session_uuid", "market_rep"], as_index=False)
        .agg(
            share_feedback=("share_feedback", "mean"),
            share_speculator=("share_speculator", "mean"),
            share_fundamental=("share_fundamental", "mean"),
            share_other=("share_other", "mean"),
        )
    )

    market_period_means = (
        period_stats.groupby(["trading_session_uuid", "market_rep"], as_index=False)
        .agg(
            avg_abs_mispricing=("abs_mispricing", "mean"),
            avg_norm_mispricing=("norm_mispricing", "mean"),
            avg_n_trades=("n_trades", "mean"),
        )
        .rename(columns={"trading_session_uuid": "session_uuid"})
    )

    panel_market = market_base.merge(surge_counts, on=["session_uuid", "market_rep"], how="left")
    panel_market = panel_market.merge(bubble_runs, on=["session_uuid", "market_rep"], how="left")
    panel_market = panel_market.merge(type_shares, on=["session_uuid", "market_rep"], how="left")
    panel_market = panel_market.merge(market_period_means, on=["session_uuid", "market_rep"], how="left")
    panel_market["n_surge_crash"] = panel_market["n_surge_crash"].fillna(0)
    panel_market["n_bubble_runs"] = panel_market["n_bubble_runs"].fillna(0)

    # ── SECTION: MBP-1 STATS ──────────────────────────────────────────────────
    mbp1["event_ts"] = pd.to_datetime(mbp1["event_ts"], errors="coerce")
    mbp1 = mbp1.sort_values(["trading_session_uuid", "trading_day", "event_ts"]).copy()
    mbp1["next_ts"] = mbp1.groupby(["trading_session_uuid", "trading_day"])["event_ts"].shift(-1)
    mbp1["dt_seconds"] = (mbp1["next_ts"] - mbp1["event_ts"]).dt.total_seconds()

    # For final event in a period, use the median event spacing in that period; fallback to 1 second.
    med_dt = mbp1.groupby(["trading_session_uuid", "trading_day"])["dt_seconds"].transform("median")
    mbp1["dt_seconds"] = mbp1["dt_seconds"].fillna(med_dt).fillna(1.0).clip(lower=0.0)

    mbp1_stats = (
        mbp1.groupby(["trading_session_uuid", "trading_day"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "tw_avg_spread": np.average(g["spread"], weights=g["dt_seconds"])
                    if g["spread"].notna().any()
                    else np.nan,
                    "tw_avg_midpoint": np.average(g["midpoint"], weights=g["dt_seconds"])
                    if g["midpoint"].notna().any()
                    else np.nan,
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # ── SECTION: COMBINE PERIOD-MARKET PANEL ──────────────────────────────────
    panel_period_market = period_stats.merge(
        mbp1_stats, on=["trading_session_uuid", "trading_day"], how="left"
    ).rename(columns={"trading_session_uuid": "session_uuid"})

    # ── SECTION: SAVE ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_mbo = f"{OUTPUT_DIR}/mbo_processed.csv"
    out_ppm = f"{OUTPUT_DIR}/panel_period_market.csv"
    out_pm = f"{OUTPUT_DIR}/panel_market.csv"
    out_pt = f"{OUTPUT_DIR}/panel_trader.csv"

    mbo.to_csv(out_mbo, index=False)
    panel_period_market.to_csv(out_ppm, index=False)
    panel_market.to_csv(out_pm, index=False)
    trader_scores.to_csv(out_pt, index=False)

    print(f"[SAVE] {out_mbo} — {mbo.shape}")
    print(mbo.head(3))
    print(f"[SAVE] {out_ppm} — {panel_period_market.shape}")
    print(panel_period_market.head(3))
    print(f"[SAVE] {out_pm} — {panel_market.shape}")
    print(panel_market.head(3))
    print(f"[SAVE] {out_pt} — {trader_scores.shape}")
    print(trader_scores.head(3))


if __name__ == "__main__":
    main()
