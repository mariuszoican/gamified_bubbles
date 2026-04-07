# PREPARE_PANELS.PY

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("../Bubble_Project/raw_data")
OUT_DIR = Path("../Bubble_Project/processed_data")
OUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def gini(x):
    x = pd.Series(x).dropna().astype(float).values
    if len(x) == 0:
        return np.nan
    if np.allclose(x, 0):
        return 0.0
    mean_x = x.mean()
    if pd.isna(mean_x) or mean_x == 0:
        return np.nan
    diffsum = np.abs(x[:, None] - x[None, :]).sum()
    return diffsum / (2 * len(x) ** 2 * mean_x)


def standardize_series(s):
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(skipna=True)
    if pd.isna(std):
        return s
    if std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean(skipna=True)) / std


# -----------------------------
# Load raw raw_data
# -----------------------------
def load_data():
    intro = pd.read_csv(DATA_DIR / "intro_2026-03-12.csv")
    post_exp = pd.read_csv(DATA_DIR / "post_exp_2026-03-12.csv")
    app = pd.read_csv(DATA_DIR / "trader_bridge_app_2026-03-12.csv")
    mbo = pd.read_csv(DATA_DIR / "trader_bridge_app_custom_export_mbo_2026-03-12.csv")
    return intro, post_exp, app, mbo


# -----------------------------
# Participant / market mapping
# -----------------------------
def build_participant_map(app):
    cols = [
        "participant.code",
        "player.trader_uuid",
        "player.id_in_group",
        "group.trading_session_uuid",
        "group.treatment",
        "group.market_design",
        "group.group_composition",
        "subsession.round_number",
    ]
    cols = [c for c in cols if c in app.columns]

    out = (
        app[cols]
        .drop_duplicates()
        .rename(
            columns={
                "participant.code": "participant_code",
                "player.trader_uuid": "trader_uuid",
                "player.id_in_group": "id_in_group",
                "group.trading_session_uuid": "trading_session_uuid",
                "group.treatment": "treatment",
                "group.market_design": "market_design",
                "group.group_composition": "group_composition",
                "subsession.round_number": "round_number_raw",
            }
        )
    )

    if "round_number_raw" in out.columns:
        out["round_number_raw"] = pd.to_numeric(
            out["round_number_raw"], errors="coerce"
        )
        out["repetition"] = np.where(out["round_number_raw"] <= 15, 1, 2)
    else:
        out["repetition"] = np.nan

    if "market_design" in out.columns:
        out["gamified"] = (out["market_design"] == "gamified").astype(int)
    else:
        out["gamified"] = np.nan

    if "group_composition" in out.columns:
        out["hybrid"] = (out["group_composition"] != "human_only").astype(int)
    else:
        out["hybrid"] = np.nan

    out["market_id"] = out["trading_session_uuid"]

    return out


# -----------------------------
# Background / demographics
# -----------------------------
def build_background_panel(intro, post_exp):
    bg = None

    if "participant.code" in intro.columns:
        intro2 = intro.copy().rename(columns={"participant.code": "participant_code"})
        bg = intro2[["participant_code"]].drop_duplicates()

        if "player.self_assesment" in intro2.columns:
            bg["self_assessed_literacy"] = pd.to_numeric(
                intro2["player.self_assesment"], errors="coerce"
            )

    if "participant.code" in post_exp.columns:
        post2 = post_exp.copy().rename(columns={"participant.code": "participant_code"})
        post_keep = ["participant_code"]

        if "player.num_correct_answers" in post2.columns:
            post2["fin_quiz_correct"] = pd.to_numeric(
                post2["player.num_correct_answers"], errors="coerce"
            )
            post_keep.append("fin_quiz_correct")

        if "player.num_quiz_questions" in post2.columns:
            post2["fin_quiz_total"] = pd.to_numeric(
                post2["player.num_quiz_questions"], errors="coerce"
            )
            post_keep.append("fin_quiz_total")

        if "player.age" in post2.columns:
            post2["age"] = pd.to_numeric(post2["player.age"], errors="coerce")
            post_keep.append("age")

        if "player.gender" in post2.columns:
            post2["gender"] = post2["player.gender"]
            post_keep.append("gender")

        if "player.trading_experience" in post2.columns:
            post2["trading_experience"] = pd.to_numeric(
                post2["player.trading_experience"], errors="coerce"
            )
            post_keep.append("trading_experience")

        if "player.hl_switch_point" in post2.columns:
            post2["risk_aversion"] = pd.to_numeric(
                post2["player.hl_switch_point"], errors="coerce"
            )
            post_keep.append("risk_aversion")

        post_bg = post2[post_keep].drop_duplicates("participant_code")

        if bg is None:
            bg = post_bg
        else:
            bg = bg.merge(post_bg, on="participant_code", how="outer")

    if bg is None:
        bg = pd.DataFrame(columns=["participant_code"])

    if {"fin_quiz_correct", "fin_quiz_total"}.issubset(bg.columns):
        bg["fin_quiz_score"] = bg["fin_quiz_correct"] / bg["fin_quiz_total"]

    if {"self_assessed_literacy", "fin_quiz_score"}.issubset(bg.columns):
        bg["overconfidence"] = bg["self_assessed_literacy"] / 10 - bg["fin_quiz_score"]

    return bg


# -----------------------------
# Trades
# -----------------------------
def build_trade_panel(mbo, participant_map):
    trades = mbo.copy()

    if "record_kind" in trades.columns:
        trades = trades.loc[
            trades["record_kind"].astype(str).str.lower() == "trade"
        ].copy()

    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["trading_day"] = pd.to_numeric(
        trades["trading_day"], errors="coerce"
    ).astype("Int64")
    trades = trades.dropna(subset=["price", "trading_day"])

    trades["fundamental_value"] = 8 * (16 - trades["trading_day"])

    round_candidates = [
        "subsession.round_number",
        "subsession_round_number",
        "round_number",
    ]
    round_col = next((c for c in round_candidates if c in trades.columns), None)

    use_repetition_merge = False
    if round_col is not None:
        trades[round_col] = pd.to_numeric(trades[round_col], errors="coerce")
        if trades[round_col].notna().any():
            trades["repetition"] = np.where(trades[round_col] <= 15, 1, 2)
            use_repetition_merge = True

    if use_repetition_merge:
        meta_cols = [
            "trading_session_uuid",
            "repetition",
            "market_id",
            "treatment",
            "market_design",
            "group_composition",
            "gamified",
            "hybrid",
        ]
        meta_cols = [c for c in meta_cols if c in participant_map.columns]
        meta = participant_map[meta_cols].drop_duplicates()

        trades = trades.merge(
            meta,
            on=["trading_session_uuid", "repetition"],
            how="left",
            validate="many_to_one",
        )
    else:
        meta_cols = [
            "trading_session_uuid",
            "market_id",
            "treatment",
            "market_design",
            "group_composition",
            "gamified",
            "hybrid",
            "repetition",
        ]
        meta_cols = [c for c in meta_cols if c in participant_map.columns]
        meta = participant_map[meta_cols].drop_duplicates("trading_session_uuid")

        trades = trades.merge(
            meta,
            on="trading_session_uuid",
            how="left",
            validate="many_to_one",
        )

    if "event_ts" in trades.columns:
        trades["event_ts"] = pd.to_datetime(trades["event_ts"], errors="coerce")

    sort_cols = [
        c
        for c in [
            "market_id",
            "repetition",
            "trading_day",
            "event_ts",
            "event_seq",
            "price",
        ]
        if c in trades.columns
    ]
    if sort_cols:
        trades = trades.sort_values(sort_cols).reset_index(drop=True)

    return trades


# -----------------------------
# Market-period panel
# -----------------------------
def build_market_period(trades):
    mp = trades.groupby(["market_id", "repetition", "trading_day"], as_index=False).agg(
        n_trades=("price", "size"),
        avg_trade_price=("price", "mean"),
        closing_price=("price", "last"),
        opening_price=("price", "first"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        fundamental_value=("fundamental_value", "first"),
        gamified=("gamified", "first"),
        hybrid=("hybrid", "first"),
        treatment=("treatment", "first"),
        market_design=("market_design", "first"),
        group_composition=("group_composition", "first"),
    )

    mp = mp.sort_values(["market_id", "repetition", "trading_day"]).reset_index(
        drop=True
    )

    mp["average_mispricing"] = mp["avg_trade_price"] - mp["fundamental_value"]
    mp["absolute_mispricing"] = (mp["avg_trade_price"] - mp["fundamental_value"]).abs()
    mp["abs_mispricing_ratio"] = mp["absolute_mispricing"] / mp["fundamental_value"]

    mp["lag_price"] = mp.groupby(["market_id", "repetition"])["closing_price"].shift(1)
    mp["lead_price"] = mp.groupby(["market_id", "repetition"])["closing_price"].shift(
        -1
    )
    mp["delta_p"] = mp["closing_price"] - mp["lag_price"]
    mp["lead_delta_p"] = mp["lead_price"] - mp["closing_price"]

    mp["return"] = (mp["closing_price"] - mp["lag_price"]) / mp["lag_price"]

    mp["normalized_mispricing"] = (
        mp["avg_trade_price"] - mp["fundamental_value"]
    ) / mp["fundamental_value"]

    rep_return_mean = mp.groupby("repetition")["return"].transform("mean")
    rep_return_sd = mp.groupby("repetition")["return"].transform("std")
    mp["surge"] = (mp["return"] > rep_return_mean + 2 * rep_return_sd).astype(int)
    mp["crash"] = (mp["return"] < rep_return_mean - 2 * rep_return_sd).astype(int)

    rep_misp_mean = mp.groupby("repetition")["normalized_mispricing"].transform("mean")
    rep_misp_sd = mp.groupby("repetition")["normalized_mispricing"].transform("std")
    mp["bubble_period"] = (
        mp["normalized_mispricing"] > rep_misp_mean + 2 * rep_misp_sd
    ).astype(int)

    mp["bubble_start"] = (
        (mp["bubble_period"] == 1)
        & (
            mp.groupby(["market_id", "repetition"])["bubble_period"].shift(1).fillna(0)
            == 0
        )
    ).astype(int)

    return mp


# -----------------------------
# Market summary
# -----------------------------
def build_market_summary(market_period):
    ms = market_period.groupby(["market_id", "repetition"], as_index=False).agg(
        gamified=("gamified", "first"),
        hybrid=("hybrid", "first"),
        treatment=("treatment", "first"),
        market_design=("market_design", "first"),
        group_composition=("group_composition", "first"),
        n_periods=("trading_day", "nunique"),
        total_trades=("n_trades", "sum"),
        mean_avg_trade_price=("avg_trade_price", "mean"),
        mean_average_mispricing=("average_mispricing", "mean"),
        mean_absolute_mispricing=("absolute_mispricing", "mean"),
        mean_abs_mispricing_ratio=("abs_mispricing_ratio", "mean"),
        n_surges=("surge", "sum"),
        n_crashes=("crash", "sum"),
        n_bubble_runs=("bubble_start", "sum"),
    )

    ms["surges_crashes_total"] = ms["n_surges"] + ms["n_crashes"]
    ms["repetition2"] = (ms["repetition"] == 2).astype(int)

    return ms


# -----------------------------
# Trader-period panel
# -----------------------------
def build_trader_period(app, participant_map, background):
    df = app.copy()

    keep = [
        c
        for c in [
            "participant.code",
            "player.trader_uuid",
            "group.trading_session_uuid",
            "subsession.round_number",
            "player.num_shares",
            "player.current_cash",
            "player.forecast_price_next_day",
            "player.forecast_confidence_next_day",
            "player.algorithm_belief",
            "player.payoff",
        ]
        if c in df.columns
    ]

    tp = df[keep].copy()

    tp = tp.rename(
        columns={
            "participant.code": "participant_code",
            "player.trader_uuid": "trader_uuid",
            "group.trading_session_uuid": "trading_session_uuid",
            "subsession.round_number": "round_number_raw",
            "player.num_shares": "num_shares",
            "player.current_cash": "current_cash",
            "player.forecast_price_next_day": "forecast_price_next_day",
            "player.forecast_confidence_next_day": "forecast_confidence_next_day",
            "player.algorithm_belief": "algorithm_belief",
            "player.payoff": "payoff",
        }
    )

    if "round_number_raw" in tp.columns:
        tp["round_number_raw"] = pd.to_numeric(tp["round_number_raw"], errors="coerce")
        tp["repetition"] = np.where(tp["round_number_raw"] <= 15, 1, 2)
        tp["trading_day"] = ((tp["round_number_raw"] - 1) % 15) + 1
    else:
        tp["repetition"] = np.nan

    merge_keys = [
        c
        for c in [
            "participant_code",
            "trader_uuid",
            "trading_session_uuid",
            "repetition",
        ]
        if c in tp.columns
    ]

    pm_cols = [
        "participant_code",
        "trader_uuid",
        "trading_session_uuid",
        "repetition",
        "market_id",
        "gamified",
        "hybrid",
        "treatment",
        "market_design",
        "group_composition",
        "id_in_group",
    ]
    pm_cols = [c for c in pm_cols if c in participant_map.columns]

    tp = tp.merge(
        participant_map[pm_cols].drop_duplicates(),
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    for c in [
        "num_shares",
        "current_cash",
        "forecast_price_next_day",
        "forecast_confidence_next_day",
        "payoff",
    ]:
        if c in tp.columns:
            tp[c] = pd.to_numeric(tp[c], errors="coerce")

    tp = tp.merge(background, on="participant_code", how="left")

    sort_cols = [
        c
        for c in ["market_id", "repetition", "participant_code", "trading_day"]
        if c in tp.columns
    ]
    tp = tp.sort_values(sort_cols).reset_index(drop=True)

    if "num_shares" in tp.columns:
        tp["lag_num_shares"] = tp.groupby(
            ["market_id", "repetition", "participant_code"]
        )["num_shares"].shift(1)
        tp["holding_change"] = tp["num_shares"] - tp["lag_num_shares"]

    return tp


# -----------------------------
# Forecast panel for H7
# -----------------------------
def build_forecast_panel(trader_period, market_period):
    fp = trader_period.copy()

    price_info = market_period[
        [
            "market_id",
            "repetition",
            "trading_day",
            "closing_price",
            "lag_price",
            "delta_p",
        ]
    ].copy()

    fp = fp.merge(
        price_info,
        on=["market_id", "repetition", "trading_day"],
        how="left",
        validate="many_to_one",
    )

    if "forecast_price_next_day" in fp.columns:
        fp["forecast_gap"] = fp["forecast_price_next_day"] - fp["closing_price"]

    for c in [
        "age",
        "fin_quiz_score",
        "overconfidence",
        "trading_experience",
        "risk_aversion",
    ]:
        if c in fp.columns:
            fp[f"z_{c}"] = standardize_series(fp[c])

    return fp


# -----------------------------
# Final wealth / inequality
# -----------------------------
def build_trader_final(trader_period, market_period):
    last_fv = (
        market_period.groupby(["market_id", "repetition"], as_index=False)[
            "fundamental_value"
        ]
        .min()
        .rename(columns={"fundamental_value": "terminal_fundamental_value"})
    )

    tf = trader_period.sort_values(
        ["market_id", "repetition", "participant_code", "trading_day"]
    ).copy()
    tf = tf.groupby(
        ["market_id", "repetition", "participant_code"], as_index=False
    ).tail(1)

    tf = tf.merge(last_fv, on=["market_id", "repetition"], how="left")

    # Preferred measure: payoff, because the paper's payoff definition includes
    # forecast bonuses across both markets. If payoff is unavailable, fall back
    # to current_cash.
    if "payoff" in tf.columns and tf["payoff"].notna().any():
        tf["final_wealth"] = tf["payoff"]
    elif "current_cash" in tf.columns:
        tf["final_wealth"] = tf["current_cash"]
    else:
        tf["final_wealth"] = np.nan

    keep = [
        c
        for c in [
            "market_id",
            "repetition",
            "participant_code",
            "final_wealth",
            "current_cash",
            "num_shares",
            "gamified",
            "hybrid",
            "treatment",
            "market_design",
            "group_composition",
            "fin_quiz_score",
            "overconfidence",
        ]
        if c in tf.columns
    ]

    return tf[keep].copy()


def add_wealth_inequality(market_summary, trader_final):
    wealth_by_market = (
        trader_final.groupby(["market_id", "repetition"])
        .agg(
            wealth_sd=("final_wealth", "std"),
            wealth_mean=("final_wealth", "mean"),
            wealth_gini=("final_wealth", gini),
        )
        .reset_index()
    )

    out = market_summary.merge(
        wealth_by_market, on=["market_id", "repetition"], how="left"
    )
    return out


# -----------------------------
# Trader type classification for H5
# -----------------------------
def build_trader_types(trader_period, market_period):
    tp = trader_period.copy()

    price_ref = market_period[
        [
            "market_id",
            "repetition",
            "trading_day",
            "avg_trade_price",
            "fundamental_value",
            "delta_p",
            "lead_delta_p",
        ]
    ].copy()

    price_ref = price_ref.sort_values(["market_id", "repetition", "trading_day"])
    price_ref["feedback_ref_change"] = price_ref.groupby(["market_id", "repetition"])[
        "delta_p"
    ].shift(1)
    price_ref["price_minus_fundamental"] = (
        price_ref["avg_trade_price"] - price_ref["fundamental_value"]
    )

    tp = tp.merge(
        price_ref[
            [
                "market_id",
                "repetition",
                "trading_day",
                "feedback_ref_change",
                "lead_delta_p",
                "price_minus_fundamental",
            ]
        ],
        on=["market_id", "repetition", "trading_day"],
        how="left",
        validate="many_to_one",
    )

    def same_sign(a, b):
        return np.where(
            a.notna() & b.notna() & (np.sign(a) == np.sign(b)) & (np.sign(a) != 0),
            1,
            0,
        )

    def opposite_sign(a, b):
        return np.where(
            a.notna() & b.notna() & (np.sign(a) == -np.sign(b)) & (np.sign(a) != 0),
            1,
            0,
        )

    if "holding_change" in tp.columns:
        tp["feedback_hit"] = same_sign(tp["holding_change"], tp["feedback_ref_change"])
        tp["speculator_hit"] = same_sign(tp["holding_change"], tp["lead_delta_p"])
        tp["fundamental_hit"] = opposite_sign(
            tp["holding_change"], tp["price_minus_fundamental"]
        )
    else:
        tp["feedback_hit"] = np.nan
        tp["speculator_hit"] = np.nan
        tp["fundamental_hit"] = np.nan

    scores = tp.groupby(
        ["market_id", "repetition", "participant_code"], as_index=False
    ).agg(
        gamified=("gamified", "first"),
        hybrid=("hybrid", "first"),
        treatment=("treatment", "first"),
        market_design=("market_design", "first"),
        group_composition=("group_composition", "first"),
        feedback_score=("feedback_hit", "sum"),
        speculator_score=("speculator_hit", "sum"),
        fundamental_score=("fundamental_hit", "sum"),
    )

    def classify_row(row):
        d = {
            "feedback": row["feedback_score"],
            "speculator": row["speculator_score"],
            "fundamental": row["fundamental_score"],
        }
        max_score = max(d.values())
        if pd.isna(max_score) or max_score < 8:
            return {
                "feedback_w": 0.0,
                "speculator_w": 0.0,
                "fundamental_w": 0.0,
                "other_w": 1.0,
            }
        winners = [k for k, v in d.items() if v == max_score]
        w = 1.0 / len(winners)
        out = {
            "feedback_w": 0.0,
            "speculator_w": 0.0,
            "fundamental_w": 0.0,
            "other_w": 0.0,
        }
        for k in winners:
            out[f"{k}_w"] = w
        return out

    weights = scores.apply(classify_row, axis=1, result_type="expand")
    trader_types = pd.concat([scores, weights], axis=1)

    market_type_shares = trader_types.groupby(
        ["market_id", "repetition"], as_index=False
    ).agg(
        gamified=("gamified", "first"),
        hybrid=("hybrid", "first"),
        treatment=("treatment", "first"),
        market_design=("market_design", "first"),
        group_composition=("group_composition", "first"),
        share_feedback=("feedback_w", "mean"),
        share_speculator=("speculator_w", "mean"),
        share_fundamental=("fundamental_w", "mean"),
        share_other=("other_w", "mean"),
    )

    return tp, trader_types, market_type_shares


# -----------------------------
# Main
# -----------------------------
def main():
    intro, post_exp, app, mbo = load_data()

    participant_map = build_participant_map(app)
    background = build_background_panel(intro, post_exp)
    trade_panel = build_trade_panel(mbo, participant_map)

    print("trade_panel shape:", trade_panel.shape)
    if "market_id" in trade_panel.columns:
        print(
            "non-missing market_id in trade_panel:",
            trade_panel["market_id"].notna().sum(),
        )
    else:
        print("market_id missing in trade_panel")

    market_period = build_market_period(trade_panel)
    print("market_period shape:", market_period.shape)

    market_summary = build_market_summary(market_period)

    trader_period = build_trader_period(app, participant_map, background)
    forecast_panel = build_forecast_panel(trader_period, market_period)
    trader_final = build_trader_final(trader_period, market_period)

    market_summary = add_wealth_inequality(market_summary, trader_final)

    trader_period_h5, trader_types, market_type_shares = build_trader_types(
        trader_period, market_period
    )

    participant_map.to_csv(OUT_DIR / "participant_map.csv", index=False)
    background.to_csv(OUT_DIR / "background_panel.csv", index=False)
    trade_panel.to_csv(OUT_DIR / "trade_panel.csv", index=False)
    market_period.to_csv(OUT_DIR / "market_period.csv", index=False)
    market_summary.to_csv(OUT_DIR / "market_summary.csv", index=False)
    trader_period.to_csv(OUT_DIR / "trader_period.csv", index=False)
    forecast_panel.to_csv(OUT_DIR / "forecast_panel.csv", index=False)
    trader_final.to_csv(OUT_DIR / "trader_final.csv", index=False)
    trader_period_h5.to_csv(OUT_DIR / "trader_period_h5.csv", index=False)
    trader_types.to_csv(OUT_DIR / "trader_types.csv", index=False)
    market_type_shares.to_csv(OUT_DIR / "market_type_shares.csv", index=False)

    print("Saved generated datasets:")
    for fn in [
        "participant_map.csv",
        "background_panel.csv",
        "trade_panel.csv",
        "market_period.csv",
        "market_summary.csv",
        "trader_period.csv",
        "forecast_panel.csv",
        "trader_final.csv",
        "trader_period_h5.csv",
        "trader_types.csv",
        "market_type_shares.csv",
    ]:
        print(f" - {OUT_DIR / fn}")


if __name__ == "__main__":
    main()
