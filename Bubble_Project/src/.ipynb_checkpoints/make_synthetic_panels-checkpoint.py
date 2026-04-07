from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

OUT = Path("processed_data")
OUT.mkdir(exist_ok=True)

N_TRADERS = 8
N_PERIODS = 15

# 4 treatment cells x 2 repetitions = 8 synthetic markets
cells = [
    ("gh", 1, 0, "gamified", "human_only"),
    ("nh", 0, 0, "non_gamified", "human_only"),
    ("gm", 1, 1, "gamified", "mixed"),
    ("nm", 0, 1, "non_gamified", "mixed"),
]

markets = []
for rep in [1, 2]:
    for cell, gamified, hybrid, market_design, group_composition in cells:
        markets.append(
            {
                "market_id": f"{cell}_rep{rep}",
                "treatment": cell,
                "gamified": gamified,
                "hybrid": hybrid,
                "market_design": market_design,
                "group_composition": group_composition,
                "repetition": rep,
            }
        )

markets = pd.DataFrame(markets)

# ----------------------------
# market_period
# ----------------------------
mp_rows = []

for _, m in markets.iterrows():
    prev_close = 120.0
    bubble_started = False

    for k in range(1, N_PERIODS + 1):
        fv = 8 * (16 - k)

        # treatment-driven mispricing pattern
        treatment_shift = 0
        if m["gamified"] == 1:
            treatment_shift += 12
        if m["hybrid"] == 1:
            treatment_shift -= 6
        if m["repetition"] == 2:
            treatment_shift -= 5
        if m["gamified"] == 1 and m["repetition"] == 2:
            treatment_shift += 3  # weaker learning in gamified

        # bubble hump around middle periods
        hump = 22 * np.exp(-((k - 8) ** 2) / 10)
        noise = rng.normal(0, 4)

        avg_trade_price = max(1, fv + treatment_shift + hump + noise)
        closing_price = max(1, avg_trade_price + rng.normal(0, 2))
        opening_price = max(1, prev_close + rng.normal(0, 2))
        max_price = max(avg_trade_price, closing_price, opening_price) + abs(
            rng.normal(0, 2)
        )
        min_price = max(
            1,
            min(avg_trade_price, closing_price, opening_price) - abs(rng.normal(0, 2)),
        )
        n_trades = int(rng.integers(8, 20))

        lag_price = prev_close if k > 1 else np.nan
        lead_price = np.nan
        delta_p = closing_price - lag_price if k > 1 else np.nan
        ret = (closing_price - lag_price) / lag_price if k > 1 else np.nan

        normalized_mispricing = (avg_trade_price - fv) / fv
        bubble_period = int(normalized_mispricing > 0.35)
        bubble_start = int(bubble_period == 1 and not bubble_started)
        bubble_started = bubble_period == 1

        mp_rows.append(
            {
                "market_id": m["market_id"],
                "trading_day": k,
                "n_trades": n_trades,
                "avg_trade_price": avg_trade_price,
                "closing_price": closing_price,
                "opening_price": opening_price,
                "max_price": max_price,
                "min_price": min_price,
                "fundamental_value": fv,
                "gamified": m["gamified"],
                "hybrid": m["hybrid"],
                "repetition": m["repetition"],
                "treatment": m["treatment"],
                "market_design": m["market_design"],
                "group_composition": m["group_composition"],
                "average_mispricing": avg_trade_price - fv,
                "absolute_mispricing": abs(avg_trade_price - fv),
                "abs_mispricing_ratio": abs(avg_trade_price - fv) / fv,
                "lag_price": lag_price,
                "lead_price": lead_price,
                "delta_p": delta_p,
                "return": ret,
                "normalized_mispricing": normalized_mispricing,
                "bubble_period": bubble_period,
                "bubble_start": bubble_start,
            }
        )
        prev_close = closing_price

market_period = pd.DataFrame(mp_rows).sort_values(["market_id", "trading_day"])
market_period["lead_price"] = market_period.groupby("market_id")["closing_price"].shift(
    -1
)
market_period["lead_delta_p"] = (
    market_period["lead_price"] - market_period["closing_price"]
)

# recompute surge/crash within repetition
rep_mean = market_period.groupby("repetition")["return"].transform("mean")
rep_sd = market_period.groupby("repetition")["return"].transform("std")
market_period["surge"] = (market_period["return"] > rep_mean + 2 * rep_sd).astype(int)
market_period["crash"] = (market_period["return"] < rep_mean - 2 * rep_sd).astype(int)

# bubble_period/runs within repetition
rep_m_mean = market_period.groupby("repetition")["normalized_mispricing"].transform(
    "mean"
)
rep_m_sd = market_period.groupby("repetition")["normalized_mispricing"].transform("std")
market_period["bubble_period"] = (
    market_period["normalized_mispricing"] > rep_m_mean + 2 * rep_m_sd
).astype(int)
market_period["bubble_start"] = (
    (market_period["bubble_period"] == 1)
    & (market_period.groupby("market_id")["bubble_period"].shift(1).fillna(0) == 0)
).astype(int)

# ----------------------------
# trader_final and market_summary
# ----------------------------
tf_rows = []
mts_rows = []
tt_rows = []
tp_h5_rows = []
fp_rows = []

for _, m in markets.iterrows():
    mp_m = market_period[market_period["market_id"] == m["market_id"]].copy()

    # market-level type shares vary by treatment
    if m["gamified"] == 1 and m["hybrid"] == 0:
        shares = dict(feedback=0.50, speculator=0.20, fundamental=0.15, other=0.15)
    elif m["gamified"] == 1 and m["hybrid"] == 1:
        shares = dict(feedback=0.35, speculator=0.25, fundamental=0.25, other=0.15)
    elif m["gamified"] == 0 and m["hybrid"] == 0:
        shares = dict(feedback=0.20, speculator=0.20, fundamental=0.45, other=0.15)
    else:
        shares = dict(feedback=0.15, speculator=0.20, fundamental=0.50, other=0.15)

    types = rng.choice(
        ["feedback", "speculator", "fundamental", "other"],
        size=N_TRADERS,
        p=[
            shares["feedback"],
            shares["speculator"],
            shares["fundamental"],
            shares["other"],
        ],
    )

    for i in range(N_TRADERS):
        pid = f"{m['market_id']}_p{i+1}"
        fin_quiz = np.clip(rng.normal(0.6, 0.15), 0, 1)
        overconfidence = rng.normal(0.0, 0.25)

        # wealth varies more in gamified markets
        wealth_mean = (
            1200 + 120 * m["gamified"] - 40 * m["hybrid"] + 50 * (m["repetition"] == 2)
        )
        wealth_sd = 180 + 120 * m["gamified"] - 40 * m["hybrid"]
        final_wealth = max(100, rng.normal(wealth_mean, max(40, wealth_sd)))

        tf_rows.append(
            {
                "market_id": m["market_id"],
                "participant_code": pid,
                "final_wealth": final_wealth,
                "repetition": m["repetition"],
                "gamified": m["gamified"],
                "hybrid": m["hybrid"],
                "treatment": m["treatment"],
                "market_design": m["market_design"],
                "group_composition": m["group_composition"],
                "fin_quiz_score": fin_quiz,
                "overconfidence": overconfidence,
            }
        )

        trader_type = types[i]
        tt_rows.append(
            {
                "market_id": m["market_id"],
                "participant_code": pid,
                "repetition": m["repetition"],
                "gamified": m["gamified"],
                "hybrid": m["hybrid"],
                "treatment": m["treatment"],
                "market_design": m["market_design"],
                "group_composition": m["group_composition"],
                "feedback_score": 10 if trader_type == "feedback" else 3,
                "speculator_score": 10 if trader_type == "speculator" else 3,
                "fundamental_score": 10 if trader_type == "fundamental" else 3,
                "feedback_w": 1.0 if trader_type == "feedback" else 0.0,
                "speculator_w": 1.0 if trader_type == "speculator" else 0.0,
                "fundamental_w": 1.0 if trader_type == "fundamental" else 0.0,
                "other_w": 1.0 if trader_type == "other" else 0.0,
            }
        )

        # synthetic trader-period panel for H5 debug / inspection
        holdings = 10 + rng.integers(-2, 3)
        for _, row in mp_m.iterrows():
            td = int(row["trading_day"])
            feedback_ref_change = (
                mp_m.loc[mp_m["trading_day"] == td, "delta_p"].shift(1).iloc[0]
                if td > 2
                else np.nan
            )
            lead_delta = row["lead_delta_p"]
            mispricing = row["avg_trade_price"] - row["fundamental_value"]

            if trader_type == "feedback":
                change = np.sign(row["delta_p"]) if pd.notna(row["delta_p"]) else 0
            elif trader_type == "speculator":
                change = np.sign(lead_delta) if pd.notna(lead_delta) else 0
            elif trader_type == "fundamental":
                change = -np.sign(mispricing)
            else:
                change = rng.choice([-1, 0, 1])

            holdings += change

            tp_h5_rows.append(
                {
                    "participant_code": pid,
                    "market_id": m["market_id"],
                    "trading_day": td,
                    "stock_holdings": holdings,
                    "holding_change": change,
                    "feedback_ref_change": feedback_ref_change,
                    "lead_delta_p": lead_delta,
                    "price_minus_fundamental": mispricing,
                    "feedback_hit": int(
                        pd.notna(row["delta_p"])
                        and np.sign(change) == np.sign(row["delta_p"])
                        and change != 0
                    ),
                    "speculator_hit": int(
                        pd.notna(lead_delta)
                        and np.sign(change) == np.sign(lead_delta)
                        and change != 0
                    ),
                    "fundamental_hit": int(
                        np.sign(change) == -np.sign(mispricing) and change != 0
                    ),
                    "repetition": m["repetition"],
                    "gamified": m["gamified"],
                    "hybrid": m["hybrid"],
                    "treatment": m["treatment"],
                    "market_design": m["market_design"],
                    "group_composition": m["group_composition"],
                }
            )

            # synthetic beliefs panel
            delta_p = row["delta_p"]
            if pd.isna(delta_p):
                continue

            trend_beta = (
                0.30
                + 0.35 * m["gamified"]
                - 0.20 * m["hybrid"]
                - 0.10 * (m["gamified"] * m["hybrid"])
            )
            forecast_gap = trend_beta * delta_p + rng.normal(0, 2)

            fp_rows.append(
                {
                    "market_id": m["market_id"],
                    "participant_code": pid,
                    "trading_day": td,
                    "forecast_gap": forecast_gap,
                    "delta_p": delta_p,
                    "gamified": m["gamified"],
                    "hybrid": m["hybrid"],
                    "repetition": m["repetition"],
                    "z_age": rng.normal(),
                    "z_fin_quiz_score": rng.normal(),
                    "z_overconfidence": rng.normal(),
                    "z_trading_experience": rng.normal(),
                    "z_risk_aversion": rng.normal(),
                }
            )

# trader-level outputs
trader_final = pd.DataFrame(tf_rows)
trader_types = pd.DataFrame(tt_rows)
trader_period_h5 = pd.DataFrame(tp_h5_rows)
forecast_panel = pd.DataFrame(fp_rows)

# market_type_shares
market_type_shares = trader_types.groupby("market_id", as_index=False).agg(
    repetition=("repetition", "first"),
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


# market_summary
def gini(x):
    x = np.asarray(x, dtype=float)
    diffsum = np.abs(x[:, None] - x[None, :]).sum()
    return diffsum / (2 * len(x) ** 2 * x.mean())


wealth_by_market = (
    trader_final.groupby("market_id")
    .agg(
        wealth_sd=("final_wealth", "std"),
        wealth_mean=("final_wealth", "mean"),
    )
    .reset_index()
)

gini_df = (
    trader_final.groupby("market_id")["final_wealth"]
    .apply(gini)
    .reset_index(name="wealth_gini")
)
wealth_by_market = wealth_by_market.merge(gini_df, on="market_id", how="left")

market_summary = (
    market_period.groupby("market_id", as_index=False)
    .agg(
        repetition=("repetition", "first"),
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
    .merge(wealth_by_market, on="market_id", how="left")
)

market_summary["surges_crashes_total"] = (
    market_summary["n_surges"] + market_summary["n_crashes"]
)
market_summary["repetition2"] = (market_summary["repetition"] == 2).astype(int)

# minimal placeholder files expected by pipeline
participant_map = markets[
    [
        "market_id",
        "treatment",
        "market_design",
        "group_composition",
        "gamified",
        "hybrid",
        "repetition",
    ]
].copy()
participant_map["participant_code"] = np.nan
background_panel = trader_final[
    ["participant_code", "fin_quiz_score", "overconfidence"]
].drop_duplicates()
trade_panel = market_period[
    [
        "market_id",
        "trading_day",
        "avg_trade_price",
        "closing_price",
        "fundamental_value",
        "gamified",
        "hybrid",
        "repetition",
        "treatment",
        "market_design",
        "group_composition",
    ]
].copy()
trader_period = trader_period_h5[
    [
        "participant_code",
        "market_id",
        "trading_day",
        "stock_holdings",
        "holding_change",
        "repetition",
        "gamified",
        "hybrid",
        "treatment",
        "market_design",
        "group_composition",
    ]
].copy()

# save
participant_map.to_csv(OUT / "participant_map.csv", index=False)
background_panel.to_csv(OUT / "background_panel.csv", index=False)
trade_panel.to_csv(OUT / "trade_panel.csv", index=False)
market_period.to_csv(OUT / "market_period.csv", index=False)
market_summary.to_csv(OUT / "market_summary.csv", index=False)
trader_period.to_csv(OUT / "trader_period.csv", index=False)
forecast_panel.to_csv(OUT / "forecast_panel.csv", index=False)
trader_final.to_csv(OUT / "trader_final.csv", index=False)
trader_period_h5.to_csv(OUT / "trader_period_h5.csv", index=False)
trader_types.to_csv(OUT / "trader_types.csv", index=False)
market_type_shares.to_csv(OUT / "market_type_shares.csv", index=False)

print("Synthetic panel files written to processed_data/")
for p in sorted(OUT.glob("*.csv")):
    print(" -", p.name)
