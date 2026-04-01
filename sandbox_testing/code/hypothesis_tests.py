from config import *

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
try:
    from linearmodels.panel import PanelOLS
except Exception:
    PanelOLS = None

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)

controls_str = " + ".join(CONTROLS)
results = {}


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _fit_market_ols(df: pd.DataFrame, formula: str):
    model_df = df.copy()
    needed = [x.strip() for x in formula.replace("~", "+").split("+")]
    needed = [x for x in needed if x]
    model_df = model_df.dropna(subset=[c for c in needed if c in model_df.columns])
    if model_df.empty:
        return None, model_df

    try:
        res = smf.ols(formula=formula, data=model_df).fit(
            cov_type="cluster", cov_kwds={"groups": model_df["session_uuid"]}
        )
    except Exception:
        res = smf.ols(formula=formula, data=model_df).fit(cov_type="HC1")
    return res, model_df


def _result_to_table(res, hypothesis: str, outcome: str) -> pd.DataFrame:
    if res is None:
        return pd.DataFrame(
            {
                "hypothesis": [hypothesis],
                "outcome": [outcome],
                "term": [np.nan],
                "coef": [np.nan],
                "se": [np.nan],
                "pvalue": [np.nan],
                "nobs": [0],
                "r2": [np.nan],
            }
        )

    out = pd.DataFrame(
        {
            "hypothesis": hypothesis,
            "outcome": outcome,
            "term": res.params.index,
            "coef": res.params.values,
            "se": res.bse.values,
            "pvalue": res.pvalues.values,
            "nobs": float(res.nobs),
            "r2": getattr(res, "rsquared", np.nan),
        }
    )
    return out


def main() -> None:
    # ── SECTION: LOAD INPUT TABLES ────────────────────────────────────────────
    mbo_processed = pd.read_csv(f"{OUTPUT_DIR}/mbo_processed.csv")
    panel_period_market = pd.read_csv(f"{OUTPUT_DIR}/panel_period_market.csv")
    panel_market = pd.read_csv(f"{OUTPUT_DIR}/panel_market.csv")
    panel_trader = pd.read_csv(f"{OUTPUT_DIR}/panel_trader.csv")
    panel_trader_period = pd.read_csv(f"{OUTPUT_DIR}/panel_trader_period.csv")

    print(f"[LOAD] mbo_processed: {mbo_processed.shape}")
    print(f"[LOAD] panel_period_market: {panel_period_market.shape}")
    print(f"[LOAD] panel_market: {panel_market.shape}")
    print(f"[LOAD] panel_trader: {panel_trader.shape}")
    print(f"[LOAD] panel_trader_period: {panel_trader_period.shape}")

    # Harmonize naming for hypothesis specs.
    if "payoff_gini" in panel_market.columns and "gini" not in panel_market.columns:
        panel_market = panel_market.rename(columns={"payoff_gini": "gini"})

    if "avg_abs_mispricing" in panel_market.columns and "abs_mispricing" not in panel_market.columns:
        panel_market = panel_market.rename(columns={"avg_abs_mispricing": "abs_mispricing"})
    if "avg_norm_mispricing" in panel_market.columns and "avg_mispricing" not in panel_market.columns:
        panel_market["avg_mispricing"] = panel_market["avg_norm_mispricing"]

    # Merge trader-level controls to market panel as market means.
    controls_agg = (
        panel_trader_period.groupby(["session_uuid", "market_rep"], as_index=False)
        .agg(
            fin_lit_score=("fin_lit_score", "mean"),
            self_assess=("self_assess", "mean"),
            age=("age", "mean"),
            overconfidence=("overconfidence", "mean"),
            trading_experience=("trading_experience", "mean"),
            is_rep2=("is_rep2", "first"),
        )
    )
    market_df = panel_market.merge(controls_agg, on=["session_uuid", "market_rep"], how="left")
    market_df = _ensure_cols(
        market_df,
        [
            "is_gamified",
            "is_hybrid",
            "gamified_x_hybrid",
            "fin_lit_score",
            "self_assess",
            "age",
            "overconfidence",
            "trading_experience",
            "is_rep2",
            "avg_mispricing",
            "abs_mispricing",
            "abs_mp_ratio",
            "n_bubble_runs",
            "n_surge_crash",
            "gini",
            "sd_payoff",
            "share_feedback",
            "share_fundamental",
        ],
    )

    # Additional mispricing metrics from period-market panel.
    misp = (
        panel_period_market.groupby(["session_uuid", "market_rep"], as_index=False)
        .agg(
            avg_mispricing=("avg_mispricing", "mean"),
            abs_mispricing=("abs_mispricing", "mean"),
            abs_mp_ratio=("abs_mp_ratio", "mean"),
        )
    )
    market_df = market_df.drop(columns=["avg_mispricing", "abs_mispricing", "abs_mp_ratio"], errors="ignore")
    market_df = market_df.merge(misp, on=["session_uuid", "market_rep"], how="left")

    # ── H1: MISPRICING ────────────────────────────────────────────────────────
    h1_tables = []
    h1_formula_base = f"{{outcome}} ~ is_gamified + is_hybrid + gamified_x_hybrid + {controls_str}"
    h1_map = {
        "h1_avg_mp": "avg_mispricing",
        "h1_abs_mp": "abs_mispricing",
        "h1_ratio": "abs_mp_ratio",
    }
    for key, outcome in h1_map.items():
        formula = h1_formula_base.format(outcome=outcome)
        res, _ = _fit_market_ols(market_df, formula)
        results[key] = res
        h1_tables.append(_result_to_table(res, "H1", outcome))
    h1_out = pd.concat(h1_tables, ignore_index=True)
    h1_out.to_csv(f"{OUTPUT_DIR}/h1_results.csv", index=False)

    # ── H2: VOLATILITY ────────────────────────────────────────────────────────
    h2_tables = []
    h2_map = {"h2_bubble_runs": "n_bubble_runs", "h2_surge_crash": "n_surge_crash"}
    for key, outcome in h2_map.items():
        formula = h1_formula_base.format(outcome=outcome)
        res, _ = _fit_market_ols(market_df, formula)
        results[key] = res
        h2_tables.append(_result_to_table(res, "H2", outcome))
    h2_out = pd.concat(h2_tables, ignore_index=True)
    h2_out.to_csv(f"{OUTPUT_DIR}/h2_results.csv", index=False)

    # ── H3: INEQUALITY ────────────────────────────────────────────────────────
    h3_tables = []
    h3_map = {"h3_gini": "gini", "h3_sd_payoff": "sd_payoff"}
    for key, outcome in h3_map.items():
        formula = h1_formula_base.format(outcome=outcome)
        res, _ = _fit_market_ols(market_df, formula)
        results[key] = res
        h3_tables.append(_result_to_table(res, "H3", outcome))
    h3_out = pd.concat(h3_tables, ignore_index=True)
    h3_out.to_csv(f"{OUTPUT_DIR}/h3_results.csv", index=False)

    # ── H4: EXPERIENCE INTERACTION ────────────────────────────────────────────
    h4_formula = f"abs_mispricing ~ is_gamified * is_rep2 + is_hybrid + {controls_str}"
    h4_res, _ = _fit_market_ols(market_df, h4_formula)
    results["h4_abs_mp_exp"] = h4_res
    h4_out = _result_to_table(h4_res, "H4", "abs_mispricing")
    h4_out.to_csv(f"{OUTPUT_DIR}/h4_results.csv", index=False)

    # ── H5: TRADER TYPES ──────────────────────────────────────────────────────
    h5_tables = []
    h5_map = {"h5_feedback": "share_feedback", "h5_fundamental": "share_fundamental"}
    for key, outcome in h5_map.items():
        formula = h1_formula_base.format(outcome=outcome)
        res, _ = _fit_market_ols(market_df, formula)
        results[key] = res
        h5_tables.append(_result_to_table(res, "H5", outcome))
    h5_out = pd.concat(h5_tables, ignore_index=True)
    h5_out.to_csv(f"{OUTPUT_DIR}/h5_results.csv", index=False)

    # ── H6: ALGO BELIEFS ──────────────────────────────────────────────────────
    h6_tables = []
    h6_map = {"h6_abs_mp": "abs_mispricing", "h6_bubble_runs": "n_bubble_runs", "h6_gini": "gini"}
    for key, outcome in h6_map.items():
        formula = h1_formula_base.format(outcome=outcome)
        res, _ = _fit_market_ols(market_df, formula)
        results[key] = res
        h6_tables.append(_result_to_table(res, "H6", outcome))
    h6_out = pd.concat(h6_tables, ignore_index=True)
    h6_out.to_csv(f"{OUTPUT_DIR}/h6_results.csv", index=False)

    # ── H7: FORECAST EXTRAPOLATION (PanelOLS) ─────────────────────────────────
    h7_df = panel_trader_period.merge(
        panel_period_market[["session_uuid", "trading_day", "market_rep", "closing_price", "price_change"]],
        on=["session_uuid", "trading_day", "market_rep"],
        how="left",
    )
    h7_df["forecast_error"] = h7_df["forecast_price"] - h7_df["closing_price"]
    h7_df["pc_x_gamified"] = h7_df["price_change"] * h7_df["is_gamified"]
    h7_df["pc_x_hybrid"] = h7_df["price_change"] * h7_df["is_hybrid"]
    h7_df["pc_x_gam_x_hyb"] = h7_df["price_change"] * h7_df["gamified_x_hybrid"]
    h7_df["trading_period_idx"] = ((h7_df["market_rep"] - 1) * N_PERIODS) + h7_df["trading_day"]

    h7_rhs = [
        "price_change",
        "pc_x_gamified",
        "pc_x_hybrid",
        "pc_x_gam_x_hyb",
        "fin_lit_score",
        "self_assess",
        "age",
        "overconfidence",
        "trading_experience",
        "is_rep2",
    ]
    h7_df = h7_df.dropna(subset=["forecast_error"] + h7_rhs).copy()

    h7_res = None
    h7_mode = "PanelOLS"
    if not h7_df.empty:
        if PanelOLS is not None:
            h7_panel = h7_df.set_index(["participant_code", "trading_period_idx"])
            exog = h7_panel[h7_rhs]
            y = h7_panel["forecast_error"]
            try:
                h7_mod = PanelOLS(y, exog, entity_effects=True, time_effects=True)
                h7_res = h7_mod.fit(cov_type="clustered", cluster_entity=True)
            except Exception:
                h7_res = h7_mod.fit(cov_type="robust")
        else:
            # Fallback with equivalent FE structure when linearmodels is unavailable.
            h7_mode = "OLS_FE_fallback"
            fe_formula = (
                "forecast_error ~ price_change + pc_x_gamified + pc_x_hybrid + pc_x_gam_x_hyb + "
                "fin_lit_score + self_assess + age + overconfidence + trading_experience + is_rep2 + "
                "C(participant_code) + C(trading_period_idx)"
            )
            try:
                h7_res = smf.ols(fe_formula, data=h7_df).fit(
                    cov_type="cluster", cov_kwds={"groups": h7_df["session_uuid"]}
                )
            except Exception:
                h7_res = smf.ols(fe_formula, data=h7_df).fit(cov_type="HC1")
    results["h7_forecast_fe"] = h7_res

    if h7_res is None:
        h7_out = pd.DataFrame(
            {
                "hypothesis": ["H7"],
                "outcome": ["forecast_error"],
                "term": [np.nan],
                "coef": [np.nan],
                "se": [np.nan],
                "pvalue": [np.nan],
                "nobs": [0],
                "r2": [np.nan],
            }
        )
    elif PanelOLS is not None and h7_mode == "PanelOLS":
        h7_out = pd.DataFrame(
            {
                "hypothesis": "H7",
                "outcome": "forecast_error",
                "term": h7_res.params.index,
                "coef": h7_res.params.values,
                "se": h7_res.std_errors.values,
                "pvalue": h7_res.pvalues.values,
                "nobs": float(h7_res.nobs),
                "r2": float(h7_res.rsquared),
            }
        )
    else:
        h7_out = _result_to_table(h7_res, "H7", "forecast_error")
    h7_out.to_csv(f"{OUTPUT_DIR}/h7_results.csv", index=False)

    # ── FINAL SECTION: RESULTS SUMMARY ────────────────────────────────────────
    def _extract_key_row(hypothesis: str, outcome: str, res, key_term: str):
        if res is None:
            return {
                "Hypothesis": hypothesis,
                "Outcome": outcome,
                "Coef_is_gamified": np.nan,
                "SE": np.nan,
                "pvalue": np.nan,
                "Significant": 0,
            }
        params = res.params
        bse = res.bse if hasattr(res, "bse") else res.std_errors
        pvals = res.pvalues
        term = key_term if key_term in params.index else ("is_gamified" if "is_gamified" in params.index else None)
        if term is None:
            return {
                "Hypothesis": hypothesis,
                "Outcome": outcome,
                "Coef_is_gamified": np.nan,
                "SE": np.nan,
                "pvalue": np.nan,
                "Significant": 0,
            }
        pval = float(pvals[term])
        return {
            "Hypothesis": hypothesis,
            "Outcome": outcome,
            "Coef_is_gamified": float(params[term]),
            "SE": float(bse[term]),
            "pvalue": pval,
            "Significant": int(pval < 0.05),
        }

    summary_rows = [
        _extract_key_row("H1", "avg_mispricing", results.get("h1_avg_mp"), "is_gamified"),
        _extract_key_row("H1", "abs_mispricing", results.get("h1_abs_mp"), "is_gamified"),
        _extract_key_row("H1", "abs_mp_ratio", results.get("h1_ratio"), "is_gamified"),
        _extract_key_row("H2", "n_bubble_runs", results.get("h2_bubble_runs"), "is_gamified"),
        _extract_key_row("H2", "n_surge_crash", results.get("h2_surge_crash"), "is_gamified"),
        _extract_key_row("H3", "gini", results.get("h3_gini"), "is_gamified"),
        _extract_key_row("H3", "sd_payoff", results.get("h3_sd_payoff"), "is_gamified"),
        _extract_key_row("H4", "abs_mispricing", results.get("h4_abs_mp_exp"), "is_gamified:is_rep2"),
        _extract_key_row("H5", "share_feedback", results.get("h5_feedback"), "is_gamified"),
        _extract_key_row("H5", "share_fundamental", results.get("h5_fundamental"), "is_gamified"),
        _extract_key_row("H6", "abs_mispricing", results.get("h6_abs_mp"), "is_gamified"),
        _extract_key_row("H6", "n_bubble_runs", results.get("h6_bubble_runs"), "is_gamified"),
        _extract_key_row("H6", "gini", results.get("h6_gini"), "is_gamified"),
        _extract_key_row("H7", "forecast_error", results.get("h7_forecast_fe"), "pc_x_gamified"),
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{OUTPUT_DIR}/results_summary.csv", index=False)
    print(summary)

    # Save notices
    for path in [
        f"{OUTPUT_DIR}/h1_results.csv",
        f"{OUTPUT_DIR}/h2_results.csv",
        f"{OUTPUT_DIR}/h3_results.csv",
        f"{OUTPUT_DIR}/h4_results.csv",
        f"{OUTPUT_DIR}/h5_results.csv",
        f"{OUTPUT_DIR}/h6_results.csv",
        f"{OUTPUT_DIR}/h7_results.csv",
        f"{OUTPUT_DIR}/results_summary.csv",
    ]:
        print(f"[SAVE] {path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
