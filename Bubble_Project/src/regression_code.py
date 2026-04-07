# regression_code.py

# This script estimates the empirical models for Hypotheses 1–7 and exports regression tables.

# Hypotheses:
# H1: Bubble size
# H2: Volatility
# H3: Wealth inequality
# H4: Experience
# H5: Feedback trading
# H6: Algo-trading attenuation
# H7: Beliefs / trend extrapolation

from pathlib import Path
import warnings
import pandas as pd
import statsmodels.formula.api as smf

from table_utils import regression_table_to_latex, summary_stats_to_latex

DATA_DIR = Path("processed_data")
TABLE_DIR = Path("tables")
TABLE_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")


def load_panels():
    out = {}
    for fn in [
        "market_period.csv",
        "market_summary.csv",
        "forecast_panel.csv",
        "trader_types.csv",
        "market_type_shares.csv",
        "trader_final.csv",
        "trader_period_h5.csv",
    ]:
        path = DATA_DIR / fn
        if path.exists():
            out[fn.replace(".csv", "")] = pd.read_csv(path)
    return out


def fit_clustered_ols(formula, data, cluster_col="market_id"):
    d = data.dropna().copy()
    if d.empty:
        return None

    if cluster_col in d.columns and d[cluster_col].nunique() >= 2:
        return smf.ols(formula, data=d).fit(
            cov_type="cluster",
            cov_kwds={"groups": d[cluster_col]},
        )

    return smf.ols(formula, data=d).fit(cov_type="HC1")


def fit_count_model(formula, data, cluster_col="market_id"):
    d = data.dropna().copy()
    if d.empty:
        return None

    if cluster_col in d.columns and d[cluster_col].nunique() >= 2:
        return smf.poisson(formula, data=d).fit(
            disp=False, cov_type="cluster", cov_kwds={"groups": d[cluster_col]}
        )

    return smf.poisson(formula, data=d).fit(disp=False, cov_type="HC1")


def check_treatment_variation(df):
    cols = [
        c for c in ["market_id", "gamified", "hybrid", "repetition"] if c in df.columns
    ]
    out = df[cols].drop_duplicates().copy()

    has_gamified_var = (
        out["gamified"].nunique(dropna=True) >= 2
        if "gamified" in out.columns
        else False
    )
    has_hybrid_var = (
        out["hybrid"].nunique(dropna=True) >= 2 if "hybrid" in out.columns else False
    )
    return has_gamified_var and has_hybrid_var


def make_summary_stats(panels):
    if "market_summary" in panels:
        ms = panels["market_summary"].copy()
        cols = [
            c
            for c in [
                "mean_average_mispricing",
                "mean_absolute_mispricing",
                "mean_abs_mispricing_ratio",
                "n_surges",
                "n_crashes",
                "n_bubble_runs",
                "wealth_sd",
                "wealth_gini",
            ]
            if c in ms.columns
        ]

        if cols:
            # Force numeric so describe() returns numeric summary stats
            ms[cols] = ms[cols].apply(pd.to_numeric, errors="coerce")

            # Keep only columns with at least one non-missing numeric value
            cols = [c for c in cols if ms[c].notna().any()]

            if cols:
                summary_stats_to_latex(
                    ms,
                    TABLE_DIR / "summary_stats.tex",
                    columns=cols,
                    title="Summary Statistics",
                )


def run_h1_bubble_size(panels):
    mp = panels["market_period"].copy()
    if not check_treatment_variation(mp):
        return None

    return fit_clustered_ols(
        "abs_mispricing_ratio ~ gamified * hybrid + C(repetition)",
        mp[["abs_mispricing_ratio", "gamified", "hybrid", "repetition", "market_id"]],
    )


def run_h2_volatility(panels):
    ms = panels["market_summary"].copy()
    if not check_treatment_variation(ms):
        return None, None

    m1 = fit_count_model(
        "surges_crashes_total ~ gamified * hybrid + C(repetition)",
        ms[["surges_crashes_total", "gamified", "hybrid", "repetition", "market_id"]],
    )
    m2 = fit_count_model(
        "n_bubble_runs ~ gamified * hybrid + C(repetition)",
        ms[["n_bubble_runs", "gamified", "hybrid", "repetition", "market_id"]],
    )
    return m1, m2


def run_h3_wealth_inequality(panels):
    ms = panels["market_summary"].copy()
    if not check_treatment_variation(ms):
        return None, None

    m1 = fit_clustered_ols(
        "wealth_sd ~ gamified * hybrid + C(repetition)",
        ms[["wealth_sd", "gamified", "hybrid", "repetition", "market_id"]],
    )
    m2 = fit_clustered_ols(
        "wealth_gini ~ gamified * hybrid + C(repetition)",
        ms[["wealth_gini", "gamified", "hybrid", "repetition", "market_id"]],
    )
    return m1, m2


def run_h4_experience(panels):
    ms = panels["market_summary"].copy()
    if "repetition2" not in ms.columns:
        ms["repetition2"] = (ms["repetition"] == 2).astype(int)

    if ms["gamified"].nunique() < 2 or ms["repetition2"].nunique() < 2:
        return None

    return fit_clustered_ols(
        "mean_abs_mispricing_ratio ~ gamified * repetition2 + hybrid",
        ms[
            [
                "mean_abs_mispricing_ratio",
                "gamified",
                "repetition2",
                "hybrid",
                "market_id",
            ]
        ],
    )


def run_h5_feedback_trading(panels):
    mts = panels["market_type_shares"].copy()
    if not check_treatment_variation(mts):
        return None, None

    m1 = fit_clustered_ols(
        "share_feedback ~ gamified * hybrid + C(repetition)",
        mts[["share_feedback", "gamified", "hybrid", "repetition", "market_id"]],
    )
    m2 = fit_clustered_ols(
        "share_fundamental ~ gamified * hybrid + C(repetition)",
        mts[["share_fundamental", "gamified", "hybrid", "repetition", "market_id"]],
    )
    return m1, m2


def run_h6_algo_interaction(panels):
    ms = panels["market_summary"].copy()
    if not check_treatment_variation(ms):
        return None, None, None

    m1 = fit_clustered_ols(
        "mean_abs_mispricing_ratio ~ gamified * hybrid + C(repetition)",
        ms[
            [
                "mean_abs_mispricing_ratio",
                "gamified",
                "hybrid",
                "repetition",
                "market_id",
            ]
        ],
    )
    m2 = fit_count_model(
        "surges_crashes_total ~ gamified * hybrid + C(repetition)",
        ms[["surges_crashes_total", "gamified", "hybrid", "repetition", "market_id"]],
    )
    m3 = fit_clustered_ols(
        "wealth_gini ~ gamified * hybrid + C(repetition)",
        ms[["wealth_gini", "gamified", "hybrid", "repetition", "market_id"]],
    )
    return m1, m2, m3


def run_h7_beliefs(panels):
    fp = panels["forecast_panel"].copy()
    required = ["forecast_gap", "delta_p", "gamified", "hybrid", "market_id"]
    if not set(required).issubset(fp.columns):
        return None

    check_cols = ["market_id", "gamified", "hybrid"]
    if "repetition" in fp.columns:
        check_cols.append("repetition")

    if not check_treatment_variation(fp[check_cols].drop_duplicates()):
        return None

    keep_cols = ["forecast_gap", "delta_p", "gamified", "hybrid", "market_id"]
    for c in [
        "z_age",
        "z_fin_quiz_score",
        "z_overconfidence",
        "z_trading_experience",
        "z_risk_aversion",
        "trading_day",
        "repetition",
    ]:
        if c in fp.columns:
            keep_cols.append(c)

    d = fp[keep_cols].copy()

    controls = []
    for c in [
        "z_age",
        "z_fin_quiz_score",
        "z_overconfidence",
        "z_trading_experience",
        "z_risk_aversion",
        "trading_day",
    ]:
        if c in d.columns:
            controls.append(c)

    rhs = "delta_p * gamified * hybrid"
    if "repetition" in d.columns:
        rhs += " + C(repetition)"
    if controls:
        rhs += " + " + " + ".join(controls)

    return fit_clustered_ols(f"forecast_gap ~ {rhs}", d)


def export_tables(panels):
    variable_labels = {
        "Intercept": "Constant",
        "gamified": "Gamified",
        "hybrid": "Hybrid",
        "gamified:hybrid": "Gamified × Hybrid",
        "C(repetition)[T.2]": "Repetition 2",
        "repetition2": "Repetition 2",
        "gamified:repetition2": "Gamified × Repetition 2",
        "delta_p": "Recent price change",
        "delta_p:gamified": "Price change × Gamified",
        "delta_p:hybrid": "Price change × Hybrid",
        "delta_p:gamified:hybrid": "Price change × Gamified × Hybrid",
        "z_age": "Age (z)",
        "z_fin_quiz_score": "Financial quiz (z)",
        "z_overconfidence": "Overconfidence (z)",
        "z_trading_experience": "Trading experience (z)",
        "z_risk_aversion": "Risk aversion (z)",
        "trading_day": "Trading day",
    }

    make_summary_stats(panels)

    h1 = run_h1_bubble_size(panels)
    if h1 is not None:
        regression_table_to_latex(
            [h1],
            ["(1)"],
            TABLE_DIR / "bubble_size_table.tex",
            title="Bubble Size",
            variable_labels=variable_labels,
        )

    h2a, h2b = run_h2_volatility(panels)
    if h2a is not None and h2b is not None:
        regression_table_to_latex(
            [h2a, h2b],
            ["(1)", "(2)"],
            TABLE_DIR / "volatility_table.tex",
            title="Volatility",
            variable_labels=variable_labels,
        )

    h3a, h3b = run_h3_wealth_inequality(panels)
    if h3a is not None and h3b is not None:
        regression_table_to_latex(
            [h3a, h3b],
            ["(1)", "(2)"],
            TABLE_DIR / "wealth_inequality_table.tex",
            title="Wealth Inequality",
            variable_labels=variable_labels,
        )

    h4 = run_h4_experience(panels)
    if h4 is not None:
        regression_table_to_latex(
            [h4],
            ["(1)"],
            TABLE_DIR / "experience_table.tex",
            title="Experience",
            variable_labels=variable_labels,
        )

    h5a, h5b = run_h5_feedback_trading(panels)
    if h5a is not None and h5b is not None:
        regression_table_to_latex(
            [h5a, h5b],
            ["(1)", "(2)"],
            TABLE_DIR / "feedback_trading_table.tex",
            title="Feedback Trading",
            variable_labels=variable_labels,
        )

    h6a, h6b, h6c = run_h6_algo_interaction(panels)
    if h6a is not None and h6b is not None and h6c is not None:
        regression_table_to_latex(
            [h6a, h6b, h6c],
            ["(1)", "(2)", "(3)"],
            TABLE_DIR / "algo_attenuation_table.tex",
            title="Algo Trading Attenuation",
            variable_labels=variable_labels,
        )

    h7 = run_h7_beliefs(panels)
    if h7 is not None:
        regression_table_to_latex(
            [h7],
            ["(1)"],
            TABLE_DIR / "beliefs_table.tex",
            title="Beliefs",
            variable_labels=variable_labels,
        )


def main():
    panels = load_panels()
    export_tables(panels)
    print(f"Saved tables to: {TABLE_DIR.resolve()}")


if __name__ == "__main__":
    main()
