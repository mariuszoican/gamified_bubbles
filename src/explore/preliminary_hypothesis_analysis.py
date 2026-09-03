"""
Preliminary hypothesis-by-hypothesis look at the Gamified Bubbles sample.

Scratch analysis (src/explore) — does NOT feed the paper pipeline.
Reads data/processed/*_full.csv plus raw gamification_ui exports
(manipulation checks) and writes a JSON summary next to this script.

Statistical philosophy for the preliminary sample:
  - The independent unit is the participant group (6 subjects trading
    twice). All headline tests are Mann-Whitney U on group-level means.
  - Two-sided p-values are reported; hypotheses are directional, so
    one-sided values are half of these when the sign agrees.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = Path(__file__).resolve().parent / "prelim_results.json"

TREATS = ["ng", "gh", "gp", "ghp"]

mkt = pd.read_csv(PROCESSED / "market_day_panel_full.csv")
trd = pd.read_csv(PROCESSED / "trader_day_panel_full.csv")
pay = pd.read_csv(PROCESSED / "participant_payments_full.csv")

with open(ROOT / "config" / "sessions.yaml") as fh:
    sessions_cfg = yaml.safe_load(fh)["sessions"]

results: dict = {}


def r(x, nd=3):
    """Round scalars, tolerate NaN."""
    if x is None:
        return None
    try:
        if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def mw(a, b):
    """Two-sided Mann-Whitney U (returns None if either side empty)."""
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return None
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return None


# ---------------------------------------------------------------- sample map
# market_uuid identifies a market-repetition; a "group" is the set of six
# participants who trade two consecutive repetitions.
grp_key = (
    trd.groupby("market_uuid")["participant_code"]
    .apply(lambda s: "|".join(sorted(s.unique())))
    .rename("group_key")
)
mkt = mkt.merge(grp_key, left_on="market_uuid", right_index=True)
trd = trd.merge(grp_key, left_on="market_uuid", right_index=True)

sess_map = trd.groupby("group_key")["session_code"].first()
code2id = {c: s["id"] for s in sessions_cfg for c in s["oTree_codes"]}

groups = (
    mkt.groupby("group_key")
    .agg(treatment=("treatment", "first"))
    .join(sess_map)
    .reset_index()
)
groups["session_id"] = groups["session_code"].map(code2id)
groups = groups.sort_values(["session_id", "treatment"]).reset_index(drop=True)
groups["group_label"] = [
    f"{row.session_id}-{row.treatment}{i}"
    for i, row in enumerate(groups.itertuples(), 1)
]
# simpler labels: session + treatment + running index within session-treatment
groups["group_label"] = (
    groups["session_id"].astype(str)
    + "/"
    + groups["treatment"]
    + groups.groupby(["session_id", "treatment"]).cumcount().add(1).astype(str)
)
glabel = groups.set_index("group_key")["group_label"]
mkt["group_label"] = mkt["group_key"].map(glabel)
trd["group_label"] = trd["group_key"].map(glabel)
mkt["session_id"] = mkt["group_key"].map(groups.set_index("group_key")["session_id"])

# ---------------------------------------------------------- sample filters
# KEEP_SESSIONS: restrict the analysis sample to these session ids
# (None = all included sessions).
KEEP_SESSIONS: set[str] | None = {"20260826"}

if KEEP_SESSIONS is not None:
    keep_keys = groups.loc[groups.session_id.isin(KEEP_SESSIONS), "group_key"]
    mkt = mkt[mkt.group_key.isin(keep_keys)].copy()
    trd = trd[trd.group_key.isin(keep_keys)].copy()
    groups = groups[groups.group_key.isin(keep_keys)].copy()

# 20260520_PM/ng1: ~3x the trades of any other market (495 + 578),
# AMR ~2.6 in both repetitions, Gini 0.218. Excluded on request;
# summary stats of the dropped group retained for transparency.
# (No-op when KEEP_SESSIONS already drops that session.)
EXCLUDE_GROUPS = {"20260520_PM/ng1"}

_excl = mkt[mkt.group_label.isin(EXCLUDE_GROUPS)]
excluded_stats = []
for g, sub in _excl.groupby("group_label"):
    per_rep = sub.groupby("repetition").agg(
        trades=("n_trades_market", "sum"),
        amr=("abs_mispricing_ratio", "mean"),
        peak_price=("avg_trade_price", "max"),
    )
    excluded_stats.append(
        {
            "group": g,
            "treatment": sub.treatment.iloc[0],
            "trades_r1": int(per_rep.trades.get(1, 0)),
            "trades_r2": int(per_rep.trades.get(2, 0)),
            "amr_r1": r(per_rep.amr.get(1), 2),
            "amr_r2": r(per_rep.amr.get(2), 2),
            "peak_avg_price": r(per_rep.peak_price.max(), 0),
        }
    )

mkt = mkt[~mkt.group_label.isin(EXCLUDE_GROUPS)].copy()
trd = trd[~trd.group_label.isin(EXCLUDE_GROUPS)].copy()
groups = groups[~groups.group_label.isin(EXCLUDE_GROUPS)].copy()

results["sample"] = {
    "sessions": [
        {
            "id": s["id"],
            "included": bool(s.get("include", False)),
            "notes": s.get("notes", ""),
        }
        for s in sessions_cfg
    ],
    "groups_by_treatment": groups["treatment"].value_counts().to_dict(),
    "n_groups": int(groups.shape[0]),
    "n_market_reps": int(mkt["market_uuid"].nunique()),
    "n_traders": int(trd["participant_code"].nunique()),
    "n_paid": (
        int(pay.shape[0])
        if KEEP_SESSIONS is None
        else int(
            sum(
                pd.read_csv(ROOT / "data" / "interim" / s / "participant_payments.csv").shape[0]
                for s in KEEP_SESSIONS
            )
        )
    ),
    "market_day_rows": int(mkt.shape[0]),
    "kept_sessions": sorted(KEEP_SESSIONS) if KEEP_SESSIONS else "all",
    "excluded_outlier": excluded_stats,
}

# share conservation sanity check (90 shares per market)
shares_ok = (
    trd.groupby(["market_uuid", "trading_day"])["num_shares"].sum().round(2).unique()
)
results["sample"]["shares_outstanding_check"] = [float(x) for x in shares_ok]

# ---------------------------------------------------------------- balance
bal_vars = {
    "fin_quiz_score": "Financial literacy (share correct)",
    "self_assessment": "Self-assessed literacy (0-10)",
    "overconfidence": "Overconfidence",
    "age": "Age",
    "gender_female": "Female",
    "trading_experience": "Trading experience",
    "finance_course": "Finance course",
    "cq_attempt_count": "Comprehension-quiz attempts",
}
tr1 = trd.drop_duplicates("participant_code")
balance = []
for var, label in bal_vars.items():
    cells = {t: tr1.loc[tr1.treatment == t, var].dropna() for t in TREATS}
    kw = stats.kruskal(*[c for c in cells.values() if len(c)])
    balance.append(
        {
            "var": label,
            **{t: r(cells[t].mean(), 2) for t in TREATS},
            "kw_p": r(kw.pvalue, 3),
        }
    )
results["balance"] = balance

# ------------------------------------------------- manipulation checks (raw)
badge_rows, alert_rows = [], []
for s in sessions_cfg:
    if not s.get("include", False):
        continue
    f = (
        RAW
        / s["id"]
        / f"trader_bridge_app_custom_export_gamification_ui_{s['export_date']}.csv"
    )
    if not f.exists():
        continue
    g = pd.read_csv(f)
    g = g[g.trading_session_uuid.isin(mkt.market_uuid.unique())]
    g["session_id"] = s["id"]
    badge_rows.append(g)

gui = pd.concat(badge_rows, ignore_index=True)
gui = gui.merge(
    mkt[["market_uuid", "treatment", "repetition"]].drop_duplicates(),
    left_on="trading_session_uuid",
    right_on="market_uuid",
)

results["manipulation"] = {
    "event_types": gui.groupby(["element_type", "event_name"])
    .size()
    .reset_index()
    .rename(columns={0: "n"})
    .to_dict("records"),
    "by_treatment": gui.groupby(["treatment", "element_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
    .to_dict("records"),
    "alert_messages": gui.loc[gui.element_type == "trend_alert", "payload_json"]
    .str.extract(r'"message": "([^"]+)"')[0]
    .value_counts()
    .head(6)
    .to_dict(),
}

# badge share among hedonic traders: trades per trader per market-rep
tpt = (
    trd.groupby(["market_uuid", "participant_code", "treatment", "group_label"])[
        ["n_buys", "n_sells"]
    ]
    .sum()
    .assign(n_trades=lambda d: d.n_buys + d.n_sells)
    .reset_index()
)
thresholds = {"bronze_10": 10, "silver_15": 15, "gold_35": 35, "plat_50": 50, "diam_60": 60}
badge_att = {}
for t in TREATS:
    sub = tpt[tpt.treatment == t]
    badge_att[t] = {
        "mean_trades_per_trader": r(sub.n_trades.mean(), 1),
        "median_trades_per_trader": r(sub.n_trades.median(), 1),
        **{k: r((sub.n_trades >= v).mean(), 2) for k, v in thresholds.items()},
    }
results["manipulation"]["trades_vs_badges"] = badge_att

# ------------------------------------------------------------- group panel
# market-rep aggregates
mrep = (
    mkt.groupby(["market_uuid", "group_label", "treatment", "repetition", "session_id"])
    .agg(
        avg_mispricing=("avg_mispricing", "mean"),
        abs_mispricing=("avg_abs_mispricing", "mean"),
        amr=("abs_mispricing_ratio", "mean"),
        n_trades=("n_trades_market", "sum"),
        bubble_periods=("bubble_period", "sum"),
        surges=("surge", "sum"),
        crashes=("crash", "sum"),
        gini_last=("gini", "last"),
        sd_wealth_last=("sd_wealth", "last"),
        share_feedback=("share_feedback", "first"),
        share_speculator=("share_speculator", "first"),
        share_fundamental=("share_fundamental", "first"),
        share_other=("share_other", "first"),
    )
    .reset_index()
)
mrep["turnover"] = mrep["n_trades"] / 90.0  # 90 shares outstanding
mrep["surge_crash"] = mrep["surges"] + mrep["crashes"]
mrep["gamified"] = (mrep.treatment != "ng").astype(int)
mrep["hedonic"] = mrep.treatment.isin(["gh", "ghp"]).astype(int)
mrep["pn"] = mrep.treatment.isin(["gp", "ghp"]).astype(int)

# pooled (paper-definition) surge/crash/bubble flags: mean/sd across ALL
# markets within the same repetition, full sample
mkt["_ret"] = mkt["return"]
mkt["_nm"] = mkt["normalized_mispricing"]
for col, flag_hi, flag_lo in [("_ret", "surge_p", "crash_p"), ("_nm", "bubble_p", None)]:
    mu = mkt.groupby("repetition")[col].transform("mean")
    sd = mkt.groupby("repetition")[col].transform("std")
    mkt[flag_hi] = (mkt[col] > mu + 2 * sd).astype(int)
    if flag_lo:
        mkt[flag_lo] = (mkt[col] < mu - 2 * sd).astype(int)

pooled = (
    mkt.groupby("market_uuid")[["surge_p", "crash_p", "bubble_p"]].sum().reset_index()
)
mrep = mrep.merge(pooled, on="market_uuid")
mrep["surge_crash_pooled"] = mrep["surge_p"] + mrep["crash_p"]

# group-level means across the two repetitions (independent observations)
gp_num = mrep.groupby(["group_label", "treatment"]).mean(numeric_only=True).reset_index()
gp_num["gamified"] = (gp_num.treatment != "ng").astype(int)
gp_num["hedonic"] = gp_num.treatment.isin(["gh", "ghp"]).astype(int)
gp_num["pn"] = gp_num.treatment.isin(["gp", "ghp"]).astype(int)


def cellmeans(df, var, nd=3):
    return {t: r(df.loc[df.treatment == t, var].mean(), nd) for t in TREATS}


def split_test(df, var, dummy):
    """Group-level MW test of var by dummy (1 vs 0)."""
    a = df.loc[df[dummy] == 1, var]
    b = df.loc[df[dummy] == 0, var]
    return {
        "mean1": r(a.mean()),
        "mean0": r(b.mean()),
        "n1": int(a.notna().sum()),
        "n0": int(b.notna().sum()),
        "p": r(mw(a, b)),
    }


# ---------------------------------------------------------------- H1
results["h1"] = {
    "cells": {
        "avg_mispricing": cellmeans(gp_num, "avg_mispricing", 1),
        "abs_mispricing": cellmeans(gp_num, "abs_mispricing", 1),
        "amr": cellmeans(gp_num, "amr", 3),
        "bubble_periods_pooled": cellmeans(gp_num, "bubble_p", 2),
    },
    "gamified_vs_ng": {
        "avg_mispricing": split_test(gp_num, "avg_mispricing", "gamified"),
        "abs_mispricing": split_test(gp_num, "abs_mispricing", "gamified"),
        "amr": split_test(gp_num, "amr", "gamified"),
        "bubble_periods_pooled": split_test(gp_num, "bubble_p", "gamified"),
    },
}

# ---------------------------------------------------------------- H1a experience
piv = mrep.pivot_table(index=["group_label", "treatment"], columns="repetition",
                       values="amr").reset_index()
piv.columns = ["group_label", "treatment", "amr_r1", "amr_r2"]
piv["d_amr"] = piv["amr_r2"] - piv["amr_r1"]
piv["gamified"] = (piv.treatment != "ng").astype(int)

rep_cells = {
    t: {
        "r1": r(piv.loc[piv.treatment == t, "amr_r1"].mean()),
        "r2": r(piv.loc[piv.treatment == t, "amr_r2"].mean()),
    }
    for t in TREATS
}
wsr = {}
for lab, sub in [("gamified", piv[piv.gamified == 1]), ("ng", piv[piv.gamified == 0])]:
    d = sub["d_amr"].dropna()
    wsr[lab] = {
        "mean_delta": r(d.mean()),
        "p_wilcoxon": r(stats.wilcoxon(d).pvalue) if len(d) >= 5 else None,
        "n": int(len(d)),
    }
results["h1a"] = {
    "amr_by_rep": rep_cells,
    "delta_within": wsr,
    "dd_gamified_vs_ng": split_test(piv, "d_amr", "gamified"),
}

# ---------------------------------------------------------------- H2 volume
results["h2"] = {
    "cells_trades": cellmeans(gp_num, "n_trades", 1),
    "cells_turnover": cellmeans(gp_num, "turnover", 2),
    "gamified_vs_ng": split_test(gp_num, "n_trades", "gamified"),
    "hedonic_vs_not": split_test(gp_num, "n_trades", "hedonic"),
    "pn_vs_not": split_test(gp_num, "n_trades", "pn"),
}

# ---------------------------------------------------------------- H3 inequality
# Primary measures rebuilt from the MBO trade stream + realized dividends
# (panel wealth uses the unreliable player.num_shares snapshot).
recw = pd.read_csv(Path(__file__).resolve().parent / "recon_final_wealth.csv")
recw = recw.merge(
    mrep[["market_uuid", "group_label", "repetition"]], on="market_uuid"
)


def gini_v(v):
    v = np.sort(np.asarray(v, float))
    n = len(v)
    return float(np.abs(v[:, None] - v[None, :]).sum() / (2 * n * n * v.mean()))


ineq = (
    recw.groupby(["market_uuid", "group_label", "treatment"])
    .agg(
        gini_recon=("cash", gini_v),
        sd_recon=("cash", "std"),
    )
    .reset_index()
)
ineq_grp = ineq.groupby(["group_label", "treatment"]).mean(numeric_only=True).reset_index()
ineq_grp["gamified"] = (ineq_grp.treatment != "ng").astype(int)

# mechanism: literacy vs within-market relative payoff (trader-market obs)
lit = trd.drop_duplicates(["market_uuid", "participant_code"])[
    ["market_uuid", "participant_code", "trader_uuid", "treatment", "fin_quiz_score"]
]
rw = recw.merge(lit, on=["market_uuid", "trader_uuid"], suffixes=("", "_l"))
rw["cash_dm"] = rw.groupby("market_uuid")["cash"].transform(lambda x: x - x.mean())
corr = {}
for lab, sub in [("gamified", rw[rw.treatment != "ng"]), ("ng", rw[rw.treatment == "ng"])]:
    rho, pv = stats.spearmanr(sub.fin_quiz_score, sub.cash_dm, nan_policy="omit")
    corr[lab] = {"spearman_rho": r(rho), "p": r(pv), "n": int(sub.shape[0])}

results["h3"] = {
    "cells_gini_recon": {
        t: r(ineq_grp.loc[ineq_grp.treatment == t, "gini_recon"].mean(), 3)
        for t in TREATS
    },
    "cells_sd_recon": {
        t: r(ineq_grp.loc[ineq_grp.treatment == t, "sd_recon"].mean(), 0)
        for t in TREATS
    },
    "cells_gini_panel_recon": cellmeans(gp_num, "gini_last", 3),
    "gamified_vs_ng_gini_recon": split_test(ineq_grp, "gini_recon", "gamified"),
    "gamified_vs_ng_sd_recon": split_test(ineq_grp, "sd_recon", "gamified"),
    "literacy_relpayoff_corr": corr,
}

# ---------------------------------------------------------------- H4 trader types
results["h4"] = {
    "cells": {
        v: cellmeans(gp_num, v, 3)
        for v in ["share_feedback", "share_speculator", "share_fundamental", "share_other"]
    },
    "feedback_by_pn": split_test(gp_num, "share_feedback", "pn"),
    "other_by_hedonic": split_test(gp_num, "share_other", "hedonic"),
    "fundamental_by_gamified": split_test(gp_num, "share_fundamental", "gamified"),
}

# ---------------------------------------------------------------- H5 surges/crashes
results["h5"] = {
    "cells_sessionflags": cellmeans(gp_num, "surge_crash", 2),
    "cells_pooledflags": cellmeans(gp_num, "surge_crash_pooled", 2),
    "pn_vs_not_session": split_test(gp_num, "surge_crash", "pn"),
    "pn_vs_not_pooled": split_test(gp_num, "surge_crash_pooled", "pn"),
    "flag_totals": {
        "session_def": int(mrep.surge_crash.sum()),
        "pooled_def": int(mrep.surge_crash_pooled.sum()),
        "bubble_session_def": int(mrep.bubble_periods.sum()),
        "bubble_pooled_def": int(mrep.bubble_p.sum()),
    },
}

# ---------------------------------------------------------------- H6 beliefs
fc = trd.dropna(subset=["forecast"]).copy()
fc["fv_next"] = fc["fundamental_value"] - 8
fc["err_fv"] = (fc["forecast"] - fc["fv_next"]).abs() / fc["fv_next"]
fc["err_realized"] = (fc["forecast"] - fc["price_next"]).abs() / fc["price_next"]
fc["dep"] = fc["forecast"] - fc["closing_price"]
fc["dP"] = fc["closing_price"] - fc["price_L1"]
fc["pn"] = fc.treatment.isin(["gp", "ghp"]).astype(int)
fc["hedonic"] = fc.treatment.isin(["gh", "ghp"]).astype(int)

results["h6"] = {
    "n_forecasts": int(fc.shape[0]),
    "forecast_days": sorted(fc.trading_day.unique().tolist()),
    "err_fv_cells": {
        t: r(fc.loc[fc.treatment == t, "err_fv"].mean(), 3) for t in TREATS
    },
    "err_realized_cells": {
        t: r(fc.loc[fc.treatment == t, "err_realized"].mean(), 3) for t in TREATS
    },
}

# group-level forecast error test (hedonic prediction)
fe_grp = fc.groupby(["group_label", "treatment"])["err_fv"].mean().reset_index()
fe_grp["hedonic"] = fe_grp.treatment.isin(["gh", "ghp"]).astype(int)
fe_grp["gamified"] = (fe_grp.treatment != "ng").astype(int)
results["h6"]["err_fv_hedonic_test"] = split_test(fe_grp, "err_fv", "hedonic")
results["h6"]["err_fv_gamified_test"] = split_test(fe_grp, "err_fv", "gamified")

# trend extrapolation: slope of (F - P_k) on dP_k by arm + interaction
import statsmodels.formula.api as smf  # noqa: E402

fx = fc.dropna(subset=["dep", "dP"]).copy()
slopes = {}
for lab, sub in [("pn1", fx[fx.pn == 1]), ("pn0", fx[fx.pn == 0])]:
    if len(sub) > 5:
        m = smf.ols("dep ~ dP", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["group_label"]}
        )
        slopes[lab] = {
            "beta": r(m.params["dP"]),
            "se": r(m.bse["dP"]),
            "p": r(m.pvalues["dP"]),
            "n": int(m.nobs),
        }
mi = smf.ols("dep ~ dP * pn", data=fx).fit(
    cov_type="cluster", cov_kwds={"groups": fx["group_label"]}
)
slopes["interaction"] = {
    "beta": r(mi.params["dP:pn"]),
    "se": r(mi.bse["dP:pn"]),
    "p": r(mi.pvalues["dP:pn"]),
}
mih = smf.ols("dep ~ dP * hedonic", data=fx).fit(
    cov_type="cluster", cov_kwds={"groups": fx["group_label"]}
)
slopes["interaction_hedonic"] = {
    "beta": r(mih.params["dP:hedonic"]),
    "se": r(mih.bse["dP:hedonic"]),
    "p": r(mih.pvalues["dP:hedonic"]),
}
results["h6"]["extrapolation"] = slopes

# ---------------------------------------------------------------- price paths
pp = (
    mkt.groupby(["treatment", "repetition", "trading_day"])
    .agg(price=("avg_trade_price", "mean"), n=("market_uuid", "nunique"))
    .reset_index()
)
pp["fv"] = 8 * (15 + 1 - pp.trading_day)
results["pricepaths"] = [
    {
        "treatment": row.treatment,
        "rep": int(row.repetition),
        "day": int(row.trading_day),
        "price": r(row.price, 1),
        "fv": int(row.fv),
        "n": int(row.n),
    }
    for row in pp.itertuples()
]

# volume by day/treatment (mean trades per period)
vv = (
    mkt.groupby(["treatment", "trading_day"])["n_trades_market"].mean().reset_index()
)
results["volumepaths"] = [
    {"treatment": row.treatment, "day": int(row.trading_day), "trades": r(row.n_trades_market, 1)}
    for row in vv.itertuples()
]

# ---------------------------------------------------------------- markets table
mt = []
for g, sub in mrep.groupby("group_label"):
    sub = sub.sort_values("repetition")
    row = {
        "group": g,
        "session": sub.session_id.iloc[0],
        "treatment": sub.treatment.iloc[0],
        "trades_r1": int(sub.n_trades.iloc[0]),
        "trades_r2": int(sub.n_trades.iloc[1]) if len(sub) > 1 else None,
        "amr_r1": r(sub.amr.iloc[0], 3),
        "amr_r2": r(sub.amr.iloc[1], 3) if len(sub) > 1 else None,
        "gini": r(sub.gini_last.mean(), 3),
        "bubbles_pooled": int(sub.bubble_p.sum()),
        "surge_crash_pooled": int(sub.surge_crash_pooled.sum()),
    }
    mt.append(row)
results["markets_table"] = sorted(mt, key=lambda d: (d["session"], d["treatment"]))

# ---------------------------------------------------------------- data integrity
# quantified in the diagnostic pass; hard-coded summary constants recomputed
# there (see chat/forensics): share sums, zero-trade phantom shares, excess
# dividends. Recompute the cheap ones here for reproducibility.
ssum = trd.groupby(["market_uuid", "trading_day"])["num_shares"].sum()
results["integrity"] = {
    "share_sum_min": r(ssum.min(), 0),
    "share_sum_max": r(ssum.max(), 0),
    "share_sum_mean": r(ssum.mean(), 1),
    "pct_market_days_off": r((ssum != 90).mean(), 3),
}

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("saved", OUT)
print(json.dumps(results["sample"], indent=1, default=str))
