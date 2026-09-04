"""
Relative mispricing by treatment, two normalizations:

  AMR (Palan 2013):        mean_k [ mean_t |p_t - v_k| / v_k ]
  RAD (Stockl et al 2010): mean_k [ |Pbar_k - v_k| ] / vbar,  vbar = mean_k v_k
  RD  (Stockl et al 2010): mean_k [  Pbar_k - v_k  ] / vbar   (signed)

Full sample (all include:true sessions), excluding the outlier group
20260520_PM/ng1. Scratch analysis — prints results and writes a JSON
sidecar next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent / "amr_by_treatment.json"

TREATS = ["ng", "gh", "gp", "ghp"]
EXCLUDE_GROUPS = {"20260520_PM/ng1", "20280904/ghp1"}

mkt = pd.read_csv(PROCESSED / "market_day_panel_full.csv")
trd = pd.read_csv(PROCESSED / "trader_day_panel_full.csv")

with open(ROOT / "config" / "sessions.yaml") as fh:
    sessions_cfg = yaml.safe_load(fh)["sessions"]
code2id = {c: s["id"] for s in sessions_cfg for c in s["oTree_codes"]}

# --- group labels (same convention as preliminary_hypothesis_analysis.py)
grp_key = (
    trd.groupby("market_uuid")["participant_code"]
    .apply(lambda s: "|".join(sorted(s.unique())))
    .rename("group_key")
)
mkt = mkt.merge(grp_key, left_on="market_uuid", right_index=True)
trd = trd.merge(grp_key, left_on="market_uuid", right_index=True)
sess_map = trd.groupby("group_key")["session_code"].first()

groups = (
    mkt.groupby("group_key")
    .agg(treatment=("treatment", "first"))
    .join(sess_map)
    .reset_index()
)
groups["session_id"] = groups["session_code"].map(code2id)
groups = groups.sort_values(["session_id", "treatment"]).reset_index(drop=True)
groups["group_label"] = (
    groups["session_id"].astype(str)
    + "/"
    + groups["treatment"]
    + groups.groupby(["session_id", "treatment"]).cumcount().add(1).astype(str)
)
glabel = groups.set_index("group_key")["group_label"]
mkt["group_label"] = mkt["group_key"].map(glabel)
mkt["session_id"] = mkt["group_key"].map(groups.set_index("group_key")["session_id"])

mkt = mkt[~mkt.group_label.isin(EXCLUDE_GROUPS)].copy()
groups = groups[~groups.group_label.isin(EXCLUDE_GROUPS)].copy()

res: dict = {
    "excluded": sorted(EXCLUDE_GROUPS),
    "n_groups_by_treatment": groups.treatment.value_counts().to_dict(),
}

# --- market-rep AMR / RAD / RD (means across periods with trades)
# RAD normalizes |Pbar_k - v_k| by the horizon-average fundamental vbar,
# so end-of-horizon deviations are not inflated by a shrinking denominator.
VBAR = mkt.groupby("trading_day")["fundamental_value"].first().mean()  # = 64
mkt["abs_dev"] = mkt["avg_mispricing"].abs()  # |Pbar_k - v_k|

mrep = (
    mkt.groupby(["market_uuid", "group_label", "treatment", "repetition", "session_id"])
    .agg(amr=("abs_mispricing_ratio", "mean"),
         rad_num=("abs_dev", "mean"),
         rd_num=("avg_mispricing", "mean"),
         n_days_traded=("abs_mispricing_ratio", "count"))
    .reset_index()
)
mrep["rad"] = mrep["rad_num"] / VBAR
mrep["rd"] = mrep["rd_num"] / VBAR
gp = (
    mrep.groupby(["group_label", "treatment"])[["amr", "rad", "rd"]]
    .mean()
    .reset_index()
)
gp["gamified"] = (gp.treatment != "ng").astype(int)
gp["hedonic"] = gp.treatment.isin(["gh", "ghp"]).astype(int)
gp["pn"] = gp.treatment.isin(["gp", "ghp"]).astype(int)


def mw(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return None
    return round(float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue), 4)


res["group_level"] = [
    {"group": g, "treatment": t,
     "amr": round(a, 3), "rad": round(rv, 3), "rd": round(dv, 3)}
    for g, t, a, rv, dv in gp[
        ["group_label", "treatment", "amr", "rad", "rd"]
    ].itertuples(index=False)
]

for var in ["amr", "rad", "rd"]:
    res[f"cells_{var}"] = {
        t: {
            "mean": round(gp.loc[gp.treatment == t, var].mean(), 3),
            "median": round(gp.loc[gp.treatment == t, var].median(), 3),
            "n_groups": int((gp.treatment == t).sum()),
        }
        for t in TREATS
    }
    res[f"tests_{var}"] = {
        "gamified_vs_ng": {
            "mean_gamified": round(gp.loc[gp.gamified == 1, var].mean(), 3),
            "mean_ng": round(gp.loc[gp.gamified == 0, var].mean(), 3),
            "n": [int((gp.gamified == 1).sum()), int((gp.gamified == 0).sum())],
            "p_mw_two_sided": mw(gp.loc[gp.gamified == 1, var], gp.loc[gp.gamified == 0, var]),
        },
        "ghp_vs_ng": {
            "p_mw_two_sided": mw(gp.loc[gp.treatment == "ghp", var], gp.loc[gp.treatment == "ng", var]),
        },
        "hedonic_vs_not": {
            "mean_1": round(gp.loc[gp.hedonic == 1, var].mean(), 3),
            "mean_0": round(gp.loc[gp.hedonic == 0, var].mean(), 3),
            "p_mw_two_sided": mw(gp.loc[gp.hedonic == 1, var], gp.loc[gp.hedonic == 0, var]),
        },
        "pn_vs_not": {
            "mean_1": round(gp.loc[gp.pn == 1, var].mean(), 3),
            "mean_0": round(gp.loc[gp.pn == 0, var].mean(), 3),
            "p_mw_two_sided": mw(gp.loc[gp.pn == 1, var], gp.loc[gp.pn == 0, var]),
        },
    }

# --- by repetition (experience)
for var in ["amr", "rad"]:
    rep = mrep.pivot_table(index=["group_label", "treatment"], columns="repetition",
                           values=var).reset_index()
    rep.columns = ["group_label", "treatment", "r1", "r2"]
    res[f"by_repetition_{var}"] = {
        t: {
            "r1": round(rep.loc[rep.treatment == t, "r1"].mean(), 3),
            "r2": round(rep.loc[rep.treatment == t, "r2"].mean(), 3),
        }
        for t in TREATS
    }

# --- time paths by trading day (pooled reps): AMR and RAD contribution
mkt["rad_day"] = mkt["abs_dev"] / VBAR  # |Pbar_k - v_k| / vbar
path = (
    mkt.groupby(["treatment", "trading_day"])[["abs_mispricing_ratio", "rad_day"]]
    .mean()
    .reset_index()
)
res["paths"] = [
    {"treatment": row.treatment, "day": int(row.trading_day),
     "amr": round(row.abs_mispricing_ratio, 4), "rad": round(row.rad_day, 4)}
    for row in path.itertuples()
    if pd.notna(row.abs_mispricing_ratio)
]

# --- per-market detail
res["market_reps"] = [
    {"group": row.group_label, "treatment": row.treatment, "rep": int(row.repetition),
     "amr": round(row.amr, 3), "rad": round(row.rad, 3), "rd": round(row.rd, 3),
     "days_traded": int(row.n_days_traded)}
    for row in mrep.sort_values(["treatment", "group_label", "repetition"]).itertuples()
]

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: res[k] for k in
                  ["excluded", "n_groups_by_treatment",
                   "cells_rad", "tests_rad", "cells_rd", "by_repetition_rad"]},
                 indent=1))
print("saved", OUT)
