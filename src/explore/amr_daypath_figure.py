"""
Day-by-day AMR and RAD by treatment (pooled across repetitions).

Same sample conventions as amr_by_treatment.py: all include:true sessions,
outlier group 20260520_PM/ng1 excluded.
  AMR_day = mean over markets of |Pbar_kt - v_t| / v_t
  RAD_day = mean over markets of |Pbar_kt - v_t| / vbar,  vbar = 64
Writes amr_daypath.png and rad_daypath.png next to this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
HERE = Path(__file__).resolve().parent

TREATS = ["ng", "gh", "gp", "ghp"]
LABELS = {
    "ng": "ng (control)",
    "gh": "gh (hedonic)",
    "gp": "gp (price notif.)",
    "ghp": "ghp (both)",
}
COLORS = {"ng": "#444444", "gh": "#d62728", "gp": "#1f77b4", "ghp": "#9467bd"}
EXCLUDE_GROUPS = {"20260520_PM/ng1", "20280904/ghp1"}

mkt = pd.read_csv(PROCESSED / "market_day_panel_full.csv")
trd = pd.read_csv(PROCESSED / "trader_day_panel_full.csv")

with open(ROOT / "config" / "sessions.yaml") as fh:
    sessions_cfg = yaml.safe_load(fh)["sessions"]
code2id = {c: s["id"] for s in sessions_cfg for c in s["oTree_codes"]}

# group labels, same convention as amr_by_treatment.py
grp_key = (
    trd.groupby("market_uuid")["participant_code"]
    .apply(lambda s: "|".join(sorted(s.unique())))
    .rename("group_key")
)
mkt = mkt.merge(grp_key, left_on="market_uuid", right_index=True)
sess_map = trd.merge(grp_key, left_on="market_uuid", right_index=True) \
    .groupby("group_key")["session_code"].first()

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
mkt["group_label"] = mkt["group_key"].map(groups.set_index("group_key")["group_label"])
mkt = mkt[~mkt.group_label.isin(EXCLUDE_GROUPS)].copy()

n_reps = mkt.groupby("treatment")["market_uuid"].nunique()

VBAR = mkt.groupby("trading_day")["fundamental_value"].first().mean()  # = 64
mkt["rad_day"] = mkt["avg_mispricing"].abs() / VBAR

path = (
    mkt.groupby(["treatment", "trading_day"])[["abs_mispricing_ratio", "rad_day"]]
    .mean()
    .reset_index()
)

SPECS = [
    dict(
        col="abs_mispricing_ratio",
        out="amr_daypath.png",
        ylabel=r"AMR:  mean$_k$ $|\bar{P}_{kt} - v_t|\,/\,v_t$",
        title="Day-by-day mispricing (AMR) by treatment, pooled repetitions",
        footnote=(
            "Excludes 20260520_PM/ng1. AMR denominator is v_t, which falls 120 \u2192 8, "
            "inflating late-day values mechanically; see RAD for the robust normalization."
        ),
    ),
    dict(
        col="rad_day",
        out="rad_daypath.png",
        ylabel=r"RAD:  mean$_k$ $|\bar{P}_{kt} - v_t|\,/\,\bar{v}$,  $\bar{v}=64$",
        title="Day-by-day mispricing (RAD) by treatment, pooled repetitions",
        footnote=(
            "Excludes 20260520_PM/ng1. RAD normalizes by the horizon-average "
            "fundamental (64), so late-day values are not inflated by the shrinking v_t."
        ),
    ),
]

for spec in SPECS:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(10.5, 15.5, color="0.92", zorder=0)

    for t in TREATS:
        sub = path[path.treatment == t]
        ax.plot(
            sub.trading_day, sub[spec["col"]],
            marker="o", ms=4, lw=1.8, color=COLORS[t],
            label=f"{LABELS[t]}  (n={n_reps[t]} reps)",
        )

    ax.set_xlabel("Trading day")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"])
    ax.set_xticks(range(1, 16))
    ax.legend(frameon=False, loc="upper left")
    ax.annotate("late window\n(days 11\u201315)", xy=(13, ax.get_ylim()[1] * 0.97),
                ha="center", va="top", fontsize=9, color="0.35")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, spec["footnote"], fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = HERE / spec["out"]
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)

    print(path.pivot(index="trading_day", columns="treatment",
                     values=spec["col"]).round(3)[TREATS])
