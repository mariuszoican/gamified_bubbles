"""
Volume, liquidity, and churn mechanisms for Gamified Bubbles.

Scratch analysis (src/explore) — does NOT feed the paper pipeline.

Question set (Marius, Aug 2026):
  1. Isolate the outlier ng group.
  2. Verify: gamified treatments trade much more, but price / mispricing /
     bubble effects are small. Report truthfully either way.
  3. Mechanisms: where does the extra volume go? Liquidity? Churn?

Statistical unit: the participant group (6 subjects, 2 market repetitions).
Headline tests are two-sided Mann-Whitney U on group-level means
(n = 8 gamified vs 3 ng after excluding the outlier), so the strongest
attainable two-sided p-value is 2/165 ≈ 0.012. Effect sizes matter more
than stars in this sample.

Outputs (this folder):
  mech_results.json            all headline numbers
  market_rep_metrics.csv       per market-repetition metric panel
  fig_*.png                    report figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

TREATS = ["ng", "gh", "gp", "ghp"]
SHARES_OUTSTANDING = 90  # 3 traders x 20 + 3 traders x 10
FV_MEAN = 64  # mean fundamental value over 15 periods (120..8 step 8)
BADGES = {"bronze": 10, "silver": 15, "gold": 35, "platinum": 50, "diamond": 60}

EXCLUDE_GROUPS = {"20260520_PM/ng1"}  # outlier, see report

results: dict = {}


def r(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def mw(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return None
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return None


def group_test(gdf: pd.DataFrame, col: str) -> dict:
    """Cell means + gamified-vs-ng Mann-Whitney at the group level."""
    cells = {t: r(gdf.loc[gdf.treatment == t, col].mean()) for t in TREATS}
    g1 = gdf.loc[gdf.gamified == 1, col]
    g0 = gdf.loc[gdf.gamified == 0, col]
    hed1 = gdf.loc[gdf.treatment.isin(["gh", "ghp"]), col]
    hed0 = gdf.loc[~gdf.treatment.isin(["gh", "ghp"]), col]
    pn1 = gdf.loc[gdf.treatment.isin(["gp", "ghp"]), col]
    pn0 = gdf.loc[~gdf.treatment.isin(["gp", "ghp"]), col]
    return {
        "cells": cells,
        "gamified_mean": r(g1.mean()),
        "ng_mean": r(g0.mean()),
        "ratio": r(g1.mean() / g0.mean()) if g0.mean() not in (0, None) else None,
        "n": [int(g1.notna().sum()), int(g0.notna().sum())],
        "p_mw": r(mw(g1, g0)),
        "p_hedonic": r(mw(hed1, hed0)),
        "p_pn": r(mw(pn1, pn0)),
    }


# ================================================================ sample map
mkt = pd.read_csv(PROCESSED / "market_day_panel_full.csv")
trd = pd.read_csv(PROCESSED / "trader_day_panel_full.csv")

# DATA-QUALITY FIX: the panel's `gamified` dummy is derived from
# group.market_design == "gamified", but market_design is "hedonic_only" /
# "info_only" for the gh / gp arms, so those markets are miscoded as
# non-gamified. Re-derive all treatment dummies from the treatment label.
for df in (mkt, trd):
    df["gamified"] = (df["treatment"] != "ng").astype(int)
    df["hedonic"] = df["treatment"].isin(["gh", "ghp"]).astype(int)
    df["price_notifications"] = df["treatment"].isin(["gp", "ghp"]).astype(int)
results["data_quality"] = {
    "gamified_dummy_bug": (
        "processed panels code gh/gp as gamified=0 because "
        "group.market_design is 'hedonic_only'/'info_only'; re-derived from "
        "treatment labels here"
    ),
    "cumulative_exports": (
        "custom exports are cumulative oTree dumps; 20260520_AM contains all "
        "20260512 markets — rows assigned to their own session before use"
    ),
}

grp_key = (
    trd.groupby("market_uuid")["participant_code"]
    .apply(lambda s: "|".join(sorted(s.unique())))
    .rename("group_key")
)
mkt = mkt.merge(grp_key, left_on="market_uuid", right_index=True)
trd = trd.merge(grp_key, left_on="market_uuid", right_index=True)

with open(ROOT / "config" / "sessions.yaml") as fh:
    sessions_cfg = yaml.safe_load(fh)["sessions"]
code2id = {c: s["id"] for s in sessions_cfg for c in s["oTree_codes"]}

sess_map = trd.groupby("group_key")["session_code"].first()
groups = (
    mkt.groupby("group_key").agg(treatment=("treatment", "first")).join(sess_map).reset_index()
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
for df in (mkt, trd):
    df["group_label"] = df["group_key"].map(glabel)
    df["session_id"] = df["group_key"].map(groups.set_index("group_key")["session_id"])

# ---------------------------------------------------------------- outlier
out_mkt = mkt[mkt.group_label.isin(EXCLUDE_GROUPS)]
outlier_stats = (
    out_mkt.groupby(["group_label", "repetition"])
    .agg(
        trades=("n_trades_market", "sum"),
        amr=("abs_mispricing_ratio", "mean"),
        peak_avg_price=("avg_trade_price", "max"),
    )
    .reset_index()
)
ng_clean_tot = (
    mkt[(mkt.treatment == "ng") & (~mkt.group_label.isin(EXCLUDE_GROUPS))]
    .groupby("group_label")["n_trades_market"]
    .sum()
)
results["outlier"] = {
    "group": sorted(EXCLUDE_GROUPS),
    "per_rep": outlier_stats.assign(
        amr=lambda d: d.amr.round(2), peak_avg_price=lambda d: d.peak_avg_price.round(1)
    ).to_dict("records"),
    "ng_clean_total_trades": {k: int(v) for k, v in ng_clean_tot.items()},
}

mkt_all = mkt.copy()  # keep for robustness checks
mkt = mkt[~mkt.group_label.isin(EXCLUDE_GROUPS)].copy()
trd = trd[~trd.group_label.isin(EXCLUDE_GROUPS)].copy()

analysis_uuids = set(mkt.market_uuid.unique())
uuid_meta = (
    mkt[["market_uuid", "group_label", "treatment", "repetition", "gamified",
         "hedonic", "price_notifications", "session_id"]]
    .drop_duplicates()
    .set_index("market_uuid")
)

# ================================================== headline H2 vs H1 checks
# market-rep level price metrics
day_mkt = mkt.drop_duplicates(["market_uuid", "trading_day"])
rep_price = (
    day_mkt.groupby("market_uuid")
    .agg(
        trades=("n_trades_market", "sum"),
        avg_mispricing=("avg_mispricing", "mean"),
        abs_mispricing=("avg_abs_mispricing", "mean"),
        amr=("abs_mispricing_ratio", "mean"),
        bubbles=("bubble_period", "sum"),
        surges=("surge", "sum"),
        crashes=("crash", "sum"),
    )
    .join(uuid_meta)
    .reset_index()
)
# RAD / RD (Stockl, Huber, Kirchler 2010): normalize by mean FV, robust to
# the tiny end-of-market FV denominators that inflate AMR.
rad = (
    day_mkt.assign(
        dev=lambda d: (d.avg_trade_price - d.fundamental_value),
    )
    .groupby("market_uuid")
    .agg(RAD=("dev", lambda s: s.abs().mean() / FV_MEAN), RD=("dev", lambda s: s.mean() / FV_MEAN))
)
rep_price = rep_price.merge(rad, on="market_uuid")
rep_price["turnover"] = rep_price.trades / SHARES_OUTSTANDING

# group level (mean over the two repetitions)
gcols = ["trades", "turnover", "avg_mispricing", "abs_mispricing", "amr",
         "RAD", "RD", "bubbles", "surges", "crashes"]
grp_price = (
    rep_price.groupby(["group_label", "treatment", "gamified"])[gcols].mean().reset_index()
)

results["headline"] = {c: group_test(grp_price, c) for c in gcols}

# experience: within-group change rep1 -> rep2
piv = rep_price.pivot_table(index=["group_label", "treatment", "gamified"],
                            columns="repetition", values=["trades", "RAD"])
piv.columns = [f"{a}_r{int(b)}" for a, b in piv.columns]
piv = piv.reset_index()
piv["d_trades"] = piv.trades_r2 - piv.trades_r1
piv["d_RAD"] = piv.RAD_r2 - piv.RAD_r1
results["by_repetition"] = {
    "table": piv.round(3).to_dict("records"),
    "d_RAD_gamified_vs_ng_p": r(mw(piv.loc[piv.gamified == 1, "d_RAD"],
                                   piv.loc[piv.gamified == 0, "d_RAD"])),
}

# ================================================== raw MBO / MBP1 loading
def load_raw_stream(kind: str) -> pd.DataFrame:
    frames = []
    for s in sessions_cfg:
        if not s.get("include", False):
            continue
        d = RAW / s["id"]
        f = d / f"trader_bridge_app_custom_export_{kind}_{s['export_date']}.csv"
        df = pd.read_csv(f)
        df["session_id"] = s["id"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


mbo = load_raw_stream("mbo")
mbp = load_raw_stream("mbp1")
# custom exports are cumulative database dumps: a later export can contain
# earlier sessions' markets (e.g. 20260520_AM contains all of 20260512).
# Keep each market's rows only from the export of its own lab session.
expected_session = uuid_meta["session_id"]
mbo = mbo[mbo.trading_session_uuid.isin(analysis_uuids)].copy()
mbo = mbo[mbo.session_id == mbo.trading_session_uuid.map(expected_session)].copy()
mbp = mbp[mbp.trading_session_uuid.isin(analysis_uuids)].copy()
mbp = mbp[mbp.session_id == mbp.trading_session_uuid.map(expected_session)].copy()
mbo["event_ts"] = pd.to_datetime(mbo["event_ts"], format="ISO8601")
mbp["event_ts"] = pd.to_datetime(mbp["event_ts"], format="ISO8601")

# validation: MBO trade counts must match the panel
chk = (
    mbo[mbo.record_kind == "trade"]
    .groupby("trading_session_uuid")
    .size()
    .rename("mbo_trades")
    .to_frame()
    .join(rep_price.set_index("market_uuid")["trades"])
)
assert (chk.mbo_trades == chk.trades).all(), "MBO trade counts do not match panel"
results["validation"] = {"mbo_vs_panel_trades_match": True,
                         "n_market_reps": int(len(chk))}

# ============================================================ order activity
orders = mbo[mbo.record_kind == "order"]
oact = (
    orders.groupby("trading_session_uuid")["event_type"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={"add": "n_adds", "cancel": "n_cancels", "fill": "n_fills"})
)
oact = oact.join(rep_price.set_index("market_uuid")[["trades"]])
oact["orders_per_trade"] = oact.n_adds / oact.trades
oact["cancel_share"] = oact.n_cancels / oact.n_adds
# marketable share: aggressive orders never rest; a trade consumes one resting
# order, so resting adds = adds - trades that were marketable. Approximate
# marketable share as trades whose aggressor order id never appears as 'add'.
adds_ids = set(orders.loc[orders.event_type == "add", "order_id"])
trades_raw = mbo[mbo.record_kind == "trade"].copy()
fills = orders[orders.event_type == "fill"][["match_id", "order_id"]]
trades_raw = trades_raw.merge(
    fills.groupby("match_id")["order_id"].apply(list).rename("fill_orders"),
    on="match_id", how="left",
)
trades_raw["n_resting_legs"] = trades_raw.fill_orders.apply(
    lambda ids: sum(i in adds_ids for i in ids) if isinstance(ids, list) else np.nan
)

# ==================================================== churn decomposition
tr = trades_raw.rename(
    columns={"trading_session_uuid": "market_uuid",
             "bid_trader_uuid": "buyer_uuid", "ask_trader_uuid": "seller_uuid"}
)[["market_uuid", "trading_day", "event_seq", "event_ts", "aggressor_side",
   "price", "buyer_uuid", "seller_uuid", "match_id"]].copy()
tr = tr.merge(uuid_meta.reset_index(), on="market_uuid")

# trader x day gross/net (in shares; unit size verified)
buys = tr.groupby(["market_uuid", "trading_day", "buyer_uuid"]).size().rename("b")
sells = tr.groupby(["market_uuid", "trading_day", "seller_uuid"]).size().rename("s")
buys.index.names = sells.index.names = ["market_uuid", "trading_day", "trader"]
td = pd.concat([buys, sells], axis=1).fillna(0)
td["gross"] = td.b + td.s
td["net"] = (td.b - td.s).abs()

# decomposition of total sides (= 2 x trades) per market-rep:
#   intraday churn   = sum_(i,d) (gross - |net_d|)          (round-trips within a day)
#   interday churn   = sum_i ( sum_d |net_d| - |sum_d net_d| ) (position flips across days)
#   net reallocation = sum_i |sum_d net_signed|              (rep-level repositioning)
tds = td.reset_index()
tds["net_signed"] = tds.b - tds.s
per_trader = tds.groupby(["market_uuid", "trader"]).agg(
    gross=("gross", "sum"), sum_abs_day_net=("net", "sum"),
    abs_rep_net=("net_signed", lambda s: abs(s.sum())),
)
per_trader["intraday_churn"] = per_trader.gross - per_trader.sum_abs_day_net
per_trader["interday_churn"] = per_trader.sum_abs_day_net - per_trader.abs_rep_net

churn = per_trader.groupby("market_uuid")[
    ["gross", "intraday_churn", "interday_churn", "abs_rep_net"]
].sum()
churn["intraday_share"] = churn.intraday_churn / churn.gross
churn["interday_share"] = churn.interday_churn / churn.gross
churn["net_share"] = churn.abs_rep_net / churn.gross
churn["churn_ratio"] = churn.gross / churn.abs_rep_net  # sides per net share moved

# hot-potato pairs: min(trades A sells to B, B sells to A) per unordered pair
pair = tr.groupby(["market_uuid", "buyer_uuid", "seller_uuid"]).size().rename("n").reset_index()
pair["lo"] = pair[["buyer_uuid", "seller_uuid"]].min(axis=1)
pair["hi"] = pair[["buyer_uuid", "seller_uuid"]].max(axis=1)
pair["dir"] = np.where(pair.buyer_uuid == pair.lo, "lo_buys", "hi_buys")
pp = pair.pivot_table(index=["market_uuid", "lo", "hi"], columns="dir",
                      values="n", fill_value=0)
for c in ("lo_buys", "hi_buys"):
    if c not in pp:
        pp[c] = 0
pp["two_way"] = 2 * pp[["lo_buys", "hi_buys"]].min(axis=1)  # trades in offsetting pairs
hot = pp.groupby("market_uuid").agg(two_way_trades=("two_way", "sum"))
hot = hot.join(rep_price.set_index("market_uuid")["trades"])
hot["hot_potato_share"] = hot.two_way_trades / hot.trades

# concentration: share of trade sides by the two most active traders; HHI
sides = pd.concat([
    tr[["market_uuid", "buyer_uuid"]].rename(columns={"buyer_uuid": "trader"}),
    tr[["market_uuid", "seller_uuid"]].rename(columns={"seller_uuid": "trader"}),
])
part = sides.groupby(["market_uuid", "trader"]).size().rename("n").reset_index()
part["share"] = part.n / part.groupby("market_uuid").n.transform("sum")
conc = part.groupby("market_uuid").agg(
    top2_share=("share", lambda s: s.nlargest(2).sum()),
    hhi=("share", lambda s: (s**2).sum()),
    n_active=("share", "size"),
)

# ==================================================== liquidity from MBP1
mbp = mbp.sort_values(["trading_session_uuid", "event_seq"]).copy()
mbp["rel_spread"] = mbp.spread / mbp.midpoint
mbp["depth"] = mbp.best_bid_sz.fillna(0) + mbp.best_ask_sz.fillna(0)
mbp["two_sided"] = (mbp.best_bid_px.notna() & mbp.best_ask_px.notna()).astype(int)

# time weights: interval to next book update within a market, capped at 60 s
mbp["dur"] = (
    mbp.groupby("trading_session_uuid")["event_ts"].diff().dt.total_seconds().shift(-1)
)
mbp["dur"] = mbp.dur.clip(lower=0, upper=60).fillna(0)


def tw_mean(df, col):
    d = df.dropna(subset=[col])
    if d.dur.sum() == 0:
        return np.nan
    return np.average(d[col], weights=d.dur.clip(lower=1e-6))


liq_rows = []
for muid, d in mbp.groupby("trading_session_uuid"):
    liq_rows.append({
        "market_uuid": muid,
        "tw_quoted_spread": tw_mean(d, "spread"),
        "tw_rel_spread": tw_mean(d, "rel_spread"),
        "med_rel_spread": d.rel_spread.median(),
        "tw_depth": tw_mean(d, "depth"),
        "tw_two_sided": tw_mean(d, "two_sided"),
    })
liq = pd.DataFrame(liq_rows).set_index("market_uuid")

# ---------------- trade-based measures: pre-trade quotes, effective spread,
# post-trade impact. Pre-trade book = last MBP1 row with source seq strictly
# below the trade's event_seq; post-trade = row sourced from the trade itself.
eff_frames = []
for muid, d in mbp.groupby("trading_session_uuid"):
    t = tr[tr.market_uuid == muid].sort_values("event_seq").copy()
    if t.empty:
        continue
    q = d.sort_values("source_mbo_event_seq")[
        ["source_mbo_event_seq", "midpoint", "spread", "rel_spread",
         "best_bid_sz", "best_ask_sz"]
    ]
    pre = pd.merge_asof(
        t[["event_seq", "market_uuid", "trading_day", "price", "aggressor_side"]],
        q, left_on="event_seq", right_on="source_mbo_event_seq",
        direction="backward", allow_exact_matches=False,
    )
    post = d[d.source_order_event_type == "trade"][
        ["source_mbo_event_seq", "midpoint"]
    ].rename(columns={"source_mbo_event_seq": "event_seq", "midpoint": "mid_post"})
    pre = pre.merge(post, on="event_seq", how="left")
    eff_frames.append(pre)

eff = pd.concat(eff_frames, ignore_index=True)
eff["q"] = np.where(eff.aggressor_side == "B", 1, -1)
eff["eff_rel_halfspread"] = eff.q * (eff.price - eff.midpoint) / eff.midpoint
eff["impact_rel"] = eff.q * (eff.mid_post - eff.midpoint) / eff.midpoint
eff["pre_depth"] = eff.best_bid_sz.fillna(0) + eff.best_ask_sz.fillna(0)

eff_mkt = eff.groupby("market_uuid").agg(
    eff_spread_med=("eff_rel_halfspread", "median"),
    eff_spread_mean=("eff_rel_halfspread", "mean"),
    pretrade_rel_spread_med=("rel_spread", "median"),
    pretrade_depth_mean=("pre_depth", "mean"),
    impact_med=("impact_rel", "median"),
    impact_mean=("impact_rel", "mean"),
    share_buy_aggr=("q", lambda s: (s > 0).mean()),
    n_priced=("eff_rel_halfspread", "count"),
)

# daily order-flow imbalance: |#buy aggr - #sell aggr| / trades
ofi = (
    tr.assign(q=np.where(tr.aggressor_side == "B", 1, -1))
    .groupby(["market_uuid", "trading_day"])
    .agg(imb=("q", "sum"), n=("q", "size"))
)
ofi["abs_imb_share"] = ofi.imb.abs() / ofi.n
ofi_mkt = ofi.groupby("market_uuid")["abs_imb_share"].mean().rename("flow_imbalance")

# =========================================== assemble market-rep metric panel
mrep = (
    rep_price.set_index("market_uuid")
    .join(oact[["n_adds", "n_cancels", "orders_per_trade", "cancel_share"]])
    .join(churn[["intraday_share", "interday_share", "net_share",
                 "churn_ratio", "abs_rep_net"]])
    .join(hot[["hot_potato_share"]])
    .join(conc)
    .join(liq)
    .join(eff_mkt)
    .join(ofi_mkt)
    .reset_index()
)
mrep.to_csv(HERE / "market_rep_metrics.csv", index=False)

MECH_COLS = [
    "n_adds", "orders_per_trade", "cancel_share",
    "intraday_share", "interday_share", "net_share", "churn_ratio", "abs_rep_net",
    "hot_potato_share", "top2_share", "hhi",
    "tw_quoted_spread", "tw_rel_spread", "med_rel_spread", "tw_depth", "tw_two_sided",
    "eff_spread_med", "eff_spread_mean", "pretrade_rel_spread_med",
    "pretrade_depth_mean", "impact_med", "impact_mean", "flow_imbalance",
]
grp_mech = (
    mrep.groupby(["group_label", "treatment", "gamified"])[MECH_COLS].mean().reset_index()
)
results["mechanisms"] = {c: group_test(grp_mech, c) for c in MECH_COLS}

# volume x price impact cross-check: total repricing volume
mrep["net_shares_moved"] = mrep.abs_rep_net / 2  # sides -> shares
results["volume_decomposition_shares"] = (
    mrep.groupby("treatment")
    .apply(lambda d: pd.Series({
        "trades": d.trades.mean(),
        "intraday_rt_trades": (d.intraday_share * d.trades).mean(),
        "interday_rt_trades": (d.interday_share * d.trades).mean(),
        "net_trades": (d.net_share * d.trades).mean(),
    }), include_groups=False)
    .round(1)
    .to_dict("index")
)

# ============================================= per-trader volume & badges
trader_rep = tds.groupby(["market_uuid", "trader"]).agg(trades_sides=("gross", "sum"))
trader_rep["n_trades"] = trader_rep.trades_sides  # sides for that trader = trades participated
trader_rep = trader_rep.join(uuid_meta[["treatment", "gamified", "group_label"]])
results["per_trader_volume"] = {
    t: {
        "mean": r(trader_rep.loc[trader_rep.treatment == t, "n_trades"].mean(), 1),
        "median": r(trader_rep.loc[trader_rep.treatment == t, "n_trades"].median(), 1),
        "p90": r(trader_rep.loc[trader_rep.treatment == t, "n_trades"].quantile(0.9), 1),
        "max": r(trader_rep.loc[trader_rep.treatment == t, "n_trades"].max(), 1),
    }
    for t in TREATS
}

# ==================================================================== figures
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})
TCOLORS = {"ng": "#444444", "gh": "#1f77b4", "gp": "#2ca02c", "ghp": "#d62728"}
TLAB = {"ng": "Non-gamified", "gh": "Hedonic (gh)", "gp": "Price notif. (gp)",
        "ghp": "Both (ghp)"}

# --- Fig 1: price paths vs FV, clean sample, by repetition
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for rep, ax in zip((1, 2), axes):
    sub = day_mkt[day_mkt.repetition == rep]
    for t in TREATS:
        p = sub[sub.treatment == t].groupby("trading_day")["avg_trade_price"].mean()
        ax.plot(p.index, p.values, label=TLAB[t], color=TCOLORS[t], lw=1.6)
    fv = 8 * (16 - np.arange(1, 16))
    ax.plot(range(1, 16), fv, "k--", lw=1, label="Fundamental value")
    ax.set_title(f"Repetition {rep}")
    ax.set_xlabel("Trading day")
axes[0].set_ylabel("Avg. transaction price (E$)")
axes[0].legend(fontsize=7)
fig.suptitle("Price paths by treatment (outlier excluded)", y=1.02)
fig.tight_layout()
fig.savefig(HERE / "fig_price_paths.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 2: volume by treatment (per market-rep) + outlier marker
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
order = ["ng", "gh", "ghp", "gp"]
for i, t in enumerate(order):
    y = rep_price.loc[rep_price.treatment == t, "trades"]
    ax.scatter(np.full(len(y), i) + np.random.default_rng(1).uniform(-0.08, 0.08, len(y)),
               y, color=TCOLORS[t], s=28, zorder=3)
    ax.hlines(y.mean(), i - 0.22, i + 0.22, color=TCOLORS[t], lw=2.5)
out_tr = out_mkt.drop_duplicates(["market_uuid", "trading_day"]).groupby("market_uuid")[
    "n_trades_market"].sum()
ax.scatter([0, 0], out_tr.values, marker="x", color="crimson", s=45,
           label="excluded ng outlier")
ax.set_xticks(range(len(order)), [TLAB[t] for t in order], fontsize=7)
ax.set_ylabel("Trades per market repetition")
ax.legend(fontsize=7)
ax.set_title("Volume by treatment")

ax = axes[1]
for t in TREATS:
    v = (tr[tr.treatment == t].groupby(["market_uuid", "trading_day"]).size()
         .rename("n").reset_index()
         .groupby("trading_day")["n"].mean())
    ax.plot(v.index, v.values, color=TCOLORS[t], lw=1.6, label=TLAB[t])
ax.set_xlabel("Trading day")
ax.set_ylabel("Trades per day (mean)")
ax.set_title("Volume across the market lifetime")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "fig_volume.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 3: churn decomposition stacked bars (shares of volume)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
dec = mrep.groupby("treatment")[["intraday_share", "interday_share", "net_share"]].mean()
dec = dec.loc[order]
bottom = np.zeros(len(dec))
for comp, lab, col in [
    ("intraday_share", "Intraday round-trips", "#d62728"),
    ("interday_share", "Across-day flips", "#ff9896"),
    ("net_share", "Net repositioning (rep-level)", "#aec7e8"),
]:
    ax.bar(range(len(dec)), dec[comp], bottom=bottom, label=lab, color=col,
           edgecolor="white")
    bottom += dec[comp].values
ax.set_xticks(range(len(dec)), [TLAB[t] for t in dec.index], fontsize=7)
ax.set_ylabel("Share of trading volume")
ax.set_title("Where the volume goes: churn decomposition")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(HERE / "fig_churn_decomposition.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 4: liquidity (pre-trade relative spread and effective spread)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
panels = [
    ("pretrade_rel_spread_med", "Median pre-trade rel. quoted spread"),
    ("eff_spread_med", "Median rel. effective half-spread"),
    ("tw_depth", "Time-weighted depth at best (shares)"),
]
for ax, (col, title) in zip(axes, panels):
    for i, t in enumerate(order):
        y = mrep.loc[mrep.treatment == t, col]
        ax.scatter(np.full(len(y), i) + np.random.default_rng(2).uniform(-0.07, 0.07, len(y)),
                   y, color=TCOLORS[t], s=26, zorder=3)
        ax.hlines(y.mean(), i - 0.22, i + 0.22, color=TCOLORS[t], lw=2.5)
    ax.set_xticks(range(len(order)), order, fontsize=8)
    ax.set_title(title, fontsize=8)
fig.suptitle("Liquidity by treatment (market-repetition level)", y=1.03)
fig.tight_layout()
fig.savefig(HERE / "fig_liquidity.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 5: per-trader trade counts vs badge thresholds
fig, ax = plt.subplots(figsize=(6.2, 3.4))
bins = np.arange(0, 130, 5)
for t in TREATS:
    x = trader_rep.loc[trader_rep.treatment == t, "n_trades"]
    ax.hist(x, bins=bins, histtype="step", lw=1.6, density=True,
            color=TCOLORS[t], label=f"{TLAB[t]} (median {x.median():.0f})")
for b, thr in BADGES.items():
    ax.axvline(thr, color="gray", ls=":", lw=0.8)
    ax.text(thr, ax.get_ylim()[1] * 0.97, b[0].upper(), ha="center", fontsize=6,
            color="gray")
ax.set_xlabel("Trades per trader per market repetition")
ax.set_ylabel("Density")
ax.set_title("Individual trading intensity vs badge thresholds (B/S/G/P/D)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "fig_trader_intensity.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 6: outlier illustration
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
for g, d in mkt_all[mkt_all.treatment == "ng"].drop_duplicates(
        ["market_uuid", "trading_day"]).groupby("group_label"):
    for rep, dd in d.groupby("repetition"):
        style = dict(color="crimson", lw=1.8) if g in EXCLUDE_GROUPS else dict(
            color="#777777", lw=1.1)
        ax.plot(dd.trading_day, dd.avg_trade_price, **style)
ax.plot(range(1, 16), 8 * (16 - np.arange(1, 16)), "k--", lw=1)
ax.set_title("ng price paths (red = 20260520_PM/ng1)")
ax.set_xlabel("Trading day")
ax.set_ylabel("Avg. price (E$)")
ax = axes[1]
tot = (mkt_all[mkt_all.treatment == "ng"]
       .drop_duplicates(["market_uuid", "trading_day"])
       .groupby(["group_label", "repetition"])["n_trades_market"].sum().reset_index())
labs = tot.group_label + " r" + tot.repetition.astype(str)
cols = ["crimson" if g in EXCLUDE_GROUPS else "#777777" for g in tot.group_label]
ax.barh(labs, tot.n_trades_market, color=cols)
ax.set_xlabel("Trades per market repetition")
ax.set_title("ng volume: the outlier trades ~7x the median")
fig.tight_layout()
fig.savefig(HERE / "fig_outlier.png", bbox_inches="tight")
plt.close(fig)

# ==================================================================== save
with open(HERE / "mech_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)

print(json.dumps(results["headline"], indent=1, default=str))
print(json.dumps(results["mechanisms"], indent=1, default=str))
print(json.dumps(results["volume_decomposition_shares"], indent=1, default=str))
print("saved to", HERE)
