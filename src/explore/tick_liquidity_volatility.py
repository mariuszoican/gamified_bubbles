"""
Tick-level liquidity and volatility by treatment (ghp vs ng focus).

From raw MBO/MBP1 exports, per market-day:
  - time-weighted quoted spread and relative spread (both sides quoted)
  - time-weighted depth at best (bid+ask size)
  - share of time with a two-sided book
  - relative effective spread per trade: 2|p - prevailing mid| / mid
  - realized volatility: sqrt(sum of squared log midquote returns)
  - trader-day directionality |B-S|/(B+S) from the trader panel
    (1 = one-way flow, 0 = intraday round-tripping / churn)

Aggregated to market-rep means, then Mann-Whitney ghp vs ng at the
market-rep and group level. Outlier group 20260520_PM/ng1 excluded.
Writes tick_liq_vol.csv (market-day panel) next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = Path(__file__).resolve().parent / "tick_liq_vol.csv"
OUT_JSON = Path(__file__).resolve().parent / "tick_liq_vol.json"

mkt = pd.read_csv(ROOT / "data/processed/market_day_panel_full.csv")
trd = pd.read_csv(ROOT / "data/processed/trader_day_panel_full.csv")
cfg = yaml.safe_load(open(ROOT / "config/sessions.yaml"))["sessions"]
code2id = {c: s["id"] for s in cfg for c in s["oTree_codes"]}

gk = trd.groupby("market_uuid")["participant_code"].apply(
    lambda s: "|".join(sorted(s.unique()))
)
mkt["gkey"] = mkt.market_uuid.map(gk)
mkt["sess"] = mkt.market_uuid.map(
    trd.groupby("market_uuid")["session_code"].first()
).map(code2id)
out_keys = mkt.loc[(mkt.sess == "20260520_PM") & (mkt.treatment == "ng"), "gkey"].unique()
mkt = mkt[~mkt.gkey.isin(out_keys)]
keep = set(mkt.market_uuid)
meta = mkt.drop_duplicates("market_uuid").set_index("market_uuid")[
    ["treatment", "repetition", "gkey", "sess"]
]

rows = []
for s in cfg:
    if not s.get("include", False):
        continue
    raw = ROOT / "data/raw" / s["id"]
    mbo = pd.read_csv(raw / f"trader_bridge_app_custom_export_mbo_{s['export_date']}.csv")
    mbp = pd.read_csv(raw / f"trader_bridge_app_custom_export_mbp1_{s['export_date']}.csv")
    for df in (mbo, mbp):
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
    mbo = mbo[mbo.trading_session_uuid.isin(keep)].copy()
    mbp = mbp[mbp.trading_session_uuid.isin(keep)].copy()
    mbp["ts"] = pd.to_datetime(mbp.event_ts, format="ISO8601")

    trades = mbo[mbo.record_kind == "trade"][
        ["trading_session_uuid", "trading_day", "event_seq", "price"]
    ].sort_values("event_seq")

    for (mu, day), q in mbp.groupby(["trading_session_uuid", "trading_day"]):
        q = q.sort_values("event_seq").copy()
        two = q.best_bid_px.notna() & q.best_ask_px.notna()
        dur = (q.ts.shift(-1) - q.ts).dt.total_seconds().clip(lower=0)
        dur.iloc[-1] = 0.0
        day_len = dur.sum()

        w = dur[two]
        qs = rqs = depth = np.nan
        if day_len > 0 and w.sum() > 0:
            qs = np.average(q.spread[two], weights=w)
            rqs = np.average((q.spread / q.midpoint)[two], weights=w)
            depth = np.average((q.best_bid_sz + q.best_ask_sz)[two], weights=w)
        pct_two = w.sum() / day_len if day_len > 0 else np.nan

        # realized volatility from midquote updates (two-sided book only)
        mid = q.midpoint[two]
        r = np.log(mid).diff().dropna()
        rv = np.sqrt((r**2).sum()) if len(r) else np.nan

        # effective spread: prevailing mid strictly before each trade
        t = trades[(trades.trading_session_uuid == mu) & (trades.trading_day == day)]
        eff = np.nan
        if len(t):
            qq = q[two][["event_seq", "midpoint"]].rename(
                columns={"event_seq": "src_seq"}
            )
            m = pd.merge_asof(
                t.sort_values("event_seq"),
                qq.sort_values("src_seq"),
                left_on="event_seq",
                right_on="src_seq",
                direction="backward",
                allow_exact_matches=False,
            ).dropna(subset=["midpoint"])
            if len(m):
                eff = (2 * (m.price - m.midpoint).abs() / m.midpoint).mean()

        rows.append(
            dict(market_uuid=mu, trading_day=day, quoted_spread=qs,
                 rel_quoted_spread=rqs, depth_best=depth, pct_two_sided=pct_two,
                 rel_eff_spread=eff, rv_mid=rv, n_trades=len(t))
        )

# --- intraday churn from the trader panel: |buys-sells|/(buys+sells) per
# trader-day, averaged to market-day (1 = one-way flow, 0 = round-tripping)
trd_k = trd[trd.market_uuid.isin(keep)].copy()
active = trd_k[(trd_k.n_buys + trd_k.n_sells) > 0].copy()
active["net_gross"] = (active.n_buys - active.n_sells).abs() / (
    active.n_buys + active.n_sells
)
churn = (
    active.groupby(["market_uuid", "trading_day"])["net_gross"]
    .mean()
    .rename("trader_day_directionality")
    .reset_index()
)

panel = pd.DataFrame(rows).join(meta, on="market_uuid")
panel = panel.merge(churn, on=["market_uuid", "trading_day"], how="left")
panel.to_csv(OUT_CSV, index=False)

# --- aggregate to market-rep and test ghp vs ng
metrics = ["rel_quoted_spread", "quoted_spread", "depth_best", "pct_two_sided",
           "rel_eff_spread", "rv_mid", "trader_day_directionality"]
mrep = panel.groupby(["market_uuid", "treatment", "gkey"])[metrics].mean().reset_index()
grp = mrep.groupby(["gkey", "treatment"])[metrics].mean().reset_index()

res = {}
for v in metrics:
    a = mrep.loc[mrep.treatment == "ghp", v].dropna()
    b = mrep.loc[mrep.treatment == "ng", v].dropna()
    ga = grp.loc[grp.treatment == "ghp", v].dropna()
    gb = grp.loc[grp.treatment == "ng", v].dropna()
    res[v] = {
        "ghp_mean": round(a.mean(), 4), "ng_mean": round(b.mean(), 4),
        "p_rep_two": round(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue, 4),
        "p_grp_two": round(stats.mannwhitneyu(ga, gb, alternative="two-sided").pvalue, 4),
        "n_rep": [len(a), len(b)], "n_grp": [len(ga), len(gb)],
        "all_treatments": {
            t: round(mrep.loc[mrep.treatment == t, v].mean(), 4)
            for t in ["ng", "gh", "gp", "ghp"]
        },
    }

with open(OUT_JSON, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
print("saved", OUT_CSV.name, OUT_JSON.name)
