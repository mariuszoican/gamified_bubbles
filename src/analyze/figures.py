"""
Publication-grade figures for the Gamified Bubbles paper.

Reads ONLY the processed panels in data/processed/ and writes figures
(PDF + 300-dpi PNG) to output/figures/.

Sample: GHP (full gamification) vs NG (control) market-reps. Outliers
excluded throughout: 20260520_PM/ng1 (two-trader churn/peg, ~3x volume)
and 20280904/ghp1 (price opens below 10 and stays there in both reps).

Figures
  1. mispricing_daypath   – absolute mispricing, AMR, RAD by trading day
  2. liquidity_spreads    – relative effective and realized spreads
  3. volume_orderflow     – volume, order-flow imbalance, limit orders, churn
  4. literacy_gini_payoff – Gini day path; payoff by financial literacy
  9. price_paths          – mean trade price by day in GHP vs NG, plus v_t
 10. carry_daypath        – realized price drop vs expected dividend (carry)

Confidence intervals are 95% HC1 bands (mean ± 1.96·s/√n), the same
White SE as the tables. Day-path figures use market-days on that day;
bar figures use market-days (not first-collapsed market-rep means),
except fig 8, whose outcomes are market-rep counts. Trader panels use
trader-markets.

Usage:  python src/analyze/figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "output" / "figures"

TREATMENTS = ["ng", "ghp"]
LABELS = {"ng": "Control (NG)", "ghp": "Gamified (GHP)"}
COLORS = {"ng": "#4d4d4d", "ghp": "#0072b2"}
EXCLUDE_GROUPS = {
    "20260520_PM/ng1",  # wash/churn peg at ~130
    "20280904/ghp1",  # collapsed: P stays below 10 both reps
}
Z95 = 1.96

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_market_panel() -> pd.DataFrame:
    mkt = pd.read_csv(PROCESSED / "market_day_panel_full.csv")
    return mkt[
        mkt["treatment"].isin(TREATMENTS)
        & ~mkt["group_label"].isin(EXCLUDE_GROUPS)
    ].copy()


def load_trader_panel() -> pd.DataFrame:
    trd = pd.read_csv(PROCESSED / "trader_day_panel_full.csv")
    return trd[
        trd["treatment"].isin(TREATMENTS)
        & ~trd["group_label"].isin(EXCLUDE_GROUPS)
    ].copy()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def mean_ci(vals, scale: float = 1.0) -> tuple[float, float]:
    """Sample mean and HC1 95% half-width (s/√n). Same VCOV as tables."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)] * scale
    n = v.size
    if n == 0:
        return np.nan, np.nan
    if n < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(Z95 * v.std(ddof=1) / np.sqrt(n))


def day_path(mkt: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean and HC1 95% CI of `col` per treatment × trading day
    (nan-aware: days without trades are skipped)."""
    g = mkt.groupby(["treatment", "trading_day"])[col]
    out = g.agg(mean="mean", sd="std", n="count").reset_index()
    out["ci"] = Z95 * out["sd"] / np.sqrt(out["n"])
    return out


def draw_daypath(ax, path: pd.DataFrame) -> None:
    for t in TREATMENTS:
        sub = path[path["treatment"] == t]
        ax.plot(
            sub["trading_day"], sub["mean"],
            color=COLORS[t], lw=1.6, marker="o", ms=3.5, label=LABELS[t],
        )
        ax.fill_between(
            sub["trading_day"],
            sub["mean"] - sub["ci"],
            sub["mean"] + sub["ci"],
            color=COLORS[t], alpha=0.15, lw=0,
        )
    ax.set_xticks(range(1, 16, 2))
    ax.set_xlim(0.6, 15.4)
    ax.set_xlabel("Trading day")


def draw_bars(ax, df: pd.DataFrame, col: str, scale: float = 1.0) -> None:
    """Treatment-mean bars with HC1 95% CIs on the rows of `df`."""
    for i, t in enumerate(TREATMENTS):
        m, ci = mean_ci(df.loc[df["treatment"] == t, col], scale=scale)
        ax.bar(i, m, width=0.55, color=COLORS[t], alpha=0.85, zorder=2)
        ax.errorbar(
            i, m, yerr=ci, fmt="none", ecolor="black",
            elinewidth=1.1, capsize=4, capthick=1.1, zorder=4,
        )
    ax.set_xticks(range(len(TREATMENTS)))
    ax.set_xticklabels([LABELS[t] for t in TREATMENTS])
    ax.margins(x=0.18)


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"saved {FIG_DIR / name}.pdf / .png")


# ----------------------------------------------------------------------
# Figure 9: price paths vs fundamental value
# ----------------------------------------------------------------------

def fig_price_paths(mkt: pd.DataFrame) -> None:
    """Mean transaction price by day (HC1 95% CI) and the
    deterministic declining fundamental v_t = 8 × remaining dividends."""
    fig, ax = plt.subplots(figsize=(8.8, 4.95))  # ~16:9

    draw_daypath(ax, day_path(mkt, "avg_trade_price"))
    fv = (
        mkt.groupby("trading_day", as_index=False)["fundamental_value"]
        .first()
        .sort_values("trading_day")
    )
    ax.plot(
        fv["trading_day"], fv["fundamental_value"],
        color="black", ls="--", lw=1.5, label="Fundamental value", zorder=3,
    )
    ax.set_ylabel("Average trade price (exp. currency)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig9_price_paths")


# ----------------------------------------------------------------------
# Figure 10: overnight carry — price drop vs expected dividend
# ----------------------------------------------------------------------

def fig_carry(mkt: pd.DataFrame) -> None:
    """Mean realized overnight price drop −ΔP_{t→t+1} vs E[D] = 8.

    Holding a share overnight pays 8 + ΔP. The dashed line is both the
    expected dividend and |Δv_t|. When the treatment path sits below 8,
    riding is myopically profitable; when it rises above, carrying loses
    money. Day 15 has no next price and is omitted.
    """
    df = mkt.sort_values(["market_uuid", "trading_day"]).copy()
    df["price_drop"] = -(
        df.groupby("market_uuid")["avg_trade_price"].shift(-1)
        - df["avg_trade_price"]
    )
    df = df[df["trading_day"] < 15]

    fig, ax = plt.subplots(figsize=(8.8, 4.95))
    draw_daypath(ax, day_path(df, "price_drop"))
    ax.axhline(
        8.0, color="black", ls="--", lw=1.5,
        label=r"Expected dividend $E[D]=8$ $(=|\Delta v_t|)$",
    )
    ax.set_ylabel("Overnight price drop (exp. currency)")
    ax.set_xticks(range(1, 15, 2))
    ax.set_xlim(0.6, 14.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig10_carry_daypath")


# ----------------------------------------------------------------------
# Figure 1: mispricing by trading day
# ----------------------------------------------------------------------

def fig_mispricing(mkt: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))

    specs = [
        (
            "avg_abs_mispricing",
            "A. Absolute mispricing",
            r"mean$_n$ $|P_n - v_t|$ (exp. currency)",
        ),
        (
            "abs_mispricing_ratio",
            "B. Relative mispricing (AMR)",
            r"mean$_n$ $|P_n - v_t|\,/\,v_t$",
        ),
        (
            "rad",
            "C. Relative absolute deviation (RAD)",
            r"mean$_n$ $|P_n - v_t|\,/\,\bar{v}$,  $\bar{v}=64$",
        ),
    ]
    for ax, (col, title, ylab) in zip(axes, specs):
        draw_daypath(ax, day_path(mkt, col))
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig1_mispricing_daypath")


# ----------------------------------------------------------------------
# Figure 2: liquidity — effective and realized spreads
# ----------------------------------------------------------------------

def fig_liquidity(mkt: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0))

    pct = "Percent of prevailing midpoint"
    specs = [
        ("rel_quoted_spread", "A. Relative quoted spread", pct),
        ("rel_eff_spread", "B. Relative effective spread", pct),
        ("rel_realized_spread", "C. Relative realized spread", pct),
        ("rel_price_impact", "D. Relative price impact", pct),
        ("depth_best", "E. Depth at best quotes", "Shares (bid + ask), time-weighted"),
        ("rv_mid", "F. Midquote volatility", "Realized volatility per market-day"),
    ]
    for ax, (col, title, ylab) in zip(axes.ravel(), specs):
        scale = 100.0 if ylab == pct else 1.0
        draw_bars(ax, mkt, col, scale=scale)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        if scale == 100.0:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    fig.tight_layout(w_pad=2.5, h_pad=2.4)
    save(fig, "fig2_liquidity_spreads")


# ----------------------------------------------------------------------
# Figure 3: volume, order flow, and trading strategies
# ----------------------------------------------------------------------

def fig_volume_orderflow(mkt: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # A. trading volume by day
    draw_daypath(ax_a, day_path(mkt, "n_trades_market"))
    ax_a.set_title("A. Trading volume", loc="left")
    ax_a.set_ylabel("Trades per market-day")
    ax_a.set_ylim(bottom=0)
    ax_a.legend(loc="upper right")

    # B. order-flow imbalance
    draw_bars(ax_b, mkt, "order_flow_imbalance")
    ax_b.set_title("B. Order-flow imbalance", loc="left")
    ax_b.set_ylabel(r"$(V^{buy} - V^{sell})\,/\,(V^{buy} + V^{sell})$")
    ax_b.axhline(0, color="0.6", lw=0.8, zorder=1)

    # C. limit-order activity: submissions and cancellations per market-day
    # (left axis) and the share of passive limit orders among all order
    # submissions (right axis)
    ax_c2 = ax_c.twinx()
    width = 0.38
    x = np.arange(3)
    for i, t in enumerate(TREATMENTS):
        sub = mkt[mkt["treatment"] == t]
        for ax, cols, scale in (
            (ax_c, ["n_limit_orders", "n_cancels"], 1.0),
            (ax_c2, ["share_limit_orders"], 100.0),
        ):
            pos = x[:2] if ax is ax_c else x[2:]
            means, cis = [], []
            for c in cols:
                m, ci = mean_ci(sub[c], scale=scale)
                means.append(m)
                cis.append(ci)
            ax.bar(
                pos + (i - 0.5) * width, means, width=width * 0.92,
                color=COLORS[t], alpha=0.85, zorder=2,
                label=LABELS[t] if ax is ax_c else None,
            )
            ax.errorbar(
                pos + (i - 0.5) * width, means, yerr=cis, fmt="none",
                ecolor="black", elinewidth=1.1, capsize=4, capthick=1.1,
                zorder=3,
            )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        ["Limit orders\nsubmitted", "Cancellations", "Share of limit\norders (right)"]
    )
    ax_c.set_title("C. Limit-order activity", loc="left")
    ax_c.set_ylabel("Orders per market-day")
    ax_c.set_ylim(0, ax_c.get_ylim()[1] * 1.3)
    ax_c2.set_ylabel("Limit orders, % of orders submitted")
    ax_c2.set_ylim(0, 100)
    ax_c2.spines["right"].set_visible(True)
    ax_c.legend(loc="upper left")

    # D. intraday churn
    draw_bars(ax_d, mkt, "churn")
    ax_d.set_title("D. Intraday churn", loc="left")
    ax_d.set_ylabel(r"$1 - |B - S|\,/\,(B + S)$ per trader-day")

    fig.tight_layout(w_pad=2.2, h_pad=2.4)
    save(fig, "fig3_volume_orderflow")


# ----------------------------------------------------------------------
# Figure 5: the liquidity-provision mechanism
# ----------------------------------------------------------------------

def fig_liquidity_provision(mkt: pd.DataFrame) -> None:
    """Why spreads tighten despite heavier order-flow consumption: gamified
    traders undercut the standing book more often and replenish quotes
    faster after trades."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    draw_bars(ax_a, mkt, "n_improving_adds")
    ax_a.set_title("A. Spread-improving limit orders", loc="left")
    ax_a.set_ylabel("Submissions tightening the quoted\nspread, per market-day")

    draw_bars(ax_b, mkt, "share_improving_adds", scale=100.0)
    ax_b.set_title("B. Share of submissions improving the spread", loc="left")
    ax_b.set_ylabel("Percent of limit-order submissions")

    draw_bars(ax_c, mkt, "time_to_same_side_order_s")
    ax_c.set_title("C. Order replenishment after a trade", loc="left")
    ax_c.set_ylabel("Median seconds from trade to next\nlimit order on the consumed side")

    draw_bars(ax_d, mkt, "spread_recovery_s")
    ax_d.set_title("D. Spread recovery after a trade", loc="left")
    ax_d.set_ylabel("Median seconds until spread returns\nto its pre-trade level")

    fig.tight_layout(w_pad=2.2, h_pad=2.4)
    save(fig, "fig5_liquidity_provision")


# ----------------------------------------------------------------------
# Figure 4: inequality and payoffs by financial literacy
# ----------------------------------------------------------------------

def fig_literacy(mkt: pd.DataFrame, trd: pd.DataFrame) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # A. wealth inequality (Gini) by trading day, from trade-stream
    # reconstructed wealth (the oTree num_shares snapshot is unreliable)
    draw_daypath(ax_a, day_path(mkt, "gini"))
    ax_a.set_title("A. Wealth inequality", loc="left")
    ax_a.set_ylabel("Gini coefficient of trader wealth")
    ax_a.set_ylim(bottom=0)
    ax_a.legend(loc="upper left")

    # B. within-market relative wealth by financial literacy (sample-wide
    # median split from the panel). Payoff = rel_wealth at day 15 (the
    # trade-stream reconstructed economic payoff; oTree's trade_payoff
    # includes dividends overpaid on phantom shares), averaged across a
    # participant's two reps.
    traders = (
        trd[trd["trading_day"] == 15]
        .groupby("participant_code")
        .agg(
            treatment=("treatment", "first"),
            above=("above_median_literacy", "first"),
            payoff_recon=("rel_wealth", "mean"),
        )
        .reset_index()
        .dropna()
    )
    traders["literacy"] = np.where(traders["above"] > 0, "above", "below")

    width, x = 0.38, np.arange(2)  # x: below, above median
    for i, t in enumerate(TREATMENTS):
        means, cis = [], []
        for lit in ("below", "above"):
            v = traders.loc[
                (traders["treatment"] == t) & (traders["literacy"] == lit),
                "payoff_recon",
            ]
            m, ci = mean_ci(v)
            means.append(m)
            cis.append(ci)
        ax_b.bar(
            x + (i - 0.5) * width, means, width=width * 0.92,
            color=COLORS[t], alpha=0.85, label=LABELS[t], zorder=2,
        )
        ax_b.errorbar(
            x + (i - 0.5) * width, means, yerr=cis, fmt="none",
            ecolor="black", elinewidth=1.1, capsize=4, capthick=1.1, zorder=3,
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(
        ["Below-median literacy", "Above-median literacy"]
    )
    ax_b.set_title("B. Relative wealth by financial literacy", loc="left")
    ax_b.set_ylabel("Final wealth relative to market mean (exp. currency)")
    ax_b.axhline(0, color="0.6", lw=0.8, zorder=1)
    lo, hi = ax_b.get_ylim()
    ax_b.set_ylim(lo * 1.1 if lo < 0 else lo, hi * 1.25)
    ax_b.legend(loc="upper left", ncols=2)

    fig.tight_layout(w_pad=2.5)
    save(fig, "fig4_literacy_gini_payoff")


# ----------------------------------------------------------------------
# Figure 6: trader types — shares and payoffs
# ----------------------------------------------------------------------

TYPE_ORDER = ["feedback", "speculator", "fundamental", "market_maker", "other"]
TYPE_LABELS = {
    "feedback": "Feedback",
    "speculator": "Speculator",
    "fundamental": "Fundamentalist",
    "other": "Unclassified",
    "market_maker": "Market maker",
}


def _trader_types(trd: pd.DataFrame) -> pd.DataFrame:
    """One row per trader-market: mutually exclusive `trader_type` (from the
    panel), gross trading volume, and final relative wealth (`rel_wealth`
    at day 15)."""
    gross = (
        (trd["n_buys"] + trd["n_sells"])
        .groupby([trd["market_uuid"], trd["participant_code"]])
        .sum()
        .rename("gross")
    )
    tm = trd[trd["trading_day"] == 15][
        [
            "market_uuid", "participant_code", "treatment", "trader_type",
            "rel_wealth",
        ]
    ].copy()
    tm = tm.rename(columns={"trader_type": "type"})
    return tm.join(gross, on=["market_uuid", "participant_code"])


def fig_trader_types(trd: pd.DataFrame) -> None:
    tm = _trader_types(trd)
    fig, (ax_a, ax_v, ax_b) = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # A. share of trader-markets by (mutually exclusive) type and treatment.
    # V. share of gross trading volume by type.
    width = 0.38
    x = np.arange(len(TYPE_ORDER))
    for i, t in enumerate(TREATMENTS):
        sub = tm[tm["treatment"] == t]
        shares = [(sub["type"] == ty).mean() for ty in TYPE_ORDER]
        vol_shares = [
            sub.loc[sub["type"] == ty, "gross"].sum() / sub["gross"].sum()
            for ty in TYPE_ORDER
        ]
        for ax, vals in ((ax_a, shares), (ax_v, vol_shares)):
            ax.bar(
                x + (i - 0.5) * width, np.array(vals) * 100,
                width=width * 0.92, color=COLORS[t], alpha=0.85,
                label=LABELS[t] if ax is ax_a else None, zorder=2,
            )
    for ax in (ax_a, ax_v):
        ax.set_xticks(x)
        ax.set_xticklabels(
            [TYPE_LABELS[ty] for ty in TYPE_ORDER], rotation=30, ha="right"
        )
    ax_a.set_title("A. Share of traders by type", loc="left")
    ax_a.set_ylabel("Percent of trader-markets")
    ax_a.legend(loc="upper right")
    ax_v.set_title("B. Share of trading volume by type", loc="left")
    ax_v.set_ylabel("Percent of gross trading volume")

    # C. mean relative final wealth by type and treatment, HC1 95% CIs
    # across trader-markets
    for i, t in enumerate(TREATMENTS):
        sub = tm[tm["treatment"] == t]
        means, cis = [], []
        for ty in TYPE_ORDER:
            m, ci = mean_ci(sub.loc[sub["type"] == ty, "rel_wealth"])
            means.append(m)
            cis.append(ci)
        ax_b.bar(
            x + (i - 0.5) * width, means, width=width * 0.92,
            color=COLORS[t], alpha=0.85, zorder=2,
        )
        ax_b.errorbar(
            x + (i - 0.5) * width, means, yerr=cis, fmt="none",
            ecolor="black", elinewidth=1.1, capsize=4, capthick=1.1, zorder=3,
        )
    ax_b.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax_b.set_xticks(range(len(TYPE_ORDER)))
    ax_b.set_xticklabels(
        [TYPE_LABELS[ty] for ty in TYPE_ORDER], rotation=30, ha="right"
    )
    ax_b.set_title("C. Payoff by trader type", loc="left")
    ax_b.set_ylabel("Final wealth relative to market mean\n(exp. currency)")

    fig.tight_layout(w_pad=2.5)
    save(fig, "fig6_trader_types")


# ----------------------------------------------------------------------
# Figure 7: forecast accuracy
# ----------------------------------------------------------------------

def fig_forecasts(trd: pd.DataFrame) -> None:
    """Next-day price forecasts (elicited on days 3, 6, 9, 12) versus the
    realized next-day average price and the next-day fundamental value.
    Trader-level errors are aggregated to the market-day median (robust to
    fat-finger forecasts), then averaged with HC1 CIs; all errors are
    normalized by the horizon-average fundamental (v-bar = 64), as in RAD."""
    # forecast_err_* / forecast_bias_* come from the panel (normalized by
    # v-bar, fat-finger entries already set to NaN there)
    sub = trd.dropna(subset=["forecast"]).rename(
        columns={
            "forecast_err_price": "err_price",
            "forecast_err_fund": "err_fund",
            "forecast_bias_fund": "bias_fund",
        }
    )

    med = (
        sub.groupby(["market_uuid", "treatment", "trading_day"])[
            ["err_price", "err_fund", "bias_fund"]
        ]
        .median()
        .reset_index()
    )

    METRICS = [
        (
            "err_price",
            "Forecast error vs next-day price",
            r"$|F_{it} - \bar{P}_{t+1}|\,/\,\bar{v}$",
        ),
        (
            "err_fund",
            "Forecast error vs fundamental",
            r"$|F_{it} - v_{t+1}|\,/\,\bar{v}$",
        ),
        (
            "bias_fund",
            "Forecast bias vs fundamental",
            r"$(F_{it} - v_{t+1})\,/\,\bar{v}$",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))

    # top row: day paths by treatment (forecasts elicited on days 3,6,9,12)
    for k, (ax, (col, title, ylab)) in enumerate(zip(axes[0], METRICS)):
        g = med.groupby(["treatment", "trading_day"])[col]
        path = g.agg(mean="mean", sd="std", n="count").reset_index()
        path["ci"] = Z95 * path["sd"] / np.sqrt(path["n"])
        for t in TREATMENTS:
            p = path[path["treatment"] == t]
            ax.errorbar(
                p["trading_day"], p["mean"], yerr=p["ci"],
                color=COLORS[t], lw=1.6, marker="o", ms=4.5,
                capsize=3.5, capthick=1.0, elinewidth=1.0, label=LABELS[t],
            )
        ax.set_xticks(sorted(med["trading_day"].unique()))
        ax.set_xlabel("Trading day of forecast")
        ax.set_title(f"{'ABC'[k]}. {title}", loc="left")
        ax.set_ylabel("median$_i$ " + ylab)
        if col == "bias_fund":
            ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    axes[0][0].legend(loc="upper left")

    # bottom row: same metrics by trader type and treatment (median across
    # a trader's four forecasts, HC1 95% CIs across trader-markets)
    types = _trader_types(trd)[["market_uuid", "participant_code", "type"]]
    per_trader = (
        sub.groupby(["market_uuid", "participant_code", "treatment"])[
            ["err_price", "err_fund", "bias_fund"]
        ]
        .median()
        .reset_index()
        .merge(types, on=["market_uuid", "participant_code"])
    )
    width = 0.38
    x = np.arange(len(TYPE_ORDER))
    for k, (ax, (col, title, ylab)) in enumerate(zip(axes[1], METRICS)):
        for i, t in enumerate(TREATMENTS):
            s = per_trader[per_trader["treatment"] == t]
            means, cis = [], []
            for ty in TYPE_ORDER:
                m, ci = mean_ci(s.loc[s["type"] == ty, col])
                means.append(m)
                cis.append(ci)
            ax.bar(
                x + (i - 0.5) * width, means, width=width * 0.92,
                color=COLORS[t], alpha=0.85, zorder=2,
            )
            ax.errorbar(
                x + (i - 0.5) * width, means, yerr=cis, fmt="none",
                ecolor="black", elinewidth=1.1, capsize=3.5, capthick=1.0,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [TYPE_LABELS[ty] for ty in TYPE_ORDER], rotation=30, ha="right"
        )
        ax.set_title(f"{'DEF'[k]}. {title}, by type", loc="left")
        ax.set_ylabel("median$_t$ " + ylab)
        if col == "bias_fund":
            ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    fig.tight_layout(w_pad=2.2, h_pad=2.6)
    save(fig, "fig7_forecast_accuracy")


# ----------------------------------------------------------------------
# Figure 8: bubble incidence
# ----------------------------------------------------------------------

def fig_bubble_incidence(mkt: pd.DataFrame) -> None:
    """Counts of flagged extreme-price events per market-rep (+/- 2 sigma
    flags from the build pipeline: bubble_period on normalized mispricing,
    surge/crash on day-over-day returns; bubble_start marks the first day
    of each distinct bubble episode)."""
    rep = (
        mkt.groupby(["market_uuid", "treatment"])[
            ["bubble_period", "bubble_start", "surge", "crash"]
        ]
        .sum()
        .reset_index()
    )
    specs = [
        ("bubble_period", "A. Bubble days", "Days flagged per market-rep"),
        ("bubble_start", "B. Bubble episodes", "Episodes per market-rep"),
        ("surge", "C. Price surges", "Days flagged per market-rep"),
        ("crash", "D. Price crashes", "Days flagged per market-rep"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.0))
    for ax, (col, title, ylab) in zip(axes, specs):
        draw_bars(ax, rep, col)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig8_bubble_incidence")


# ----------------------------------------------------------------------

def main() -> None:
    mkt = load_market_panel()
    trd = load_trader_panel()
    print(
        "Sample:",
        {t: int(n) for t, n in mkt.groupby("treatment")["market_uuid"].nunique().items()},
        "market-reps,", mkt.shape[0], "market-days",
    )
    fig_price_paths(mkt)
    fig_carry(mkt)
    fig_mispricing(mkt)
    fig_liquidity(mkt)
    fig_volume_orderflow(mkt)
    fig_literacy(mkt, trd)
    fig_liquidity_provision(mkt)
    fig_trader_types(trd)
    fig_forecasts(trd)
    fig_bubble_incidence(mkt)


if __name__ == "__main__":
    main()
