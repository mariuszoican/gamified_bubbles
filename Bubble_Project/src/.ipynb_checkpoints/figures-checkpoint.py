# FIGURES.PY
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("../project/processed_data")
FIG_DIR = Path("../project/figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def load_data():
    out = {}
    for fn in [
        "market_period.csv",
        "market_summary.csv",
        "market_type_shares.csv",
        "forecast_panel.csv",
    ]:
        path = DATA_DIR / fn
        if path.exists():
            out[fn.replace(".csv", "")] = pd.read_csv(path)
    return out


def treatment_label(cell):
    return {
        "gh": "Gamified Human",
        "gm": "Gamified Mixed",
        "nh": "Non-Gamified Human",
        "nm": "Non-Gamified Mixed",
    }.get(cell, cell)


def savefig(name):
    if not name.endswith(".eps"):
        name = f"{name}.eps"
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, format="eps", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def prepare(df):
    df = df.copy()

    if "treatment" in df.columns:
        df["cell"] = df["treatment"].astype(str)
    else:
        if {"gamified", "hybrid"}.issubset(df.columns):

            def make_cell(row):
                if row["gamified"] == 1 and row["hybrid"] == 0:
                    return "gh"
                elif row["gamified"] == 1 and row["hybrid"] == 1:
                    return "gm"
                elif row["gamified"] == 0 and row["hybrid"] == 0:
                    return "nh"
                elif row["gamified"] == 0 and row["hybrid"] == 1:
                    return "nm"
                return None

            df["cell"] = df.apply(make_cell, axis=1)
        else:
            df["cell"] = None

    return df


def plot_price_paths(mp):
    mp = prepare(mp)
    mp = mp[mp["repetition"] == 1]

    cells = ["gh", "gm", "nh", "nm"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, c in zip(axes, cells):
        g = (
            mp[mp["cell"] == c]
            .groupby("trading_day", as_index=False)
            .agg(
                avg_trade_price=("avg_trade_price", "mean"),
                fundamental_value=("fundamental_value", "mean"),
            )
            .sort_values("trading_day")
        )

        if g.empty:
            ax.text(
                0.5,
                0.5,
                "No raw_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(treatment_label(c))
            continue

        ax.plot(
            g["trading_day"],
            g["avg_trade_price"],
            marker="o",
            label="Average trade price",
        )
        ax.plot(
            g["trading_day"],
            g["fundamental_value"],
            marker="s",
            label="Fundamental value",
        )
        ax.set_title(treatment_label(c))
        ax.set_xlabel("Period")
        ax.set_ylabel("Price")
        ax.legend(frameon=False)

    plt.suptitle("Price Paths by Treatment")
    savefig("price_paths_2x2")


def plot_mispricing_paths(mp):
    mp = prepare(mp)
    mp = mp[mp["repetition"] == 1]

    cells = ["gh", "gm", "nh", "nm"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, c in zip(axes, cells):
        g = (
            mp[mp["cell"] == c]
            .groupby("trading_day", as_index=False)
            .agg(
                absolute_mispricing=("absolute_mispricing", "mean"),
                abs_mispricing_ratio=("abs_mispricing_ratio", "mean"),
            )
            .sort_values("trading_day")
        )

        if g.empty:
            ax.text(
                0.5,
                0.5,
                "No raw_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(treatment_label(c))
            continue

        ax.plot(
            g["trading_day"],
            g["absolute_mispricing"],
            marker="o",
            label="Absolute mispricing",
        )
        ax.plot(
            g["trading_day"],
            g["abs_mispricing_ratio"],
            marker="s",
            label="Absolute mispricing ratio",
        )
        ax.set_title(treatment_label(c))
        ax.set_xlabel("Period")
        ax.set_ylabel("Mispricing")
        ax.legend(frameon=False)

    plt.suptitle("Mispricing Paths by Treatment")
    savefig("mispricing_paths_2x2")


def plot_surges_and_bubbles(mp):
    mp = prepare(mp)
    mp = mp[mp["repetition"] == 1]

    cells = ["gh", "gm", "nh", "nm"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, c in zip(axes, cells):
        g = (
            mp[mp["cell"] == c]
            .groupby("trading_day", as_index=False)
            .agg(
                mean_return=("return", "mean"),
                surge_share=("surge", "mean"),
                crash_share=("crash", "mean"),
                bubble_share=("bubble_period", "mean"),
            )
            .sort_values("trading_day")
        )

        if g.empty:
            ax.text(
                0.5,
                0.5,
                "No raw_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(treatment_label(c))
            continue

        ax.plot(g["trading_day"], g["mean_return"], marker="o", label="Mean return")
        ax.plot(g["trading_day"], g["surge_share"], marker="s", label="Surge share")
        ax.plot(g["trading_day"], g["crash_share"], marker="^", label="Crash share")
        ax.plot(g["trading_day"], g["bubble_share"], marker="d", label="Bubble share")
        ax.set_title(treatment_label(c))
        ax.set_xlabel("Period")
        ax.set_ylabel("Return / Share")
        ax.legend(frameon=False)

    plt.suptitle("Returns, Surges, Crashes, and Bubble Share")
    savefig("surges_crashes_bubbles_2x2")


def plot_wealth_inequality(ms):
    ms = prepare(ms)
    cells = ["gh", "gm", "nh", "nm"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axes = axes.flatten()

    for ax, c in zip(axes, cells):
        g = (
            ms[ms["cell"] == c]
            .groupby("repetition", as_index=False)
            .agg(
                wealth_gini=("wealth_gini", "mean"),
            )
            .sort_values("repetition")
        )

        if g.empty or "wealth_gini" not in g.columns:
            ax.text(
                0.5,
                0.5,
                "No raw_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(treatment_label(c))
            continue

        xlabels = [f"Market {int(r)}" for r in g["repetition"] if pd.notna(r)]
        if len(xlabels) == 0:
            ax.text(
                0.5,
                0.5,
                "No raw_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(treatment_label(c))
            continue

        ax.bar(xlabels, g["wealth_gini"])
        ax.set_title(treatment_label(c))
        ax.set_ylabel("Gini")

    plt.suptitle("Wealth Inequality (Gini)")
    savefig("wealth_gini_2x2")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axes = axes.flatten()

    for ax, c in zip(axes, cells):
        g = (
            ms[ms["cell"] == c]
            .groupby("repetition", as_index=False)
            .agg(
                wealth_sd=("wealth_sd", "mean"),
            )
            .sort_values("repetition")
        )

        if g.empty or "wealth_sd" not in g.columns:
            ax.text(
                0.5,
                0.5,
                "No raw_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(treatment_label(c))
            continue

        xlabels = [f"Market {int(r)}" for r in g["repetition"] if pd.notna(r)]
        if len(xlabels) == 0:
            ax.text(
                0.5,
                0.5,
                "No raw_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(treatment_label(c))
            continue

        ax.bar(xlabels, g["wealth_sd"])
        ax.set_title(treatment_label(c))
        ax.set_ylabel("Wealth SD")

    plt.suptitle("Wealth Dispersion")
    savefig("wealth_sd_2x2")


def plot_trader_type_shares(mts):
    mts = prepare(mts)

    cells = ["gh", "gm", "nh", "nm"]
    share_cols = [
        "share_feedback",
        "share_speculator",
        "share_fundamental",
        "share_other",
    ]
    labels = ["Feedback", "Speculator", "Fundamental", "Other"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, c in zip(axes, cells):
        g = (
            mts[mts["cell"] == c]
            .groupby("repetition", as_index=False)
            .agg(
                share_feedback=("share_feedback", "mean"),
                share_speculator=("share_speculator", "mean"),
                share_fundamental=("share_fundamental", "mean"),
                share_other=("share_other", "mean"),
            )
            .sort_values("repetition")
        )

        if g.empty:
            ax.text(
                0.5,
                0.5,
                "No raw_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(treatment_label(c))
            continue

        xlabels = [f"Market {int(r)}" for r in g["repetition"] if pd.notna(r)]
        if len(xlabels) == 0:
            ax.text(
                0.5,
                0.5,
                "No raw_data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(treatment_label(c))
            continue

        bottom = None

        for col, lab in zip(share_cols, labels):
            if col not in g.columns:
                continue
            vals = g[col].fillna(0).values
            if bottom is None:
                ax.bar(xlabels, vals, label=lab)
                bottom = vals.copy()
            else:
                ax.bar(xlabels, vals, bottom=bottom, label=lab)
                bottom = bottom + vals

        ax.set_title(treatment_label(c))
        ax.set_ylabel("Share")
        ax.set_ylim(0, 1)

    axes[0].legend(labels, frameon=False)
    plt.suptitle("Trader Type Shares by Treatment")
    savefig("trader_type_shares_2x2")


def plot_forecast_extrapolation(fp):
    if not {"delta_p", "forecast_gap"}.issubset(fp.columns):
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(fp["delta_p"], fp["forecast_gap"])
    plt.xlabel("Recent price change")
    plt.ylabel("Forecast gap")
    plt.title("Trend Extrapolation")
    savefig("forecast_extrapolation_scatter")


def main():
    data = load_data()

    if "market_period" in data:
        plot_price_paths(data["market_period"])
        plot_mispricing_paths(data["market_period"])
        plot_surges_and_bubbles(data["market_period"])

    if "market_summary" in data:
        plot_wealth_inequality(data["market_summary"])

    if "market_type_shares" in data:
        plot_trader_type_shares(data["market_type_shares"])

    if "forecast_panel" in data:
        plot_forecast_extrapolation(data["forecast_panel"])

    print(f"Saved figures to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
