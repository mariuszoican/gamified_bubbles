"""Quick exploratory plots against the full analysis panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

market = pd.read_csv(PROCESSED / "market_day_panel_full.csv")
trader = pd.read_csv(PROCESSED / "trader_day_panel_full.csv")

sns.barplot(data=market, x="trading_day", y="avg_mispricing", hue="treatment")
plt.show()

sns.barplot(data=market, x="trading_day", y="share_speculator", hue="gamified")
plt.show()

sns.barplot(
    data=market,
    x="trading_day",
    y="avg_trade_price",
    hue="gamified",
)
plt.show()

sns.barplot(data=market, x="treatment", y="n_trades_market")
plt.show()

# --- Price paths by treatment + fundamental value ---
TREATMENT_ORDER = ["ng", "gh", "gp", "ghp"]
TREATMENT_LABELS = {
    "ng": "Non-gamified",
    "gh": "Hedonic only",
    "gp": "Price notifications only",
    "ghp": "Hedonic + notifications",
}
plot_df = market.copy()
plot_df["treatment_label"] = plot_df["treatment"].map(TREATMENT_LABELS)

fv = (
    market.groupby("trading_day", as_index=False)["fundamental_value"]
    .first()
    .sort_values("trading_day")
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(
    data=plot_df,
    x="trading_day",
    y="avg_trade_price",
    hue="treatment_label",
    hue_order=[TREATMENT_LABELS[t] for t in TREATMENT_ORDER],
    errorbar=("ci", 95),
    marker="o",
    linewidth=2,
    ax=ax,
)
ax.plot(
    fv["trading_day"],
    fv["fundamental_value"],
    color="black",
    linestyle="--",
    linewidth=2,
    label="Fundamental value",
    zorder=5,
)
ax.set_xlabel("Trading day")
ax.set_ylabel("Average trade price")
ax.set_title("Price paths by treatment (95% CI)")
ax.set_xticks(range(1, 16))
ax.grid(False)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
sns.despine(ax=ax)
ax.legend(title="", frameon=False, loc="best")
fig.tight_layout()
plt.show()
