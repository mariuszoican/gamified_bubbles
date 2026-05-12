import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

market = pd.read_csv("../processed_data/20260512/market_day_panel.csv")
trader = pd.read_csv("../processed_data/20260512/trader_day_panel.csv")

# session_list = [""]

sns.barplot(data=market, x="trading_day", y="avg_mispricing", hue="gamified")
plt.show()

sns.barplot(data=market, x="trading_day", y="share_feedback", hue="gamified")
plt.show()

sns.barplot(
    data=market,
    x="trading_day",
    y="avg_trade_price",
    hue="gamified",
)
plt.show()

sns.barplot(data=market, x="gamified", y="n_trades_market")
plt.show()

mbo = pd.read_csv(
    "../raw_data/20260505/trader_bridge_app_custom_export_mbo_2026-05-05.csv"
)
trades = mbo[mbo.event_type == "trade"]
