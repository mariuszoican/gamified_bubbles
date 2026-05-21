import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

market = pd.read_csv("../main_panels/market_day_panel_full.csv")
trader = pd.read_csv("../main_panels/trader_day_panel_full.csv")

# session_list = [""]

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
