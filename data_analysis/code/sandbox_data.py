import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

market = pd.read_csv("../processed_data/20260505/market_day_panel.csv")

# session_list = [""]

sns.barplot(data=market, x="trading_day", y="avg_mispricing", hue="gamified")
plt.show()

sns.barplot(data=market, x="trading_day", y="share_feedback", hue="gamified")
plt.show()

sns.barplot(
    data=market[market.repetition == 2],
    x="trading_day",
    y="avg_trade_price",
    hue="gamified",
)
plt.show()

sns.barplot(data=market, x="gamified", y="n_trades_market")
plt.show()
