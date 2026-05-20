import os
import pandas as pd

# get list of sessions
list_sessions = os.listdir("../processed_data")
# remove pilot/other folders
for x in [".DS_Store", "20260505"]:
    list_sessions.remove(x)

market_day = pd.DataFrame()
trader_day = pd.DataFrame()
participant_payments = pd.DataFrame()

for session in list_sessions:
    print(session)
    market_temp = pd.read_csv(f"../processed_data/{session}/market_day_panel.csv")
    trader_temp = pd.read_csv(f"../processed_data/{session}/trader_day_panel.csv")
    payments_temp = pd.read_csv(f"../processed_data/{session}/participant_payments.csv")

    market_day = pd.concat([market_day, market_temp], ignore_index=True)
    trader_day = pd.concat([trader_day, trader_temp], ignore_index=True)
    participant_payments = pd.concat(
        [participant_payments, payments_temp], ignore_index=True
    )

market_day.to_csv(f"../main_panels/market_day_panel_full.csv", index=False)
trader_day.to_csv(f"../main_panels/trader_day_panel_full.csv", index=False)
participant_payments.to_csv(
    f"../main_panels/participant_payments_full.csv", index=False
)
