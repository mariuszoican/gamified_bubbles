import pandas as pd
import numpy as np

session_code = "a0b6151c-f772-48ce-b022-2662efead201"  # relevant session to study
session_code = "aed8dfaa-2c46-4965-ac5a-e5058b35aecf"

mbo = pd.read_csv(
    "data/trader_bridge_app_custom_export_mbo_2026-03-06.csv"
)  # load mbo raw_data
mbo = mbo[mbo.trading_session_uuid == session_code]  # only keep relevant session
mbo["direction"] = np.where(mbo["side"] == "bid", "buy", "sell")

mbp = pd.read_csv(
    "data/trader_bridge_app_custom_export_mbp1_2026-03-06.csv"
)  # load mbo raw_data
mbp = mbp[mbp.trading_session_uuid == session_code]  # only keep relevant session
