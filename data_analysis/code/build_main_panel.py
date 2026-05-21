import os
import pandas as pd
from process_session import process_session

# ============================================================
# 1. Build per-session panels
# ============================================================
# Pilot (excluded from main panels):
#   skdclskt - May 5, 2026, 24 people, no training round
#
# Main sessions:
#   ffe5xhmq - May 12, 2026, 14 people (1 ghp + 1 ng group of 6, 1 group of 2 with bots)
#   mn8370qr - May 20, 2026 AM, 16 people (1 gh + 1 gp group of 6, 1 ghp group of 4)
#   qxgij9sh - May 20, 2026 PM, 19 people (2 ghp + 1 ng groups of 6, 1 with bots)

runs = [
    ("2026-05-12", "20260512", ["ffe5xhmq"]),
    ("2026-05-20", "20260520_AM", ["mn8370qr"]),
    ("2026-05-21", "20260520_PM", ["qxgij9sh"]),
]

for date, folder, sessions in runs:
    process_session(date, folder, sessions)

# ============================================================
# 2. Concatenate into full panels
# ============================================================

PROCESSED_DIR = "../processed_data"
OUTPUT_DIR = "../main_panels"
PANEL_NAMES = ["market_day_panel", "trader_day_panel", "participant_payments"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

session_folders = sorted(
    f
    for f in os.listdir(PROCESSED_DIR)
    if not f.startswith(".") and os.path.isdir(f"{PROCESSED_DIR}/{f}")
)
print(f"Concatenating {len(session_folders)} sessions: {session_folders}")

for name in PANEL_NAMES:
    frames = [pd.read_csv(f"{PROCESSED_DIR}/{s}/{name}.csv") for s in session_folders]
    pd.concat(frames, ignore_index=True).to_csv(
        f"{OUTPUT_DIR}/{name}_full.csv", index=False
    )
