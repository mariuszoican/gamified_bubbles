"""Process included lab sessions and concatenate analysis panels."""

from __future__ import annotations

import argparse

import pandas as pd

from paths import INTERIM_DIR, PROCESSED_DIR, load_sessions
from process_session import process_session

PANEL_NAMES = ["market_day_panel", "trader_day_panel", "participant_payments"]


def build_all(*, only: list[str] | None = None, skip_process: bool = False) -> None:
    sessions = load_sessions()
    included = [s for s in sessions if s.get("include", False)]
    if only:
        only_set = set(only)
        included = [s for s in included if s["id"] in only_set]

    if not included:
        raise SystemExit("No included sessions to build (check config/sessions.yaml).")

    if not skip_process:
        for s in included:
            print(f"Processing {s['id']} (export {s['export_date']})…")
            process_session(s["id"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    session_ids = [s["id"] for s in included]
    print(f"Concatenating {len(session_ids)} sessions: {session_ids}")

    for name in PANEL_NAMES:
        frames = [
            pd.read_csv(INTERIM_DIR / sid / f"{name}.csv") for sid in session_ids
        ]
        out = PROCESSED_DIR / f"{name}_full.csv"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"  wrote {out} ({sum(len(f) for f in frames):,} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        help="Restrict to these session ids (still must have include: true)",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Only concatenate existing interim panels",
    )
    args = parser.parse_args()
    build_all(only=args.only, skip_process=args.skip_process)
