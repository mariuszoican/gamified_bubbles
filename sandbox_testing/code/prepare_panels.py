from config import *

import os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


def _print_filter_stats(label: str, before_n: int, after_n: int) -> None:
    print(f"[FILTER] {label}: {before_n} -> {after_n} (dropped {before_n - after_n})")


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def main() -> None:
    # ── SECTION: LOAD RAW DATA ───────────────────────────────────────────────
    main_path = f"{DATA_DIR}/trader_bridge_app_{DATE}.csv"
    post_path = f"{DATA_DIR}/post_exp_{DATE}.csv"
    intro_path = f"{DATA_DIR}/intro_{DATE}.csv"

    df = pd.read_csv(main_path).rename(columns=RENAME, errors="ignore")
    post = pd.read_csv(post_path).rename(columns=RENAME, errors="ignore")
    intro = pd.read_csv(intro_path).rename(columns=RENAME, errors="ignore")

    print(f"[LOAD] main:  {main_path} -> {df.shape}")
    print(f"[LOAD] post:  {post_path} -> {post.shape}")
    print(f"[LOAD] intro: {intro_path} -> {intro.shape}")

    # ── SECTION: FILTER (main table only) ────────────────────────────────────
    before = len(df)
    df = df[df["session_code"].isin(SESSION_CODES)]
    _print_filter_stats("session_code in SESSION_CODES", before, len(df))

    before = len(df)
    df = df[df["current_page"] == COMPLETION_PAGE]
    _print_filter_stats("current_page == COMPLETION_PAGE", before, len(df))

    before = len(df)
    df = df[df["is_bot"] == 0]
    _print_filter_stats("is_bot == 0", before, len(df))

    # ── SECTION: DERIVE PERIOD & MARKET COLUMNS ──────────────────────────────
    df["trading_day"] = ((df["otree_round"] - 1) % N_PERIODS) + 1
    df["market_rep"] = (df["otree_round"] > N_PERIODS).astype(int) + 1
    df["is_rep2"] = (df["market_rep"] == 2).astype(int)

    # ── SECTION: TREATMENT DUMMIES ───────────────────────────────────────────
    df["is_gamified"] = (df["market_design"] == "gamified").astype(int)
    df["is_hybrid"] = (df["group_composition"] == "mixed").astype(int)
    df["gamified_x_hybrid"] = df["is_gamified"] * df["is_hybrid"]

    # ── SECTION: MERGE POST-EXP ──────────────────────────────────────────────
    post_cols = [
        "participant_code",
        "fin_lit_score",
        "age",
        "gender",
        "trading_experience",
        "finance_course",
        "quiz_payoff",
        "n_quiz_questions",
    ]
    post_keep = [c for c in post_cols if c in post.columns]
    post_f = post[post["participant_code"].isin(df["participant_code"])][post_keep].copy()
    post_f = post_f.drop_duplicates(subset=["participant_code"])
    df = df.merge(post_f, on="participant_code", how="left")
    print(f"[MERGE] post_exp merged cols: {[c for c in post_keep if c != 'participant_code']}")

    # ── SECTION: MERGE INTRO ─────────────────────────────────────────────────
    intro_cols = ["participant_code", "self_assess"]
    intro_keep = [c for c in intro_cols if c in intro.columns]
    intro_f = intro[intro["participant_code"].isin(df["participant_code"])][intro_keep].copy()
    intro_f = intro_f.drop_duplicates(subset=["participant_code"])
    df = df.merge(intro_f, on="participant_code", how="left")
    print(f"[MERGE] intro merged cols: {[c for c in intro_keep if c != 'participant_code']}")

    # ── SECTION: CONSTRUCT DERIVED VARIABLES ─────────────────────────────────
    if "n_quiz_questions" in df.columns:
        denom = df["n_quiz_questions"].replace({0: np.nan})
        fin_lit_norm = df["fin_lit_score"] / denom
    else:
        # Fallback when n_quiz_questions is unavailable in export.
        fin_lit_norm = pd.Series(np.nan, index=df.index)
    df["overconfidence"] = (df["self_assess"] / 10) - fin_lit_norm
    df["gender_female"] = (df["gender"] == "Female").astype(int)

    # ── SECTION: STANDARDISE ─────────────────────────────────────────────────
    for col in CONTROLS_TO_STANDARDISE:
        if col not in df.columns:
            continue
        df[f"{col}_raw"] = df[col]
        df[col] = _zscore(df[col])

    # ── SECTION: ASSERTIONS ──────────────────────────────────────────────────
    assert df["trading_day"].between(1, N_PERIODS).all()
    assert df["market_rep"].isin([1, 2]).all()
    assert df["is_gamified"].isin([0, 1]).all()
    print(f"[CHECK] Final panel shape: {df.shape}")

    # ── SECTION: SAVE ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/panel_trader_period.csv"
    df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path} — {df.shape}")
    print(df.head(3))


if __name__ == "__main__":
    main()
