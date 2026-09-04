"""
Error-correction regression: does gamification switch off the force that
pulls prices back to fundamentals?

Pooled day-level regression on the processed market-day panel (GHP and NG
market-reps; outliers 20260520_PM/ng1 and 20280904/ghp1 excluded):

    ret_next[m,t] = b0 + b1 gamified + b2 gap + b3 gap x gamified
                       + b4 OFI + b5 OFI x gamified (+ day/rep controls)

where ret_next is the next-day log closing-price change, gap is the
normalized fundamental gap (P - v_t)/vbar (positive = overpriced), and OFI
is the signed order-flow imbalance. b2 < 0 is error correction in control
markets; b3 > 0 means gamification weakens it.

Standard errors are cluster-robust (CR1) by experimental group. With 11
clusters they are still anti-conservative: confirm borderline p-values with a
wild cluster bootstrap before quoting them in the paper.

Writes output/tables/error_correction.csv and prints the table.
Usage:  python src/analyze/error_correction.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "tables" / "error_correction.csv"

TREATMENTS = ["ng", "ghp"]
EXCLUDE_GROUPS = {"20260520_PM/ng1", "20280904/ghp1"}

def load_panel() -> pd.DataFrame:
    """ret_next and fundamental_gap come from the processed panel."""
    mkt = pd.read_csv(ROOT / "data" / "processed" / "market_day_panel_full.csv")
    mkt = mkt[
        mkt["treatment"].isin(TREATMENTS)
        & ~mkt["group_label"].isin(EXCLUDE_GROUPS)
    ].copy()
    mkt["gap"] = mkt["fundamental_gap"]
    mkt["gamified"] = (mkt["treatment"] == "ghp").astype(float)
    return mkt.dropna(subset=["ret_next", "gap", "order_flow_imbalance"])


def ols_cr1(y: np.ndarray, X: np.ndarray, clusters: pd.Series):
    """OLS with CR1 cluster-robust standard errors."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    xtxi = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    for _, idx in pd.Series(range(n)).groupby(clusters.reset_index(drop=True)):
        Xg, ug = X[idx], resid[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    m = clusters.nunique()
    dfc = (m / (m - 1)) * ((n - 1) / (n - k))
    V = dfc * xtxi @ meat @ xtxi
    return b, np.sqrt(np.diag(V))


def run_spec(d: pd.DataFrame, day_fe: bool) -> pd.DataFrame:
    names = [
        "const", "gamified", "gap", "gap x gamified",
        "OFI", "OFI x gamified",
    ]
    cols = [
        np.ones(len(d)),
        d["gamified"],
        d["gap"],
        d["gap"] * d["gamified"],
        d["order_flow_imbalance"],
        d["order_flow_imbalance"] * d["gamified"],
    ]
    if day_fe:
        for day in sorted(d["trading_day"].unique())[1:]:
            names.append(f"day{day}")
            cols.append((d["trading_day"] == day).astype(float))
        names.append("rep2")
        cols.append((d["repetition"] == 2).astype(float))
    X = np.column_stack(cols)
    b, se = ols_cr1(d["ret_next"].to_numpy(), X, d["group_label"])
    out = pd.DataFrame({"coef": b, "se": se}, index=names)
    out["t"] = out["coef"] / out["se"]
    return out.loc[["gamified", "gap", "gap x gamified", "OFI", "OFI x gamified"]]


def main() -> None:
    d = load_panel()
    n_grp = d["group_label"].nunique()
    print(
        f"Sample: {len(d)} market-days, "
        f"{d['market_uuid'].nunique()} market-reps, {n_grp} groups\n"
    )
    tables = []
    for day_fe, label in [(False, "baseline"), (True, "day FE + rep")]:
        tab = run_spec(d, day_fe).round(4)
        tab["spec"] = label
        tables.append(tab)
        print(f"--- {label} ---")
        print(tab[["coef", "se", "t"]].to_string(), "\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(OUT)
    print(
        f"saved {OUT}\nNote: {n_grp} clusters -> CR1 SEs are "
        "anti-conservative; wild cluster bootstrap before quoting p-values."
    )


if __name__ == "__main__":
    main()
