"""
Read one registered lab session from data/raw/ and write a payment workbook.

Output: data/payments/payments_{session_id}.xlsx (lab folder date).
"""

from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

from paths import (
    get_session,
    load_parameters,
    payments_xlsx_path,
    raw_dir_for,
    session_log_path,
)
from session_log import (
    CSV_READ_KW,
    build_session_record,
    classify_participants,
    collect_quality_flags,
    email_column,
    format_stdout,
    student_id_column,
    write_session_log,
)

LAB_COLUMNS = [
    "email",
    "student_id",
    "participation_fee",
    "experimental_payoff",
    "total_payment",
]


def money(value) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def _has_value(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("") & series.astype(
        str
    ).str.strip().str.lower().ne("nan")


def build_payments(
    completed: pd.DataFrame,
    *,
    exchange_rate: float,
    participation_fee: float,
) -> pd.DataFrame:
    if completed.empty:
        return pd.DataFrame(columns=LAB_COLUMNS)

    email = completed[email_column(completed)].astype(str).str.strip()
    sid_col = student_id_column(completed)
    sid = (
        completed[sid_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        if sid_col
        else pd.Series("", index=completed.index)
    )
    out = pd.DataFrame(
        {
            "email": email,
            "student_id": sid,
            "experimental_payoff_points": pd.to_numeric(
                completed["participant.payoff"], errors="coerce"
            ),
        }
    )
    out = out.loc[_has_value(out["email"]) | _has_value(out["student_id"])].copy()
    out["experimental_payoff"] = out["experimental_payoff_points"].map(
        lambda pts: money(float(pts) * exchange_rate) if pd.notna(pts) else None
    )
    out["participation_fee"] = money(participation_fee)
    out["total_payment"] = out.apply(
        lambda r: money(r["participation_fee"] + r["experimental_payoff"])
        if pd.notna(r["experimental_payoff"])
        else None,
        axis=1,
    )
    return (
        out[LAB_COLUMNS]
        .sort_values(["email", "student_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def write_payments_xlsx(payments: pd.DataFrame, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        payments.to_excel(writer, sheet_name="payments", index=False)
        ws = writer.sheets["payments"]
        for row in range(2, len(payments) + 2):
            ws[f"B{row}"].number_format = "@"
            if ws[f"B{row}"].value is not None:
                ws[f"B{row}"].value = str(ws[f"B{row}"].value)
            for col in ("C", "D", "E"):
                ws[f"{col}{row}"].number_format = '"$"#,##0.00'
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes = "A2"
    return dest


def load_post_exp(raw_path, export_date: str, oTree_codes: list[str]) -> pd.DataFrame:
    post_exp = pd.read_csv(
        raw_path / f"post_exp_{export_date}.csv",
        **CSV_READ_KW,
        dtype={
            "player.email": "string",
            "player.student_email": "string",
            "player.student_id": "string",
            "player.ucid": "string",
        },
    )
    return post_exp[post_exp["session.code"].isin(list(oTree_codes))].copy()


def process_payments(session_id: str) -> dict:
    session = get_session(session_id)
    params = load_parameters()
    raw_path = raw_dir_for(session)
    if not raw_path.is_dir():
        raise FileNotFoundError(
            f"Raw folder not found: {raw_path}. "
            "Drop the oTree CSVs under data/raw/{session_id}/ and register the session."
        )

    post_exp = load_post_exp(raw_path, session["export_date"], session["oTree_codes"])
    groups = classify_participants(
        post_exp, completed_pages=list(params["completed_pages"])
    )
    payments = build_payments(
        groups["completed"],
        exchange_rate=params["exchange_rate"],
        participation_fee=params["participation_fee"],
    )
    flags = collect_quality_flags(groups=groups, payments=payments)
    record = build_session_record(
        session=session,
        groups=groups,
        payments=payments,
        params=params,
        flags=flags,
    )

    xlsx_path = write_payments_xlsx(payments, payments_xlsx_path(session["id"]))
    write_session_log(record, session_log_path(session["id"]))

    print(format_stdout(record))
    print(f"  wrote {len(payments)} payments -> {xlsx_path}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the payment workbook for one registered lab session."
    )
    parser.add_argument(
        "--session",
        required=True,
        help="Session id from config/sessions.yaml",
    )
    args = parser.parse_args()
    process_payments(args.session)
