"""Session logging: classify participants and write a YAML sidecar."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

CSV_READ_KW = dict(encoding="utf-8-sig")


def nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("") & series.astype(
        str
    ).str.strip().str.lower().ne("nan")


def email_column(df: pd.DataFrame) -> str:
    for name in ("player.email", "player.student_email"):
        if name in df.columns:
            return name
    raise KeyError("No email column: expected player.email or player.student_email")


def student_id_column(df: pd.DataFrame) -> str | None:
    for name in ("player.student_id", "player.ucid"):
        if name in df.columns:
            return name
    return None


def classify_participants(
    post_exp: pd.DataFrame, *, completed_pages: list[str]
) -> dict[str, pd.DataFrame]:
    """Split an oTree post_exp table into completed / incomplete / never_started."""
    visited = post_exp["participant.visited"].fillna(0).astype(int).eq(1)
    page = post_exp["participant._current_page_name"].fillna("").astype(str).str.strip()
    completed = visited & page.isin(completed_pages)
    incomplete = visited & ~page.isin(completed_pages)
    never_started = ~visited
    return {
        "completed": post_exp.loc[completed].copy(),
        "incomplete": post_exp.loc[incomplete].copy(),
        "never_started": post_exp.loc[never_started].copy(),
    }


def _row_summaries(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "participant_code": r.get("participant.code"),
                "page": r.get("participant._current_page_name") or None,
                "app": r.get("participant._current_app_name") or None,
                "index_in_pages": _as_int(r.get("participant._index_in_pages")),
                "time_started_utc": r.get("participant.time_started_utc") or None,
                "payoff_points": _as_float(r.get("participant.payoff")),
            }
        )
    return rows


def _as_int(value):
    try:
        if pd.isna(value) or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value):
    try:
        if pd.isna(value) or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_session_record(
    *,
    session: dict,
    groups: dict[str, pd.DataFrame],
    payments: pd.DataFrame | None,
    params: dict,
    flags: list[str],
    processed_at: str | None = None,
) -> dict:
    completed = groups["completed"]
    incomplete = groups["incomplete"]
    never = groups["never_started"]
    payoffs = pd.to_numeric(completed.get("participant.payoff"), errors="coerce")

    exp_sum = None
    total_sum = None
    if payments is not None and not payments.empty:
        exp_sum = round(float(payments["experimental_payoff"].sum()), 2)
        total_sum = round(float(payments["total_payment"].sum()), 2)

    return {
        "session_id": session["id"],
        "export_date": session["export_date"],
        "oTree_codes": list(session["oTree_codes"]),
        "notes": session.get("notes", ""),
        "processed_at": processed_at or datetime.now(timezone.utc).isoformat(),
        "counts": {
            "slots_in_export": int(len(completed) + len(incomplete) + len(never)),
            "started": int(len(completed) + len(incomplete)),
            "completed": int(len(completed)),
            "incomplete": int(len(incomplete)),
            "never_started": int(len(never)),
            "paid": int(len(payments)) if payments is not None else 0,
        },
        "payoffs_points": {
            "n": int(payoffs.notna().sum()),
            "min": None if payoffs.dropna().empty else float(payoffs.min()),
            "max": None if payoffs.dropna().empty else float(payoffs.max()),
            "mean": None if payoffs.dropna().empty else float(payoffs.mean()),
        },
        "payments": {
            "participation_fee_cad": params["participation_fee"],
            "exchange_rate": params["exchange_rate"],
            "experimental_payoff_cad_sum": exp_sum,
            "total_payment_cad_sum": total_sum,
        },
        "incomplete": _row_summaries(incomplete),
        "flags": flags,
    }


def write_session_log(record: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True)
    return path


def format_stdout(record: dict) -> str:
    c = record["counts"]
    p = record["payments"]
    lines = [
        f"Session {record['session_id']}  (oTree {', '.join(record['oTree_codes'])})",
        f"  started {c['started']}  completed {c['completed']}  "
        f"incomplete {c['incomplete']}  unused slots {c['never_started']}",
        f"  paid {c['paid']}  experimental CAD {p['experimental_payoff_cad_sum']}  "
        f"total CAD {p['total_payment_cad_sum']}",
    ]
    if record.get("flags"):
        lines.append("  flags: " + "; ".join(record["flags"]))
    for row in record.get("incomplete") or []:
        lines.append(
            f"  incomplete: {row['participant_code']}  {row['app']}/{row['page']}"
        )
    return "\n".join(lines)


def collect_quality_flags(
    *,
    groups: dict[str, pd.DataFrame],
    payments: pd.DataFrame,
) -> list[str]:
    flags: list[str] = []
    completed = groups["completed"]
    if completed.empty:
        flags.append("no completers")
        return flags

    try:
        email = completed[email_column(completed)]
    except KeyError:
        email = pd.Series(dtype=str)
    sid_col = student_id_column(completed)
    sid = completed[sid_col] if sid_col else pd.Series(dtype=str)
    n_email = int(nonempty(email).sum()) if len(email) else 0
    n_sid = int(nonempty(sid).sum()) if len(sid) else 0
    if n_email < len(completed):
        flags.append(f"{len(completed) - n_email} completers missing email")
    if n_sid < len(completed):
        flags.append(f"{len(completed) - n_sid} completers missing student_id")

    if not payments.empty:
        dup_email = payments["email"].astype(str).str.lower().duplicated(keep=False)
        dup_sid = payments["student_id"].astype(str).duplicated(keep=False)
        if dup_email.any():
            flags.append("duplicate emails in payment file")
        if dup_sid.any():
            flags.append("duplicate student_ids in payment file")

    return flags
