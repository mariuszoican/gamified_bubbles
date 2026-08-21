"""Repo paths and session-registry helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARCHIVE_DIR = DATA_DIR / "archive"
OUTPUT_DIR = REPO_ROOT / "output"


def load_sessions(path: Path | None = None) -> list[dict]:
    cfg = path or (CONFIG_DIR / "sessions.yaml")
    with open(cfg) as f:
        return yaml.safe_load(f)["sessions"]


def load_parameters(path: Path | None = None) -> dict:
    cfg = path or (CONFIG_DIR / "parameters.yaml")
    with open(cfg) as f:
        return yaml.safe_load(f)


def get_session(session_id: str) -> dict:
    session_id = str(session_id)
    for s in load_sessions():
        if str(s["id"]) == session_id:
            s = dict(s)
            s["id"] = str(s["id"])
            return s
    raise KeyError(f"Session {session_id!r} not found in config/sessions.yaml")


def raw_dir_for(session: dict) -> Path:
    root = ARCHIVE_DIR if session.get("raw_root") == "archive" else RAW_DIR
    return root / str(session["id"])


def interim_dir_for(session_id: str) -> Path:
    return INTERIM_DIR / str(session_id)
