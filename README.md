# Gamified Bubbles — analysis

Pipeline for the laboratory experiment in *Trading Gamification and Asset Prices*.
Raw oTree / trader-bridge exports are turned into trader-day and market-day panels,
then into hypothesis tests for the paper.

## Layout

```
config/
  sessions.yaml      # lab-session registry (ids, export dates, oTree codes)
  parameters.yaml    # design constants (rounds, dividend, σ thresholds, …)
data/
  raw/{session_id}/  # immutable oTree dumps — never edit
  payments/          # payments_{session id}.xlsx (lab folder date)
  interim/           # per-session panels (rebuildable)
  processed/         # concatenated analysis sample (*_full.csv)
  archive/           # pilots / excluded sessions
src/
  build/             # Python: process_session → build_panels
  analyze/           # R: hypothesis_tests.R
  explore/           # scratch plots
output/
  tables/            # TeX tables for the paper
  figures/
```

## Setup

```bash
cd gamified_bubbles_analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

R packages used by `src/analyze/hypothesis_tests.R`: `tidyverse`, `lfe`, `stargazer`,
`fixest`, `modelsummary`, `ggplot2`, `ggfixest`, `cowplot`, `latex2exp` (optional:
`rstudioapi` when sourcing from RStudio).

## How to drop new raw data

1. **Create a session folder** under `data/raw/` named by the lab calendar day:
   - `YYYYMMDD` for a single session that day
   - `YYYYMMDD_AM` / `YYYYMMDD_PM` when two sessions share a calendar day

2. **Copy the oTree export CSVs into that folder**, keeping the export filenames.
   Expected files (date stamp = oTree export date, `YYYY-MM-DD`):

   | File | Required for panels? |
   |---|---|
   | `intro_YYYY-MM-DD.csv` | yes |
   | `post_exp_YYYY-MM-DD.csv` | yes |
   | `trader_bridge_app_YYYY-MM-DD.csv` | yes |
   | `trader_bridge_app_custom_export_mbo_YYYY-MM-DD.csv` | yes |
   | `trader_bridge_app_custom_export_mbp1_YYYY-MM-DD.csv` | keep (mechanisms) |
   | `trader_bridge_app_custom_export_messages_YYYY-MM-DD.csv` | keep (notifications) |
   | `trader_bridge_app_custom_export_gamification_ui_YYYY-MM-DD.csv` | keep (badges/UI) |
   | `trader_bridge_app_custom_export_YYYY-MM-DD.csv` | keep |
   | `PageTimes-YYYY-MM-DD.csv` | optional |
   | `all_apps_wide_YYYY-MM-DD.csv` | optional |

3. **Watch the export-date vs folder-date mismatch.**
   Folder = lab day; filename date = when the CSV was exported. They can differ
   (e.g. `data/raw/20260520_PM/` holds `*_2026-05-21.csv` because the export ran
   overnight). Always record the **filename** date in the registry.

4. **Register the session** in `config/sessions.yaml`:

   ```yaml
   - id: 20260601_AM
     export_date: "2026-06-01"    # must match CSV filename stamp
     oTree_codes: [abc123xy]      # session.code values to keep
     include: true                # false → archive / exclude from full panels
     notes: "1 ghp + 1 ng"
   ```

5. **Never overwrite** an existing raw folder. New export → new folder (or move the
   old one under `data/archive/`).

6. **Do not hand-edit files under `data/raw/`.** Fix logic in `src/build/` instead.

## Payments

```bash
make payments ID=20260512
```

Writes `data/payments/payments_{session id}.xlsx` using the lab folder date,
plus a sidecar `session_log_{session id}.yaml`. Completers are people on
`FinalForProlific` or `Payoff`.

| Column | Source |
|---|---|
| `email` | `player.email` |
| `student_id` | `player.ucid` (falls back to `player.student_id`) |
| `participation_fee` | `config/parameters.yaml` ($15 show-up) |
| `experimental_payoff` | `participant.payoff` (E$) × `exchange_rate` (0.003) |
| `total_payment` | show-up + experimental payoff |

After changing the exchange rate, re-run the same command.

## Analysis flow

```
data/raw/{id}/
      │
      ▼  make session ID=…   or   make panels
data/interim/{id}/
  trader_day_panel.csv
  market_day_panel.csv
  participant_payments.csv
      │
      ▼  (make panels concatenates include:true sessions)
data/processed/
  trader_day_panel_full.csv
  market_day_panel_full.csv
  participant_payments_full.csv
      │
      ▼  make analyze
output/tables/*.tex
```

### Commands

| Command | What it does |
|---|---|
| `make payments ID=20260512` | Write `data/payments/payments_{id}.xlsx` + session log |
| `make panels` | Process every `include: true` session, then write `data/processed/*_full.csv` |
| `make session ID=20260512` | Process one session into `data/interim/` only |
| `make analyze` | Run `hypothesis_tests.R` → `output/tables/` |
| `make explore` | Quick seaborn plots against the full panels |
| `make clean-interim` | Delete rebuildable interim panels |

Equivalent without Make:

```bash
export PYTHONPATH=src/build
python src/build/build_panels.py
python src/build/process_session.py --session 20260512
Rscript src/analyze/hypothesis_tests.R
```

## Session registry conventions

- `id` — folder name under `data/raw/` (or `data/archive/` if `raw_root: archive`).
  Always quote it in YAML (`id: "20260512"`) so it is not parsed as an integer.
- `export_date` — `YYYY-MM-DD` substring in the CSV filenames.
- `oTree_codes` — `session.code` values retained; other sessions in the same export are dropped.
- `include: false` — keep raw for provenance but omit from the analysis sample.
- Design constants (rounds, dividend, bubble σ, group size, CAD exchange rate) live in
  `config/parameters.yaml` and are read by `process_session.py`.

## Notes

- Incomplete markets (`group.realized_group_size` ≠ 6) and bot groups are dropped
  inside `process_session.py`.
- Training rounds are excluded from the saved panels (`trading_day >= 1` after the
  training offset).
- Event streams not yet in the main panels (`messages`, `gamification_ui`, `mbp1`)
  should still be archived with each raw drop — they are the natural next mechanism
  panels.
