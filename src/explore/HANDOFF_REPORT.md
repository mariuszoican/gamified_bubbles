# Gamified Bubbles — analysis handoff report

*Last updated: 2026-08-29. Written as a handoff for an experienced experimental-finance
researcher joining the analysis. All numbers reproducible from scripts in `src/explore/`
against `data/processed/*_full.csv` (4 sessions) unless noted.*

## 1. Sample and conventions

- **Design**: Smith–Suchanek–Williams declining-fundamental markets (15 days, dividend 8,
  v_k = 8·(16−k): 120 → 8, v̄ = 64). Six traders per group; each group trades **two
  consecutive repetitions** (market-reps). 2×2 treatments: `ng` (control), `gh` (hedonic:
  confetti/badges), `gp` (price-trend notifications), `ghp` (both).
- **Sample**: 4 sessions (20260512, 20260520_AM, 20260520_PM, 20260826).
  **Excluded outlier**: group `20260520_PM/ng1` — ~3× the trades of any other market
  (495+578), AMR ≈ 2.6 in both reps. Exclusion decided by PI; summary stats retained in
  `preliminary_hypothesis_analysis.py`.
  **Forensics (2026-08-29)**: not one trader but a pair — `6xvvhy1g` and `9x5fpgk2`
  jointly account for ~60% of trader-side volume in both reps, each with near-perfectly
  balanced buys/sells; 31% of all matched trades are the two trading *with each other*
  (90% involve at least one of them; zero self-trades). Average price is pegged at
  ~130–131 for the entire session in **both** repetitions while v_k falls 120 → 8 — the
  market never repriced at all. This is a two-trader churn/peg pathology, not a bubble,
  which strengthens the case for exclusion. Side effect: the group's most literate
  trader (quiz = 1.00) finished with payoff 521 vs 9,064 for the top earner.
- **Usable cells (independent groups)**: ng 3, gh 2, gp 1, ghp 5 → 11 groups, 22 market-reps,
  330 market-days.
- **Unit of inference**: the 6-subject group. Market-rep-level tests (n=22) overstate
  independence; report group-level as primary or cluster by group. With 11 clusters,
  cluster-robust SEs are anti-conservative → use wild cluster bootstrap
  (`fwildclusterboot`/`boottest`) before quoting borderline p-values in the paper.
- **Mispricing measures** (per market-rep, mean over 15 days):
  - AMR (Palan 2013): mean_t |p_t − v_k| / v_k — inflated late as v_k → 8.
  - RAD (Stöckl et al. 2010): |P̄_k − v_k| / v̄ — immune to the shrinking denominator;
    preferred.
  - RD (signed): (P̄_k − v_k) / v̄.

## 2. Findings, ranked by robustness

### Tier 1 — significant at the group level (defensible now)

**Volume.** Gamified markets trade twice as much: 151 vs 74 trades per market-rep pooled
(ghp 144 vs ng 74). MW p = 0.0008 (rep), **p = 0.012 (group)**. The single most robust
treatment effect in the data. H2 is in good shape.

### Tier 2 — rep-level significant, group-level suggestive (needs the next wave)

**Intraday churn.** Trader-day directionality |B−S|/(B+S): ng 0.92 (traders are one-way
within a day) vs ghp 0.83, gh 0.75, gp 0.62. Gamified pooled vs ng p = 0.008 (rep),
p = 0.13 (group). Over the full 15 days directionality converges (~0.44 everywhere) — the
effect is specifically *intraday round-tripping*, the entertainment-trading signature.
Computed in `tick_liquidity_volatility.py` (`trader_day_directionality`).

**Liquidity is better in ghp, not worse.** Tick data (MBO/MBP1, time-weighted):
relative quoted spread ghp 0.14 vs ng 0.26 (p = 0.056 rep / 0.14 grp); relative effective
spread 0.38 vs 0.79 (**p = 0.042 rep / 0.071 grp**); midquote realized volatility 0.27 vs
0.54 (p = 0.031 rep / 0.14 grp). Depth and two-sided-book time: no difference. Script:
`tick_liquidity_volatility.py` → `tick_liq_vol.csv/.json`.

**Late-horizon convergence failure (the headline mispricing result).** Day-level
regression (day FE + repetition, cluster by group): gamified main effect ≈ 0 on days 1–10;
gamified × late (days 11–15): AMR +0.86 (**p = 0.040**), RAD +0.23 (p = 0.084). Under RAD,
ng converges to 0.06–0.08 by days 13–15 while gh/ghp rise and gp plateaus. Frame as
"failure to converge", not "larger bubbles" — level tests are hopeless (RAD group-level
MW p ≈ 0.6; gamified 0.283 vs ng 0.216).

**Literacy redistribution.** Within gamified markets, fin-quiz score predicts within-market
relative final wealth (reconstructed from the trade stream): Spearman ρ = 0.29, p = 0.005
(n = 96 trader-markets); monotone across literacy bins (≤50%: −710 E$; >90%: +182 E$).
In ng: ρ = 0.14, p = 0.42 (n = 36). **Caution**: the literacy × gamified interaction is
NOT significant (p = 0.66) — state as "significant gradient in gamified, cannot yet show
it differs from control". Figure: `literacy_relwealth.png`.

### Tier 3 — descriptive / conditional

**Volume ↔ bubble size within ghp.** Market-rep Spearman(volume, RAD) = 0.78 (p = 0.007,
n = 10); group level ρ = 0.70 (p = 0.19, n = 5). Sign flips negative in ng and gh. Purely
cross-market: within-market day-level co-movement ≈ 0, and volume/liquidity do **not**
diverge late (gamified × late on n_trades: p = 0.91; on spreads: n.s.) even as prices
detach. Only midquote RV shows suggestive late divergence (ng falls 0.53→0.46, ghp rises
0.24→0.33; interaction p = 0.11). The microstructure machinery keeps functioning; prices
just go wrong → "liquidity without price discovery".

**Signed direction (RD).** Hedonic arms are overpriced (gh +0.27, ghp +0.11), gp
underpriced (−0.18), ng unbiased (−0.04). Consistent with hedonic features fueling the
long side.

**Experience.** RAD falls rep1 → rep2 in every gamified arm (gh 0.41→0.18, gp 0.28→0.21,
ghp 0.33→0.25) but not ng (0.21→0.22, already converged). Opposite sign to H1a as stated
(experience helps *more* in gamified markets, from a higher base). Rep2 coefficient in the
day-level regression: −0.084 (p = 0.10).

### Nulls (do not chase with current data)

- **Gini / inequality levels**: reconstructed final-wealth Gini ng 0.062 vs gamified 0.063,
  p ≈ 0.8 at every level; gamified × late null. Gini tracks *mispricing* (ρ = 0.44,
  p = 0.04 pooled), not treatment. Inequality runs through the bubble channel.
- **Order-placement composition**: share of traders quoting both sides, bid/ask balance,
  cancel ratios — all similar across arms. The noise shows in intraday round-tripping and
  price informativeness, not order-mix.
- **gh vs ng**: 2 vs 3 groups; best achievable two-sided MW p = 0.20. Mathematically
  untestable until the gh cell grows. gh is also internally extreme: one group is the
  biggest bubble in the sample (RAD 0.55), the other the calmest market (0.04).

## 3. Where to focus next

1. **Recruitment allocation** (decided with PI, ~40 students ≈ 6 groups): **3 ng + 3 ghp**.
   Power simulation (block bootstrap of observed rep pairs, one-sided MW α = 0.05 on RAD):
   this allocation gives ~29% power for ghp vs ng and ~15% pooled — the best available;
   gh-heavy allocations give ≤5% because gh's observed distribution straddles ng.
   Realistic goal: ghp vs ng into p < 0.10 one-sided; sharpen the late-horizon result.
2. **Wild cluster bootstrap** the two headline regressions (gamified × late on AMR/RAD;
   volume) before quoting p-values.
3. **Late-horizon RAD as a pre-declared endpoint** (days 11–15 mean RAD): separates far
   better than full-horizon (ng 0.099 vs ghp 0.320 at rep level, p = 0.13 one-sided
   already) and is theoretically motivated.
4. **Mechanism table**: volume → churn → liquidity → convergence failure → literacy
   redistribution. Each link has a test; assemble as the paper's arc.
5. **Trader-type / composition confound**: the volume–bubble correlation within ghp could
   reflect group composition (share_speculator, overconfidence). Worth one regression of
   market-rep RAD on volume + composition controls.
6. **Not worth effort now**: Gini levels, gh-specific claims, gp anything (1 group),
   day-level volume-mispricing dynamics (absent within-market).

## 4. Data quality notes

- `aggressor_side` is empty in MBO trade records → buy/sell-initiated imbalance not
  computable; use trader-panel n_buys/n_sells instead.
- Panel `gini`/`wealth_day` build on the unreliable `player.num_shares` snapshot; use
  `recon_final_wealth.csv` (trade-stream reconstruction) for wealth outcomes.
- MBP1 has ready-made `spread`/`midpoint`; ~14–16% of book time is one-sided (excluded
  from spread averages; share itself doesn't differ by arm).
- Raw `trading_day` in MBO/MBP1 aligns 1:1 with panel days for kept markets (verified by
  matching trade counts); non-panel `trading_session_uuid`s are training/dropped markets.
- gh absolute quoted spread (134 E$) is inflated by the 1,000-price market in
  `20260520_AM/gh1`; use relative measures.

## 5. File map (all in `src/explore/`)

| File | Contents |
|---|---|
| `amr_by_treatment.py` / `.json` | AMR/RAD/RD by treatment: cells, MW tests, rep splits, day paths, per-market detail |
| `tick_liquidity_volatility.py` / `tick_liq_vol.csv` / `.json` | Market-day tick panel: spreads, depth, RV, churn + ghp-vs-ng tests |
| `recon_final_wealth.csv` | Trade-stream-reconstructed final wealth per trader-market |
| `literacy_relwealth.png` | Literacy vs within-market relative wealth figure |
| `preliminary_hypothesis_analysis.py` / `prelim_results.json` | Earlier full H1–H6 pass (note: currently restricted to session 20260826 via `KEEP_SESSIONS`) |
| `HANDOFF_REPORT.md` | This file |

Main pipeline: `make panels` rebuilds processed CSVs; `make analyze` runs
`src/analyze/hypothesis_tests.R` (H1 regressions there don't yet include the
gamified × late specification — worth porting).
