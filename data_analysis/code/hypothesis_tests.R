# hypothesis_tests.R
# Does Trading Gamification Fuel Bubbles?
# Chapkovski, Goswami, Işık, Zoican (2026)
# Date created: 04-08-2026
# Date last modified: 04-08-2026
# Last modified by: deb
# ============================================================
# Sources:
#   market_day_panel.csv  → market × trading-day (mkt_day)
#   trader_day_panel.csv  → trader × market × trading-day (trader_day)
# ============================================================
library(conflicted)
library(tidyverse)
library(lfe)
library(stargazer)
library(fixest)
library(modelsummary)
library(rstudioapi)
library(ggplot2)
library(ggfixest)
library(cowplot)
library(latex2exp)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
setwd(dirname(getActiveDocumentContext()$path))

mkt_day    <- read.csv("../processed_data/market_day_panel.csv")
trader_day <- read.csv("../processed_data/trader_day_panel.csv")

# ── SAVE RAW COPIES BEFORE STANDARDISING ──────────────────────────────────────
trader_day <- trader_day %>%
  mutate(
    fin_quiz_score_raw  = fin_quiz_score,
    self_assessment_raw = self_assessment,
    age_raw             = age,
    overconfidence_raw  = overconfidence
  )

# ── STANDARDISE CONTINUOUS CONTROLS ───────────────────────────────────────────

trader_day <- trader_day %>%
  mutate(across(
    c(fin_quiz_score, self_assessment, age, overconfidence),
    ~ (. - mean(., na.rm = TRUE)) / sd(., na.rm = TRUE)
  ))

# ── CONSTRUCT DERIVED VARIABLES ───────────────────────────────────────────────

# --- trader × day ---
trader_day <- trader_day %>%
  mutate(
    is_rep2           = as.integer(repetition == 2),
    gamified_x_hybrid = gamified * hybrid,
    price_change      = closing_price - price_L1,     # ΔP_k = P_k − P_{k-1}
    forecast_error    = forecast - closing_price,      # F_{i,k+1} − P_k
    pc_x_gamified     = price_change * gamified,
    pc_x_hybrid       = price_change * hybrid,
    pc_x_gam_x_hyb   = price_change * gamified_x_hybrid
  )

# --- market × day ---
mkt_day <- mkt_day %>%
  mutate(
    is_rep2           = as.integer(repetition == 2),
    gamified_x_hybrid = gamified * hybrid,
    gamified_x_rep2   = gamified * is_rep2
  )

# ── AUXILIARY TABLES ────────────────────────────────────────────────────

mkt_demog <- trader_day %>%
  group_by(market_uuid) %>%
  summarise(
    fin_quiz_score     = mean(fin_quiz_score,     na.rm = TRUE),
    self_assessment    = mean(self_assessment,    na.rm = TRUE),
    age                = mean(age,                na.rm = TRUE),
    overconfidence     = mean(overconfidence,     na.rm = TRUE),
    trading_experience = mean(trading_experience, na.rm = TRUE),
    .groups = "drop"
  )


mkt_day <- mkt_day %>%
  left_join(mkt_demog, by = "market_uuid")

mkt_payoff <- trader_day %>%
  select(market_uuid, participant_code, trade_payoff) %>%
  distinct() %>%
  group_by(market_uuid) %>%
  summarise(sd_payoff = sd(trade_payoff, na.rm = TRUE), .groups = "drop")

# ── MARKET-LEVEL COLLAPSED PANEL ────────────────────────────────────────


mkt <- mkt_day %>%
  group_by(market_uuid) %>%
  summarise(
    gamified             = first(gamified),
    hybrid               = first(hybrid),
    algo_present         = first(algo_present),
    repetition           = first(repetition),
    is_rep2              = first(is_rep2),
    gamified_x_hybrid    = first(gamified_x_hybrid),
    gamified_x_rep2      = first(gamified_x_rep2),
    n_bubble_runs        = sum(bubble_start,  na.rm = TRUE),
    n_surge_crash        = sum(surge, na.rm = TRUE) + sum(crash, na.rm = TRUE),
    avg_mispricing       = mean(avg_mispricing,       na.rm = TRUE),
    avg_abs_mispricing   = mean(avg_abs_mispricing,   na.rm = TRUE),
    abs_mispricing_ratio = mean(abs_mispricing_ratio, na.rm = TRUE),
    gini                 = mean(gini,                 na.rm = TRUE),
    share_feedback       = first(share_feedback),
    share_speculator     = first(share_speculator),
    share_fundamental    = first(share_fundamental),
    share_other          = first(share_other),
    fin_quiz_score       = first(fin_quiz_score),
    self_assessment      = first(self_assessment),
    age                  = first(age),
    overconfidence       = first(overconfidence),
    trading_experience   = first(trading_experience),
    .groups = "drop"
  ) %>%
  left_join(mkt_payoff, by = "market_uuid")

# ── CONTROLS & LABEL DICT ─────────────────────────────────────────────────────

controls_str <- "fin_quiz_score + self_assessment + age + overconfidence + trading_experience + is_rep2"

setFixest_dict(c(
  gamified              = "Gamified",
  hybrid                = "Hybrid (algo possible)",
  gamified_x_hybrid     = "Gamified $\\times$ Hybrid",
  gamified_x_rep2       = "Gamified $\\times$ Repetition 2",
  is_rep2               = "Repetition 2",
  price_change          = "Price change $\\Delta P_k$",
  pc_x_gamified         = "$\\Delta P_k$ $\\times$ Gamified",
  pc_x_hybrid           = "$\\Delta P_k$ $\\times$ Hybrid",
  pc_x_gam_x_hyb       = "$\\Delta P_k$ $\\times$ Gamified $\\times$ Hybrid",
  avg_mispricing        = "Avg. mispricing",
  avg_abs_mispricing    = "Abs. mispricing",
  abs_mispricing_ratio  = "Abs. mispricing ratio",
  n_bubble_runs         = "Bubble runs",
  n_surge_crash         = "Price surges/crashes",
  gini                  = "Gini coefficient",
  sd_payoff             = "SD of payoffs",
  share_feedback        = "Share feedback traders",
  share_fundamental     = "Share fundamental traders",
  forecast_error        = "Forecast error $F_{i,k+1} - P_k$",
  fin_quiz_score        = "Financial literacy",
  self_assessment       = "Self-assessed literacy",
  overconfidence        = "Overconfidence",
  age                   = "Age",
  trading_experience    = "Trading experience",
  market_uuid           = "Market",
  participant_code      = "Participant"
))


# ============================================================
# Table 1: H1 — Mispricing
# Panel: mkt_day (market × trading-day)
# Prediction: gamified > 0
# ============================================================

h1_1 <- feols(
  as.formula(paste(
    "avg_mispricing ~ gamified + hybrid + gamified_x_hybrid +",
    controls_str, "| market_uuid + trading_day"
  )),
  data = mkt_day, cluster = ~market_uuid
)
h1_2 <- feols(
  as.formula(paste(
    "avg_abs_mispricing ~ gamified + hybrid + gamified_x_hybrid +",
    controls_str, "| market_uuid + trading_day"
  )),
  data = mkt_day, cluster = ~market_uuid
)
h1_3 <- feols(
  as.formula(paste(
    "abs_mispricing_ratio ~ gamified + hybrid + gamified_x_hybrid +",
    controls_str, "| market_uuid + trading_day"
  )),
  data = mkt_day, cluster = ~market_uuid
)
h1_4 <- feols(
  "avg_mispricing ~ gamified + hybrid + gamified_x_hybrid | market_uuid + trading_day",
  data = mkt_day, cluster = ~market_uuid
)
h1_5 <- feols(
  "avg_abs_mispricing ~ gamified + hybrid + gamified_x_hybrid | market_uuid + trading_day",
  data = mkt_day, cluster = ~market_uuid
)
h1_6 <- feols(
  "abs_mispricing_ratio ~ gamified + hybrid + gamified_x_hybrid | market_uuid + trading_day",
  data = mkt_day, cluster = ~market_uuid
)

h1_tex <- etable(
  h1_1, h1_2, h1_3, h1_4, h1_5, h1_6,
  title = "Gamification and Asset Price Mispricing",
  tex = TRUE, digits = "r2", digits.stats = "r2", depvar = TRUE,
  order = c("Gamified$", "Hybrid", "Gamified.*Hybrid",
            "Financial literacy", "Self-assessed", "Overconfidence",
            "Age", "Trading experience", "Repetition 2"),
  headers = list("With controls" = 3, "Without controls" = 3),
  fitstat = c("n", "r2")
)
writeLines(h1_tex, "../tables/h1_mispricing.tex")


# ============================================================
# Table 2: H2 — Volatility
# Panel: mkt
# Prediction: gamified > 0
# ============================================================

h2_1 <- feols(
  as.formula(paste(
    "n_bubble_runs ~ gamified + hybrid + gamified_x_hybrid +", controls_str
  )),
  data = mkt, cluster = ~market_uuid
)
h2_2 <- feols(
  as.formula(paste(
    "n_surge_crash ~ gamified + hybrid + gamified_x_hybrid +", controls_str
  )),
  data = mkt, cluster = ~market_uuid
)
h2_3 <- feols(
  "n_bubble_runs ~ gamified + hybrid + gamified_x_hybrid",
  data = mkt, cluster = ~market_uuid
)
h2_4 <- feols(
  "n_surge_crash ~ gamified + hybrid + gamified_x_hybrid",
  data = mkt, cluster = ~market_uuid
)

h2_tex <- etable(
  h2_1, h2_2, h2_3, h2_4,
  title = "Gamification and Price Volatility",
  tex = TRUE, digits = "r2", digits.stats = "r2", depvar = TRUE,
  order = c("Gamified$", "Hybrid", "Gamified.*Hybrid",
            "Financial literacy", "Self-assessed", "Overconfidence",
            "Age", "Trading experience", "Repetition 2"),
  headers = list("With controls" = 2, "Without controls" = 2),
  fitstat = c("n", "r2")
)
writeLines(h2_tex, "../tables/h2_volatility.tex")


# ============================================================
# Table 3: H3 — Wealth Inequality
# Panel: mkt
# Prediction: gamified > 0
# ============================================================

h3_1 <- feols(
  as.formula(paste(
    "gini ~ gamified + hybrid + gamified_x_hybrid +", controls_str
  )),
  data = mkt, cluster = ~market_uuid
)
h3_2 <- feols(
  as.formula(paste(
    "sd_payoff ~ gamified + hybrid + gamified_x_hybrid +", controls_str
  )),
  data = mkt, cluster = ~market_uuid
)
h3_3 <- feols(
  "gini ~ gamified + hybrid + gamified_x_hybrid",
  data = mkt, cluster = ~market_uuid
)
h3_4 <- feols(
  "sd_payoff ~ gamified + hybrid + gamified_x_hybrid",
  data = mkt, cluster = ~market_uuid
)

h3_tex <- etable(
  h3_1, h3_2, h3_3, h3_4,
  title = "Gamification and Wealth Inequality",
  tex = TRUE, digits = "r2", digits.stats = "r2", depvar = TRUE,
  order = c("Gamified$", "Hybrid", "Gamified.*Hybrid",
            "Financial literacy", "Self-assessed", "Overconfidence",
            "Age", "Trading experience", "Repetition 2"),
  headers = list("With controls" = 2, "Without controls" = 2),
  fitstat = c("n", "r2")
)
writeLines(h3_tex, "../tables/h3_inequality.tex")


# ============================================================
# Table 4: H4 — Experience
# Panel: mkt_day (market × trading-day)
# FE: market_uuid + trading_day
# Prediction: gamified:is_rep2 < 0
# ============================================================

h4_1 <- feols(
  as.formula(paste(
    "avg_abs_mispricing ~ gamified * is_rep2 + hybrid +",
    controls_str, "| market_uuid + trading_day"
  )),
  data = mkt_day, cluster = ~market_uuid
)
h4_2 <- feols(
  as.formula(paste(
    "avg_mispricing ~ gamified * is_rep2 + hybrid +",
    controls_str, "| market_uuid + trading_day"
  )),
  data = mkt_day, cluster = ~market_uuid
)
h4_3 <- feols(
  "avg_abs_mispricing ~ gamified * is_rep2 + hybrid | market_uuid + trading_day",
  data = mkt_day, cluster = ~market_uuid
)
h4_4 <- feols(
  "avg_mispricing ~ gamified * is_rep2 + hybrid | market_uuid + trading_day",
  data = mkt_day, cluster = ~market_uuid
)

h4_tex <- etable(
  h4_1, h4_2, h4_3, h4_4,
  title = "Gamification, Experience, and Mispricing",
  tex = TRUE, digits = "r2", digits.stats = "r2", depvar = TRUE,
  order = c("Gamified$", "Repetition 2", "Gamified.*Repetition 2", "Hybrid",
            "Financial literacy", "Self-assessed", "Overconfidence",
            "Age", "Trading experience"),
  headers = list("With controls" = 2, "Without controls" = 2),
  fitstat = c("n", "r2")
)
writeLines(h4_tex, "../tables/h4_experience.tex")


# ============================================================
# Table 5: H5 — Trader Types
# Panel: mkt
# Prediction: share_feedback higher, share_fundamental lower
# ============================================================

h5_1 <- feols(
  as.formula(paste(
    "share_feedback ~ gamified + hybrid + gamified_x_hybrid +", controls_str
  )),
  data = mkt, cluster = ~market_uuid
)
h5_2 <- feols(
  as.formula(paste(
    "share_fundamental ~ gamified + hybrid + gamified_x_hybrid +", controls_str
  )),
  data = mkt, cluster = ~market_uuid
)
h5_3 <- feols(
  "share_feedback ~ gamified + hybrid + gamified_x_hybrid",
  data = mkt, cluster = ~market_uuid
)
h5_4 <- feols(
  "share_fundamental ~ gamified + hybrid + gamified_x_hybrid",
  data = mkt, cluster = ~market_uuid
)

h5_tex <- etable(
  h5_1, h5_2, h5_3, h5_4,
  title = "Gamification and Trader Types",
  tex = TRUE, digits = "r2", digits.stats = "r2", depvar = TRUE,
  order = c("Gamified$", "Hybrid", "Gamified.*Hybrid",
            "Financial literacy", "Self-assessed", "Overconfidence",
            "Age", "Trading experience", "Repetition 2"),
  headers = list("With controls" = 2, "Without controls" = 2),
  fitstat = c("n", "r2")
)
writeLines(h5_tex, "../tables/h5_trader_types.tex")


# ============================================================
# Table 6: H6 — Algorithmic Trading Beliefs
# Panel: mkt
# test: gamified_x_hybrid < 0
# ============================================================

h6_1 <- feols(
  as.formula(paste(
    "avg_abs_mispricing ~ gamified + hybrid + gamified_x_hybrid +", controls_str
  )),
  data = mkt, cluster = ~market_uuid
)
h6_2 <- feols(
  as.formula(paste(
    "n_bubble_runs ~ gamified + hybrid + gamified_x_hybrid +", controls_str
  )),
  data = mkt, cluster = ~market_uuid
)
h6_3 <- feols(
  as.formula(paste(
    "gini ~ gamified + hybrid + gamified_x_hybrid +", controls_str
  )),
  data = mkt, cluster = ~market_uuid
)
h6_4 <- feols(
  "avg_abs_mispricing ~ gamified + hybrid + gamified_x_hybrid",
  data = mkt, cluster = ~market_uuid
)
h6_5 <- feols(
  "n_bubble_runs ~ gamified + hybrid + gamified_x_hybrid",
  data = mkt, cluster = ~market_uuid
)
h6_6 <- feols(
  "gini ~ gamified + hybrid + gamified_x_hybrid",
  data = mkt, cluster = ~market_uuid
)

h6_tex <- etable(
  h6_1, h6_2, h6_3, h6_4, h6_5, h6_6,
  title = "Algorithmic Trading Beliefs and Gamification Effects",
  tex = TRUE, digits = "r2", digits.stats = "r2", depvar = TRUE,
  order = c("Gamified$", "Hybrid", "Gamified.*Hybrid",
            "Financial literacy", "Self-assessed", "Overconfidence",
            "Age", "Trading experience", "Repetition 2"),
  headers = list("With controls" = 3, "Without controls" = 3),
  fitstat = c("n", "r2")
)
writeLines(h6_tex, "../tables/h6_algo_beliefs.tex")


# ============================================================
# Table 7: H7 — Forecast Extrapolation (Eq. 4)
# Panel: trader_day (trader × trading-day)
# FE: participant_code + trading_day
# Cluster: market_uuid
# Prediction: pc_x_gamified (γ₀) > 0 ; pc_x_gam_x_hyb (γ₂) < 0
# ============================================================

h7_1 <- feols(
  as.formula(paste(
    "forecast_error ~ price_change + gamified + hybrid +",
    "pc_x_gamified + pc_x_hybrid + pc_x_gam_x_hyb +",
    controls_str,
    "| participant_code + trading_day"
  )),
  data = trader_day, cluster = ~market_uuid
)
h7_2 <- feols(
  as.formula(paste(
    "forecast_error ~ price_change + gamified + hybrid +",
    "pc_x_gamified + pc_x_hybrid + pc_x_gam_x_hyb",
    "| participant_code + trading_day"
  )),
  data = trader_day, cluster = ~market_uuid
)
# Pooled OLS for robustness (no FE)
h7_3 <- feols(
  as.formula(paste(
    "forecast_error ~ price_change + gamified + hybrid +",
    "pc_x_gamified + pc_x_hybrid + pc_x_gam_x_hyb +",
    controls_str
  )),
  data = trader_day, cluster = ~market_uuid
)
h7_4 <- feols(
  "forecast_error ~ price_change + gamified + hybrid + pc_x_gamified + pc_x_hybrid + pc_x_gam_x_hyb",
  data = trader_day, cluster = ~market_uuid
)

h7_tex <- etable(
  h7_1, h7_2, h7_3, h7_4,
  title = "Gamification and Trend Extrapolation in Price Forecasts",
  tex = TRUE, digits = "r2", digits.stats = "r2", depvar = TRUE,
  order = c(
    "Price change\\b", "Gamified$", "Hybrid",
    "\\$\\\\Delta P_k\\$.*Gamified$",
    "\\$\\\\Delta P_k\\$.*Hybrid$",
    "\\$\\\\Delta P_k\\$.*Gamified.*Hybrid",
    "Financial literacy", "Self-assessed", "Overconfidence",
    "Age", "Trading experience", "Repetition 2"
  ),
  headers = list("Trader + period FE" = 2, "Pooled OLS" = 2),
  fitstat = c("n", "r2")
)
writeLines(h7_tex, "../tables/h7_forecasts.tex")


# ============================================================
# Summary Statistics
# ============================================================

sum_mkt <- mkt %>%
  select(
    avg_mispricing, avg_abs_mispricing, abs_mispricing_ratio,
    n_bubble_runs, n_surge_crash,
    gini, sd_payoff,
    share_feedback, share_fundamental,
    gamified, hybrid, is_rep2
  )

# FIX 3: removed distinct() — forecast_error and price_change vary by
# trading period so distinct() would not deduplicate to participant level
sum_trader <- trader_day %>%
  select(
    forecast_error, price_change,
    fin_quiz_score_raw, self_assessment_raw,
    overconfidence_raw, age_raw, trading_experience
  )

stargazer(
  as.data.frame(sum_mkt),
  type = "latex",
  title = "Summary Statistics — Market Level",
  digits = 2,
  summary.stat = c("mean", "sd", "p25", "median", "p75", "min", "max"),
  covariate.labels = c(
    "Avg. mispricing", "Abs. mispricing", "Abs. mispricing ratio",
    "Bubble runs", "Price surges/crashes",
    "Gini coefficient", "SD of payoffs",
    "Share feedback traders", "Share fundamental traders",
    "Gamified", "Hybrid", "Repetition 2"
  ),
  out = "../tables/summary_stats_market.tex"
)

stargazer(
  as.data.frame(sum_trader),
  type = "latex",
  title = "Summary Statistics — Trader Level",
  digits = 2,
  summary.stat = c("mean", "sd", "p25", "median", "p75", "min", "max"),
  covariate.labels = c(
    "Forecast error", "Price change $\\Delta P_k$",
    "Financial literacy", "Self-assessed literacy",
    "Overconfidence", "Age", "Trading experience"
  ),
  out = "../tables/summary_stats_trader.tex"
)
