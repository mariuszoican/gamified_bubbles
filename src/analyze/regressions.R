# regressions.R
# Does Trading Gamification Fuel Bubbles?
# Chapkovski, Goswami, Işık, Zoican (2026)
# Date created: 03-09-2026
# Date last modified: 04-09-2026
# ============================================================
# Regression tables mirroring the figures (src/analyze/figures.py):
#   Table 1  fig1  Mispricing (market x day)
#   Table 2  fig2  Liquidity: spreads, price impact, depth, volatility
#   Table 3  fig3  Volume, order flow, order composition, churn
#   Table 4  fig5  Liquidity provision mechanism
#   Table 5  fig6  Volume share by trader type (market x day)
#   Table 5b fig6  Profits by trader type
#   Table 6  fig7  Forecast accuracy and bias
#   Table 7  fig8  Bubble incidence (market-rep counts)
#   Table 8  ---   Error correction (gap x gamified; see error_correction.py)
#
# Sample (as in the figures): GHP vs NG market-reps only; outlier groups
# 20260520_PM/ng1 and 20280904/ghp1 excluded. Day-level panels wherever the outcome varies by
# day; market-rep collapse only for count/composition outcomes.
#
# Specifications: no market-average composition controls (following
# Asparouhova et al. 2024). Fixed effects:
#   trading_day  -- within-market day index (absorbs the deterministic
#                   fundamental path v_t and late-day mechanics)
#   repetition   -- market repetition 1 vs 2 (experience)
# Columns move from sparse FE to the full FE set.
#
# Intercept convention: FE and other controls are expanded as dummies /
# covariates and then centered at their NG (gamified == 0) means. The
# slope on gamified is the same as absorbed FE; the constant is the
# unconditional non-gamified mean of the outcome.
#
# SEs in parentheses: heteroskedasticity-robust HC1 (White), as in
# Asparouhova et al. (2024, RoF). No clustering. Stars from HC1 p-values.
#
# Sources:  data/processed/market_day_panel_full.csv
#           data/processed/trader_day_panel_full.csv
# Outputs:  output/tables/t1_mispricing.tex ... t8_error_correction.tex
# ============================================================
library(conflicted)
library(tidyverse)
library(fixest)
conflicts_prefer(dplyr::filter, dplyr::first, dplyr::lag)

# ── REPO ROOT ─────────────────────────────────────────────────────────────────
.resolve_root <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(file.path(dirname(sub("^--file=", "", file_arg)), "../..")))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable() &&
      !is.null(rstudioapi::getActiveDocumentContext()$path) &&
      nzchar(rstudioapi::getActiveDocumentContext()$path)) {
    return(normalizePath(file.path(
      dirname(rstudioapi::getActiveDocumentContext()$path), "../.."
    )))
  }
  if (file.exists("data/processed/market_day_panel_full.csv")) {
    return(normalizePath("."))
  }
  if (file.exists("../../data/processed/market_day_panel_full.csv")) {
    return(normalizePath("../.."))
  }
  stop("Cannot locate repo root; run from gamified_bubbles_analysis/ or via Rscript.")
}

ROOT      <- .resolve_root()
PROCESSED <- file.path(ROOT, "data", "processed")
TABLES    <- file.path(ROOT, "output", "tables")
dir.create(TABLES, recursive = TRUE, showWarnings = FALSE)

# ── LOAD & RESTRICT SAMPLE (GHP vs NG, outlier group excluded) ────────────────
EXCLUDE_GROUPS <- c("20260520_PM/ng1", "20280904/ghp1")

mkt_day <- read.csv(file.path(PROCESSED, "market_day_panel_full.csv")) %>%
  filter(treatment %in% c("ng", "ghp"), !(group_label %in% EXCLUDE_GROUPS))

trader_day <- read.csv(file.path(PROCESSED, "trader_day_panel_full.csv")) %>%
  filter(treatment %in% c("ng", "ghp"), !(group_label %in% EXCLUDE_GROUPS))

# ── DERIVED VARIABLES ─────────────────────────────────────────────────────────
# In this two-arm sample, gamified == 1{treatment == "ghp"}.
mkt_day <- mkt_day %>%
  mutate(late = as.integer(trading_day >= 11))       # days 11-15

trader_day <- trader_day %>%
  mutate(late = as.integer(trading_day >= 11))

# ── MARKET-REP COLLAPSED PANEL (counts & composition) ─────────────────────────
mkt <- mkt_day %>%
  group_by(market_uuid, group_label, session_id) %>%
  summarise(
    gamified           = first(gamified),
    repetition         = first(repetition),
    n_trades           = sum(n_trades_market, na.rm = TRUE),
    n_bubble_days      = sum(bubble_period,   na.rm = TRUE),
    n_bubble_runs      = sum(bubble_start,    na.rm = TRUE),
    n_surges           = sum(surge,           na.rm = TRUE),
    n_crashes          = sum(crash,           na.rm = TRUE),
    share_market_maker = first(share_market_maker),
    share_feedback     = first(share_feedback),
    share_speculator   = first(share_speculator),
    share_fundamental  = first(share_fundamental),
    share_other        = first(share_other),
    .groups = "drop"
  )

# ── TRADER CROSS-SECTION (final relative wealth, day 15) ──────────────────────
trader_final <- trader_day %>%
  filter(trading_day == 15)

# ── NG-CENTERED FE (intercept = unconditional NG mean) ────────────────────────
# `| fe` absorbs the intercept. `i(fe)` makes it the NG mean in the omitted
# cell (rep 1, day 1). Centering FE dummies and other controls at their NG
# means leaves treatment slopes unchanged and sets the constant equal to
# the unconditional non-gamified mean.
prep_ng_intercept <- function(df, factors = NULL, numerics = NULL, on = "ng") {
  if (!"gamified" %in% names(df)) stop("gamified missing")
  idx <- if (identical(on, "ng")) df$gamified == 0 else rep(TRUE, nrow(df))
  if (!any(idx)) stop("no rows to center on")

  for (v in numerics) {
    if (!v %in% names(df)) next
    df[[v]] <- df[[v]] - mean(df[[v]][idx], na.rm = TRUE)
  }

  for (v in factors) {
    if (!v %in% names(df)) next
    f <- factor(df[[v]])
    mm <- model.matrix(~ f)[, -1, drop = FALSE]
    for (j in seq_len(ncol(mm))) {
      mm[, j] <- as.numeric(mm[, j]) - mean(mm[idx, j], na.rm = TRUE)
    }
    colnames(mm) <- paste0("d_", v, "_", seq_len(ncol(mm)))
    already <- grep(paste0("^d_", v, "_"), names(df), value = TRUE)
    if (length(already)) df[already] <- NULL
    df <- cbind(df, mm)
  }
  df
}

rhs_fe <- function(df, factors) {
  cols <- unlist(lapply(factors, function(v) {
    grep(paste0("^d_", v, "_"), names(df), value = TRUE)
  }), use.names = FALSE)
  if (!length(cols)) return("1")
  paste(cols, collapse = " + ")
}

# OLS with NG-centered FE on the RHS (so etable prints a constant).
# Re-center on complete cases of the outcome so the intercept equals
# the NG mean in the estimation sample (not the full-panel NG mean).
# SEs: HC1 (fixest vcov = "HC1"); stars from the same VCOV.
.NG_NUMERICS <- c("late", "fundamental_gap", "order_flow_imbalance")
g <- function(extra = "gamified") extra

ols <- function(lhs, rhs, data, fe = NULL) {
  dat <- data[!is.na(data[[lhs]]), , drop = FALSE]
  nums <- intersect(.NG_NUMERICS, names(dat))
  on <- if (any(dat$gamified == 0, na.rm = TRUE)) "ng" else "sample"
  dat <- prep_ng_intercept(dat, factors = fe, numerics = nums, on = on)
  extra <- if (length(fe)) paste("+", rhs_fe(dat, fe)) else ""
  feols(as.formula(paste(lhs, "~", rhs, extra)),
        data = dat, vcov = "HC1")
}

mkt_day <- prep_ng_intercept(
  mkt_day,
  factors  = c("repetition", "trading_day"),
  numerics = c("late", "fundamental_gap", "order_flow_imbalance")
)
mkt <- prep_ng_intercept(
  mkt,
  factors  = "repetition"
)
trader_day <- prep_ng_intercept(
  trader_day,
  factors  = c("repetition", "trading_day"),
  numerics = "late"
)
trader_final <- prep_ng_intercept(
  trader_final,
  factors  = "repetition"
)

# ── LABEL DICT ────────────────────────────────────────────────────────────────
setFixest_dict(c(
  gamified                  = "Gamified",
  late                      = "Late window (days 11--15)",
  "gamified:late"           = "Gamified $\\times$ Late",
  fundamental_gap           = "Gap",
  "fundamental_gap:gamified" = "Gap $\\times$ Gamified",
  "gamified:fundamental_gap" = "Gap $\\times$ Gamified",
  order_flow_imbalance      = "OFI",
  abs_order_flow_imbalance  = "$|$OFI$|$",
  "order_flow_imbalance:gamified" = "OFI $\\times$ Gamified",
  "gamified:order_flow_imbalance" = "OFI $\\times$ Gamified",
  payoff_mm                 = "Payoff market makers",
  payoff_fundamental        = "Payoff fundamentalists",
  payoff_feedback           = "Payoff feedback",
  payoff_speculator         = "Payoff speculators",
  payoff_other              = "Payoff unclassified",
  # dependent variables
  avg_abs_mispricing        = "Abs. mispricing",
  abs_mispricing_ratio      = "AMR",
  rad                       = "RAD",
  ret_next                  = "$\\Delta \\log P_{t+1}$",
  rel_quoted_spread         = "Quoted spread",
  rel_eff_spread            = "Effective spread",
  rel_realized_spread       = "Realized spread",
  rel_price_impact          = "Price impact",
  depth_best                = "Depth at best",
  rv_mid                    = "Midquote volatility",
  n_trades_market           = "Trades",
  n_limit_orders            = "Limit orders submitted",
  share_limit_orders        = "Share limit orders",
  cancel_to_order           = "Cancel-to-order",
  churn                     = "Intraday churn",
  gini                      = "Gini",
  rel_wealth                = "Relative wealth",
  n_improving_adds          = "Spread-improving orders",
  share_improving_adds      = "Share improving",
  time_to_same_side_order_s = "Order replenishment (s)",
  time_to_next_order_s      = "Next order (s)",
  spread_recovery_s         = "Spread recovery (s)",
  forecast_err_price        = "Forecast error (price)",
  forecast_err_fund         = "Forecast error (fund.)",
  forecast_bias_fund        = "Forecast bias (fund.)",
  n_trades                  = "Trades",
  n_bubble_days             = "Bubble days",
  n_bubble_runs             = "Bubble episodes",
  n_surges                  = "Surges",
  n_crashes                 = "Crashes",
  share_market_maker        = "Share market makers",
  share_feedback            = "Share feedback",
  share_speculator          = "Share speculators",
  share_fundamental         = "Share fundamentalists",
  share_other               = "Share unclassified",
  share_vol_market_maker    = "Market makers",
  share_vol_fundamental     = "Fundamentalists",
  share_vol_feedback        = "Feedback",
  share_vol_speculator      = "Speculators",
  share_vol_other           = "Unclassified",
  vol_market_maker          = "Market makers",
  vol_fundamental           = "Fundamentalists",
  vol_feedback              = "Feedback",
  vol_speculator            = "Speculators",
  vol_other                 = "Unclassified",
  # controls / FE
  share_finance_course      = "Finance course share",
  avg_fin_quiz              = "Financial literacy",
  avg_age                   = "Age",
  share_female              = "Female share",
  share_trading_experience  = "Trading experience share",
  fin_quiz_score            = "Financial literacy",
  self_assessment           = "Self-assessed literacy",
  age                       = "Age",
  gender_female             = "Female",
  finance_course            = "Finance course",
  overconfidence            = "Overconfidence",
  trading_experience        = "Trading experience",
  trading_day               = "Trading day",
  repetition                = "Repetition",
  market_uuid               = "Market",
  group_label               = "Group",
  session_id                = "Session",
  participant_code          = "Participant"
))

ETABLE_OPTS <- list(
  tex = TRUE, digits = "r3", digits.stats = "r2", depvar = TRUE,
  fitstat = c("n", "r2"),
  notes = "Heteroskedasticity-robust (HC1) standard errors in parentheses."
)

write_table <- function(models, title, headers, file, order = NULL, drop = NULL, ...) {
  if (is.null(order)) {
    order <- c("Gamified$", "Late", "Gamified.*Late", "Gap", "OFI",
               "Market maker", "!Financial|Self|Age|Over|Trading|Repetition")
  }
  drop <- unique(c(drop, "^d_repetition_", "^d_trading_day_", "^d_session_id_"))
  tex <- do.call(etable, c(
    list(models, title = title, headers = headers, order = order, drop = drop),
    list(...),
    ETABLE_OPTS
  ))
  writeLines(tex, file.path(TABLES, file))
  message("wrote ", file)
}

# ============================================================
# Table 1 (fig 1): Mispricing — market x day
# Each measure: repetition dummy, then repetition + day dummies.
# FE dummies are NG-centered: constant = unconditional NG mean.
# No market FE: treatment is constant within market.
# ============================================================
t1_1 <- ols("abs_mispricing_ratio", g(), mkt_day, fe = "repetition")
t1_2 <- ols("abs_mispricing_ratio", g(), mkt_day, fe = c("repetition", "trading_day"))
t1_3 <- ols("rad",                   g(), mkt_day, fe = "repetition")
t1_4 <- ols("rad",                   g(), mkt_day, fe = c("repetition", "trading_day"))
t1_5 <- ols("avg_abs_mispricing",    g(), mkt_day, fe = "repetition")
t1_6 <- ols("avg_abs_mispricing",    g(), mkt_day, fe = c("repetition", "trading_day"))

write_table(
  list(t1_1, t1_2, t1_3, t1_4, t1_5, t1_6),
  title = "Gamification and Mispricing",
  headers = NULL,
  file = "t1_mispricing.tex",
  extralines = list(
    "_Repetition dummy"    = rep("Yes", 6),
    "_Trading-day dummies" = c("", "Yes", "", "Yes", "", "Yes")
  )
)

# ============================================================
# Table 2 (fig 2): Liquidity — market x day, full FE
# Spreads/impact relative to midpoint; effective = realized + impact
# ============================================================
t2_1 <- ols("rel_quoted_spread",   g(), mkt_day, fe = c("repetition", "trading_day"))
t2_2 <- ols("rel_eff_spread",      g(), mkt_day, fe = c("repetition", "trading_day"))
t2_3 <- ols("rel_realized_spread", g(), mkt_day, fe = c("repetition", "trading_day"))
t2_4 <- ols("rel_price_impact",    g(), mkt_day, fe = c("repetition", "trading_day"))
t2_5 <- ols("depth_best",          g(), mkt_day, fe = c("repetition", "trading_day"))
t2_6 <- ols("rv_mid",              g(), mkt_day, fe = c("repetition", "trading_day"))

write_table(
  list(t2_1, t2_2, t2_3, t2_4, t2_5, t2_6),
  title = "Gamification and Liquidity",
  headers = NULL,
  file = "t2_liquidity.tex",
  extralines = list(
    "_Repetition dummy"    = rep("Yes", 6),
    "_Trading-day dummies" = rep("Yes", 6)
  )
)

# ============================================================
# Table 3 (fig 3): Volume, order flow, and order composition
# ============================================================
t3_1 <- ols("n_trades_market",          g(), mkt_day, fe = c("repetition", "trading_day"))
t3_2 <- ols("abs_order_flow_imbalance", g(), mkt_day, fe = c("repetition", "trading_day"))
t3_3 <- ols("churn",                    g(), mkt_day, fe = c("repetition", "trading_day"))
t3_4 <- ols("n_limit_orders",           g(), mkt_day, fe = c("repetition", "trading_day"))
t3_5 <- ols("share_limit_orders",       g(), mkt_day, fe = c("repetition", "trading_day"))

write_table(
  list(t3_1, t3_2, t3_3, t3_4, t3_5),
  title = "Gamification, Trading Volume, and Order Flow",
  headers = NULL,
  file = "t3_volume_orderflow.tex",
  extralines = list(
    "_Repetition dummy"    = rep("Yes", 5),
    "_Trading-day dummies" = rep("Yes", 5)
  )
)

# ============================================================
# Table 4 (fig 5): Liquidity provision mechanism — market x day
# Each outcome: trading-day dummies, then trading-day + repetition.
# Replenishment = median seconds to the next same-side limit order.
# ============================================================
t4_1 <- ols("n_improving_adds",          g(), mkt_day, fe = "trading_day")
t4_2 <- ols("n_improving_adds",          g(), mkt_day, fe = c("trading_day", "repetition"))
t4_3 <- ols("time_to_same_side_order_s", g(), mkt_day, fe = "trading_day")
t4_4 <- ols("time_to_same_side_order_s", g(), mkt_day, fe = c("trading_day", "repetition"))
t4_5 <- ols("spread_recovery_s",         g(), mkt_day, fe = "trading_day")
t4_6 <- ols("spread_recovery_s",         g(), mkt_day, fe = c("trading_day", "repetition"))

write_table(
  list(t4_1, t4_2, t4_3, t4_4, t4_5, t4_6),
  title = "Gamification and Liquidity Provision",
  headers = NULL,
  file = "t4_liquidity_provision.tex",
  extralines = list(
    "_Repetition dummy"    = c("", "Yes", "", "Yes", "", "Yes"),
    "_Trading-day dummies" = rep("Yes", 6)
  )
)

# ============================================================
# Table 5 (fig 6): Volume by trader type — market x day
# All five mutually exclusive types. Each type: volume share, then
# gross trades (buys+sells). Single FE set (day + repetition).
# ============================================================
.t5_fe <- c("trading_day", "repetition")
t5_1 <- ols("share_vol_market_maker", g(), mkt_day, fe = .t5_fe)
t5_2 <- ols("vol_market_maker",       g(), mkt_day, fe = .t5_fe)
t5_3 <- ols("share_vol_fundamental",  g(), mkt_day, fe = .t5_fe)
t5_4 <- ols("vol_fundamental",        g(), mkt_day, fe = .t5_fe)
t5_5 <- ols("share_vol_feedback",     g(), mkt_day, fe = .t5_fe)
t5_6 <- ols("vol_feedback",           g(), mkt_day, fe = .t5_fe)
t5_7 <- ols("share_vol_speculator",   g(), mkt_day, fe = .t5_fe)
t5_8 <- ols("vol_speculator",         g(), mkt_day, fe = .t5_fe)
t5_9 <- ols("share_vol_other",        g(), mkt_day, fe = .t5_fe)
t5_10 <- ols("vol_other",             g(), mkt_day, fe = .t5_fe)

write_table(
  list(t5_1, t5_2, t5_3, t5_4, t5_5, t5_6, t5_7, t5_8, t5_9, t5_10),
  title = "Gamification and Trading Volume by Type",
  headers = rep(c("Share", "Trades"), 5),
  file = "t5_trader_types.tex",
  extralines = list(
    "_Repetition dummy"    = rep("Yes", 10),
    "_Trading-day dummies" = rep("Yes", 10)
  )
)

# ============================================================
# Table 5b (fig 6, panel C): Profits by trader type
# Same column / FE layout as Table 5. Outcome is day-15 relative wealth
# of traders of that type (fig 6C). Constant = NG mean payoff of the type.
# ============================================================
.type_payoff <- function(type, name) {
  d <- subset(trader_final, trader_type == type)
  d[[name]] <- d$rel_wealth
  d
}
mm   <- .type_payoff("market_maker", "payoff_mm")
fund <- .type_payoff("fundamental",  "payoff_fundamental")
fb   <- .type_payoff("feedback",     "payoff_feedback")
sp   <- .type_payoff("speculator",   "payoff_speculator")
oth  <- .type_payoff("other",        "payoff_other")

t5b_1 <- ols("payoff_mm",          g(), mm)
t5b_2 <- ols("payoff_mm",          g(), mm,   fe = "repetition")
t5b_3 <- ols("payoff_fundamental", g(), fund, fe = "repetition")
t5b_4 <- ols("payoff_feedback",    g(), fb,   fe = "repetition")
t5b_5 <- ols("payoff_speculator",  g(), sp,   fe = "repetition")
t5b_6 <- ols("payoff_other",       g(), oth,  fe = "repetition")

write_table(
  list(t5b_1, t5b_2, t5b_3, t5b_4, t5b_5, t5b_6),
  title = "Gamification and Trading Profits by Type",
  headers = list("Market makers" = 2, "Directional types" = 4),
  file = "t5b_type_profits.tex",
  extralines = list(
    "_Repetition dummy" = c("", "Yes", "Yes", "Yes", "Yes", "Yes")
  )
)

# ============================================================
# Table 6 (fig 7): Forecast accuracy and bias — trader x day
# Forecasts elicited on days 3, 6, 9, 12 (others are NA)
# ============================================================
t6_1 <- ols("forecast_err_price",  g(),                 trader_day, fe = "trading_day")
t6_2 <- ols("forecast_err_price",  g(),                 trader_day, fe = c("trading_day", "repetition"))
t6_3 <- ols("forecast_err_fund",   g(),                 trader_day, fe = c("trading_day", "repetition"))
t6_4 <- ols("forecast_bias_fund",  g(),                 trader_day, fe = "trading_day")
t6_5 <- ols("forecast_bias_fund",  g(),                 trader_day, fe = c("trading_day", "repetition"))
t6_6 <- ols("forecast_bias_fund",  g("gamified * late"), trader_day, fe = c("trading_day", "repetition"))

write_table(
  list(t6_1, t6_2, t6_3, t6_4, t6_5, t6_6),
  title = "Gamification and Price Forecasts",
  headers = list("Accuracy" = 3, "Bias vs fundamental" = 3),
  file = "t6_forecasts.tex",
  extralines = list(
    "_Repetition dummy"    = c("", "Yes", "Yes", "", "Yes", "Yes"),
    "_Trading-day dummies" = rep("Yes", 6)
  )
)

# ============================================================
# Table 7 (fig 8): Bubble incidence — market-rep counts
# ============================================================
t7_1 <- ols("n_bubble_days", g(), mkt)
t7_2 <- ols("n_bubble_days", g(), mkt, fe = "repetition")
t7_3 <- ols("n_bubble_runs", g(), mkt, fe = "repetition")
t7_4 <- ols("n_surges",      g(), mkt, fe = "repetition")
t7_5 <- ols("n_crashes",     g(), mkt, fe = "repetition")
t7_6 <- ols("n_trades",      g(), mkt, fe = "repetition")

write_table(
  list(t7_1, t7_2, t7_3, t7_4, t7_5, t7_6),
  title = "Gamification and Bubble Incidence",
  headers = list("Bubble flags" = 5, "Volume" = 1),
  file = "t7_bubble_incidence.tex",
  extralines = list(
    "_Repetition dummy" = c("", "Yes", "Yes", "Yes", "Yes", "Yes")
  )
)

# ============================================================
# Table 8: Error correction — market x day
# ret_next = next-day log price change; gap = (P - v_t)/vbar.
# gap < 0 slope = convergence force; gap x gamified > 0 = switched off.
# Gap and OFI are NG-centered, so the constant is the NG mean of ret_next.
# ============================================================
t8_1 <- ols("ret_next", g("fundamental_gap * gamified"), mkt_day)
t8_2 <- ols("ret_next", g("fundamental_gap * gamified + order_flow_imbalance * gamified"),
            mkt_day)
t8_3 <- ols("ret_next", g("fundamental_gap * gamified"), mkt_day, fe = "trading_day")
t8_4 <- ols("ret_next", g("fundamental_gap * gamified + order_flow_imbalance * gamified"),
            mkt_day, fe = "trading_day")
t8_5 <- ols("ret_next", g("fundamental_gap * gamified"),
            mkt_day, fe = c("trading_day", "repetition"))
t8_6 <- ols("ret_next", g("fundamental_gap * gamified + order_flow_imbalance * gamified"),
            mkt_day, fe = c("trading_day", "repetition"))

write_table(
  list(t8_1, t8_2, t8_3, t8_4, t8_5, t8_6),
  title = "Gamification Switches Off Error Correction",
  headers = list("Pooled" = 2, "Day FE" = 2, "Day + Repetition FE" = 2),
  file = "t8_error_correction.tex",
  order = c("^Gap$", "Gap .*Gamified", "^OFI$", "OFI .*Gamified", "Gamified$",
            "Finance course"),
  extralines = list(
    "_Repetition dummy"    = c("", "", "", "", "Yes", "Yes"),
    "_Trading-day dummies" = c("", "", "Yes", "Yes", "Yes", "Yes")
  )
)

message("All tables written to ", TABLES)
