# Descriptive statistics for the Results section.
# Uses the same GHP-versus-NG sample and exclusions as regressions.R.

suppressPackageStartupMessages({
  library(dplyr)
})

.resolve_root <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(file.path(dirname(sub("^--file=", "", file_arg)), "../..")))
  }
  if (file.exists("data/processed/market_day_panel_full.csv")) {
    return(normalizePath("."))
  }
  stop("Cannot locate repository root.")
}

ROOT <- .resolve_root()
PROCESSED <- file.path(ROOT, "data", "processed")
TABLES <- file.path(ROOT, "output", "tables")
dir.create(TABLES, recursive = TRUE, showWarnings = FALSE)

EXCLUDE_GROUPS <- c("20260520_PM/ng1", "20280904/ghp1")

mkt_day <- read.csv(file.path(PROCESSED, "market_day_panel_full.csv")) |>
  filter(treatment %in% c("ng", "ghp"), !(group_label %in% EXCLUDE_GROUPS))

trader_day <- read.csv(file.path(PROCESSED, "trader_day_panel_full.csv")) |>
  filter(treatment %in% c("ng", "ghp"), !(group_label %in% EXCLUDE_GROUPS))

mkt_rep <- mkt_day |>
  group_by(market_uuid, treatment) |>
  summarise(
    bubble_days = sum(bubble_period, na.rm = TRUE),
    bubble_episodes = sum(bubble_start, na.rm = TRUE),
    price_surges = sum(surge, na.rm = TRUE),
    price_crashes = sum(crash, na.rm = TRUE),
    total_trades = sum(n_trades_market, na.rm = TRUE),
    share_market_maker = first(share_market_maker),
    share_fundamental = first(share_fundamental),
    share_feedback = first(share_feedback),
    share_speculator = first(share_speculator),
    share_other = first(share_other),
    .groups = "drop"
  )

trader_final <- trader_day |> filter(trading_day == 15)

stat_row <- function(data, variable, label, scale = 1, unit = "Market-day") {
  x <- data[[variable]] * scale
  x <- x[is.finite(x)]
  q <- quantile(x, c(0.25, 0.50, 0.75), names = FALSE, na.rm = TRUE)
  data.frame(
    Measure = label,
    Unit = unit,
    N = length(x),
    Mean = mean(x),
    SD = sd(x),
    P25 = q[1],
    Median = q[2],
    P75 = q[3],
    check.names = FALSE
  )
}

panel <- function(title) {
  data.frame(
    Measure = paste0("\\multicolumn{8}{l}{\\textit{", title, "}}"),
    Unit = "", N = NA_integer_, Mean = NA_real_, SD = NA_real_,
    P25 = NA_real_, Median = NA_real_, P75 = NA_real_,
    check.names = FALSE
  )
}

rows <- bind_rows(
  panel("Panel A. Price efficiency"),
  stat_row(mkt_day, "avg_abs_mispricing", "Absolute mispricing (E\\$)"),
  stat_row(mkt_day, "abs_mispricing_ratio", "Absolute mispricing ratio"),
  stat_row(mkt_day, "rad", "Relative absolute deviation"),
  stat_row(mkt_rep, "bubble_days", "Bubble days", unit = "Market-repetition"),
  stat_row(mkt_rep, "bubble_episodes", "Bubble episodes", unit = "Market-repetition"),
  stat_row(mkt_rep, "price_surges", "Price surges", unit = "Market-repetition"),
  stat_row(mkt_rep, "price_crashes", "Price crashes", unit = "Market-repetition"),
  stat_row(mkt_rep, "total_trades", "Total trades", unit = "Market-repetition"),

  panel("Panel B. Trading activity"),
  stat_row(mkt_day, "n_trades_market", "Trades"),
  stat_row(mkt_day, "order_flow_imbalance", "Order-flow imbalance"),
  stat_row(mkt_day, "abs_order_flow_imbalance", "Absolute order-flow imbalance"),
  stat_row(mkt_day, "n_limit_orders", "Limit orders submitted"),
  stat_row(mkt_day, "n_cancels", "Cancellations"),
  stat_row(mkt_day, "share_limit_orders", "Share of orders that are limit orders"),
  stat_row(mkt_day, "churn", "Intraday churn"),

  panel("Panel C. Liquidity"),
  stat_row(mkt_day, "rel_quoted_spread", "Relative quoted spread", 100),
  stat_row(mkt_day, "rel_eff_spread", "Relative effective spread", 100),
  stat_row(mkt_day, "rel_realized_spread", "Relative realized spread", 100),
  stat_row(mkt_day, "rel_price_impact", "Relative price impact", 100),
  stat_row(mkt_day, "depth_best", "Depth at best quotes (shares)"),
  stat_row(mkt_day, "rv_mid", "Realized midquote volatility"),
  stat_row(mkt_day, "n_improving_adds", "Spread-improving limit orders"),
  stat_row(mkt_day, "share_improving_adds", "Share improving the spread", 100),
  stat_row(mkt_day, "time_to_same_side_order_s", "Order replenishment (seconds)"),
  stat_row(mkt_day, "spread_recovery_s", "Spread recovery (seconds)"),

  panel("Panel D. Trader types and outcomes"),
  stat_row(mkt_rep, "share_market_maker", "Trader share: market makers", 100,
           "Market-repetition"),
  stat_row(mkt_rep, "share_fundamental", "Trader share: fundamentalists", 100,
           "Market-repetition"),
  stat_row(mkt_rep, "share_feedback", "Trader share: feedback traders", 100,
           "Market-repetition"),
  stat_row(mkt_rep, "share_speculator", "Trader share: speculators", 100,
           "Market-repetition"),
  stat_row(mkt_rep, "share_other", "Trader share: unclassified", 100,
           "Market-repetition"),
  stat_row(mkt_day, "share_vol_market_maker", "Volume share: market makers", 100),
  stat_row(mkt_day, "share_vol_fundamental", "Volume share: fundamentalists", 100),
  stat_row(mkt_day, "share_vol_feedback", "Volume share: feedback traders", 100),
  stat_row(mkt_day, "share_vol_speculator", "Volume share: speculators", 100),
  stat_row(mkt_day, "share_vol_other", "Volume share: unclassified", 100),
  stat_row(mkt_day, "vol_market_maker", "Gross trades: market makers"),
  stat_row(mkt_day, "vol_fundamental", "Gross trades: fundamentalists"),
  stat_row(mkt_day, "vol_feedback", "Gross trades: feedback traders"),
  stat_row(mkt_day, "vol_speculator", "Gross trades: speculators"),
  stat_row(mkt_day, "vol_other", "Gross trades: unclassified"),
  stat_row(filter(trader_final, trader_type == "market_maker"), "rel_wealth",
           "Relative wealth: market makers (E\\$)", unit = "Trader-repetition"),
  stat_row(filter(trader_final, trader_type == "fundamental"), "rel_wealth",
           "Relative wealth: fundamentalists (E\\$)", unit = "Trader-repetition"),
  stat_row(filter(trader_final, trader_type == "feedback"), "rel_wealth",
           "Relative wealth: feedback traders (E\\$)", unit = "Trader-repetition"),
  stat_row(filter(trader_final, trader_type == "speculator"), "rel_wealth",
           "Relative wealth: speculators (E\\$)", unit = "Trader-repetition"),
  stat_row(filter(trader_final, trader_type == "other"), "rel_wealth",
           "Relative wealth: unclassified (E\\$)", unit = "Trader-repetition")
)

fmt <- function(x) {
  ifelse(is.na(x), "", formatC(x, format = "f", digits = 2, big.mark = ","))
}

tex <- c(
  "\\begin{tabular}{llrrrrrr}",
  "\\toprule",
  "Measure & Observation & $N$ & Mean & SD & P25 & Median & P75 \\\\",
  "\\midrule"
)

for (i in seq_len(nrow(rows))) {
  if (startsWith(rows$Measure[i], "\\multicolumn")) {
    tex <- c(tex, paste0(rows$Measure[i], " \\\\"))
  } else {
    tex <- c(tex, paste(
      rows$Measure[i], rows$Unit[i], rows$N[i],
      fmt(rows$Mean[i]), fmt(rows$SD[i]), fmt(rows$P25[i]),
      fmt(rows$Median[i]), fmt(rows$P75[i]),
      sep = " & "
    ) |> paste0(" \\\\"))
  }
}

tex <- c(tex, "\\bottomrule", "\\end{tabular}")
writeLines(tex, file.path(TABLES, "t0_descriptive_statistics.tex"))
message("wrote t0_descriptive_statistics.tex")
