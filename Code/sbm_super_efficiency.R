# DEPRECATED R IMPLEMENTATION
# ---------------------------
# This R script implements SBM (standard) and super-efficiency SBM with
# undesirable outputs, mirroring the formulations in Tone (2001, 2002).
#
# It is kept for reference only. The maintained and recommended
# implementations are the Python scripts in:
#   - Code/sbm_run.py              (script-style entry, reproduces Tone models)
#   - Code/sbm_super_efficiency.py (modular Python API)
#
# The R version is no longer actively maintained. For new work, please use
# the Python implementations above.

# Historical R implementation of SBM (standard) and super-efficiency SBM with undesirable outputs
# - Reads Excel file with header on second row, data from third row
# - Uses lpSolve for linear programming
# - Outputs CSV with rho_sup/rho_std, statuses, slacks and targets per DMU

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tibble)
})

safe_num <- function(x) {
  suppressWarnings(as.numeric(x))
}

load_clean_excel <- function(path, sheet = NULL) {
  sheets <- readxl::excel_sheets(path)
  sheet_name <- if (is.null(sheet)) sheets[[1]] else sheet
  raw <- readxl::read_excel(path, sheet = sheet_name, col_names = FALSE)
  cols <- as.character(unlist(raw[2, ]))
  df <- raw[-c(1,2), ]
  colnames(df) <- cols
  # Ensure numeric columns except DMU id
  df <- df %>% mutate(across(-c('年份'), safe_num))
  return(df)
}

default_mapping <- function(df) {
  cols <- colnames(df)
  need <- c('年份',
            '建设用地面积（平方千米）',
            '耕地面积（万公顷）',
            '生态用地面积(公顷)',
            '年末从业人数/万人',
            '固定资产投资额（亿元）',
            '用电量（亿千瓦时）',
            '研究与试验发展（R&D）经费支出（亿元）',
            '城乡居民可支配收入',
            '粮食总产量（万吨）',
            '森林覆盖率',
            '碳排放量/万t')
  miss <- setdiff(need, cols)
  if (length(miss) > 0) {
    stop(paste('以下列在数据表中未找到:', paste(miss, collapse = ', ')))
  }
  list(
    dmu_id = '年份',
    inputs = c('建设用地面积（平方千米）', '耕地面积（万公顷）', '生态用地面积(公顷)', '年末从业人数/万人', '固定资产投资额（亿元）', '用电量（亿千瓦时）', '研究与试验发展（R&D）经费支出（亿元）'),
    good_outputs = c('城乡居民可支配收入', '粮食总产量（万吨）', '森林覆盖率'),
    bad_outputs = c('碳排放量/万t')
  )
}

build_matrices <- function(df, mapping) {
  X <- as.matrix(df[, mapping$inputs])
  Yg <- as.matrix(df[, mapping$good_outputs])
  Yb <- as.matrix(df[, mapping$bad_outputs])
  dmu_ids <- as.character(df[[mapping$dmu_id]])
  list(X = X, Yg = Yg, Yb = Yb, dmu_ids = dmu_ids)
}

denom_safe <- function(v, eps = 1e-8) {
  v2 <- v
  v2[abs(v2) < eps] <- eps
  v2
}

# Standard SBM with undesirable outputs (rho <= 1)
sbm_standard_undesirable <- function(X, Yg, Yb, returns_to_scale = 'VRS', eps = 1e-8) {
  n <- nrow(X); m <- ncol(X)
  s1 <- ncol(Yg); s2 <- ncol(Yb)
  results <- list(rho = numeric(n), status = character(n), t = numeric(n),
                  s_input = vector('list', n), s_good = vector('list', n), s_bad = vector('list', n),
                  lambda = vector('list', n))
  for (o in seq_len(n)) {
    # variable order: lam(1..n), s_in(1..m), s_g(1..s1), s_b(1..s2), t
    L <- n; total_vars <- L + m + s1 + s2 + 1
    # Objective: t - (1/m) sum s_in'/x_o
    obj <- numeric(total_vars)
    x_o_safe <- denom_safe(X[o, ], eps)
    obj[L + (1:m)] <- - (1.0 / max(1, m)) * (1.0 / x_o_safe)
    obj[total_vars] <- 1.0

    # Constraints
    rows <- list(); rhs <- c(); dir <- c()

    # Input balances
    for (r in seq_len(m)) {
      row <- numeric(total_vars)
      # lam contributions (include all DMUs)
      row[1:L] <- X[, r]
      # s_in'
      row[L + r] <- 1.0
      # t coefficient
      row[total_vars] <- - X[o, r]
      rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 0); dir <- c(dir, '=')
    }
    # Good outputs
    for (j in seq_len(s1)) {
      row <- numeric(total_vars)
      row[1:L] <- Yg[, j]
      row[L + m + j] <- -1.0
      row[total_vars] <- - Yg[o, j]
      rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 0); dir <- c(dir, '=')
    }
    # Bad outputs
    for (k in seq_len(s2)) {
      row <- numeric(total_vars)
      row[1:L] <- Yb[, k]
      row[L + m + s1 + k] <- 1.0
      row[total_vars] <- - Yb[o, k]
      rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 0); dir <- c(dir, '=')
    }
    # VRS convexity
    if (toupper(returns_to_scale) == 'VRS') {
      row <- numeric(total_vars)
      row[1:L] <- 1.0
      row[total_vars] <- -1.0
      rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 0); dir <- c(dir, '=')
    }
    # Denominator normalization: t + (1/(s1+s2))*(sum s_g'/y_g + sum s_b'/y_b) = 1
    row <- numeric(total_vars)
    yg_safe <- denom_safe(Yg[o, ], eps); yb_safe <- denom_safe(Yb[o, ], eps)
    d_factor <- 1.0 / max(1, (s1 + s2))
    if (s1 > 0) {
      row[L + m + (1:s1)] <- d_factor * (1.0 / yg_safe)
    }
    if (s2 > 0) {
      row[L + m + s1 + (1:s2)] <- d_factor * (1.0 / yb_safe)
    }
    row[total_vars] <- 1.0
    rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 1.0); dir <- c(dir, '=')

    const.mat <- do.call(rbind, rows)

    # bounds: all >= 0
    lower <- rep(0, total_vars)
    upper <- rep(Inf, total_vars)

    sol <- lpSolve::lp(direction = 'min', objective.in = obj,
                       const.mat = const.mat, const.dir = dir, const.rhs = rhs,
                       all.int = FALSE, all.bin = FALSE, compute.sens = FALSE)

    status <- if (sol$status == 0) 'Optimal' else 'Infeasible'
    results$status[[o]] <- status
    results$rho[[o]] <- sol$objval
    vals <- sol$solution
    t_val <- vals[total_vars]
    results$t[[o]] <- t_val
    lam_vals <- vals[1:L]
    s_in_p <- vals[L + (1:m)]
    s_g_p <- if (s1 > 0) vals[L + m + (1:s1)] else numeric(0)
    s_b_p <- if (s2 > 0) vals[L + m + s1 + (1:s2)] else numeric(0)

    if (!is.na(t_val) && t_val > eps) {
      results$s_input[[o]] <- s_in_p / t_val
      results$s_good[[o]] <- s_g_p / t_val
      results$s_bad[[o]] <- s_b_p / t_val
    } else {
      results$s_input[[o]] <- rep(NA_real_, m)
      results$s_good[[o]] <- rep(NA_real_, s1)
      results$s_bad[[o]] <- rep(NA_real_, s2)
    }
    results$lambda[[o]] <- as.numeric(lam_vals)
  }
  results
}

# Super-efficiency SBM with undesirable outputs (rho >= 1)
sbm_super_undesirable <- function(X, Yg, Yb, returns_to_scale = 'CRS', eps = 1e-8) {
  n <- nrow(X); m <- ncol(X)
  s1 <- ncol(Yg); s2 <- ncol(Yb)
  results <- list(rho = numeric(n), status = character(n), t = numeric(n),
                  s_input = vector('list', n), s_good = vector('list', n), s_bad = vector('list', n),
                  lambda = vector('list', n))
  for (o in seq_len(n)) {
    # exclude o in reference set
    ref_idx <- setdiff(seq_len(n), o)
    L <- length(ref_idx)
    total_vars <- L + m + s1 + s2 + 1
    obj <- numeric(total_vars)
    # Objective: t + (1/(s1+s2))*(sum s_g'/y_g + sum s_b'/y_b)
    yg_safe <- denom_safe(Yg[o, ], eps); yb_safe <- denom_safe(Yb[o, ], eps)
    d_factor <- 1.0 / max(1, (s1 + s2))
    if (s1 > 0) obj[L + m + (1:s1)] <- d_factor * (1.0 / yg_safe)
    if (s2 > 0) obj[L + m + s1 + (1:s2)] <- d_factor * (1.0 / yb_safe)
    obj[total_vars] <- 1.0

    rows <- list(); rhs <- c(); dir <- c()
    # Input balances: x_o * t = sum lam' X + s_in'
    for (r in seq_len(m)) {
      row <- numeric(total_vars)
      row[1:L] <- X[ref_idx, r]
      row[L + r] <- 1.0
      row[total_vars] <- - X[o, r]
      rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 0); dir <- c(dir, '=')
    }
    # Good outputs: y_g_o * t = sum lam' Yg - s_g'
    for (j in seq_len(s1)) {
      row <- numeric(total_vars)
      row[1:L] <- Yg[ref_idx, j]
      row[L + m + j] <- -1.0
      row[total_vars] <- - Yg[o, j]
      rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 0); dir <- c(dir, '=')
    }
    # Bad outputs: y_b_o * t = sum lam' Yb + s_b'
    for (k in seq_len(s2)) {
      row <- numeric(total_vars)
      row[1:L] <- Yb[ref_idx, k]
      row[L + m + s1 + k] <- 1.0
      row[total_vars] <- - Yb[o, k]
      rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 0); dir <- c(dir, '=')
    }
    # Returns to scale
    if (toupper(returns_to_scale) == 'VRS') {
      row <- numeric(total_vars)
      row[1:L] <- 1.0
      row[total_vars] <- -1.0
      rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 0); dir <- c(dir, '=')
    }
    # N normalization: t - (1/m) * sum s_in'/x_o = 1
    row <- numeric(total_vars)
    x_o_safe <- denom_safe(X[o, ], eps)
    row[L + (1:m)] <- - (1.0 / max(1, m)) * (1.0 / x_o_safe)
    row[total_vars] <- 1.0
    rows[[length(rows) + 1]] <- row; rhs <- c(rhs, 1.0); dir <- c(dir, '=')

    const.mat <- do.call(rbind, rows)
    lower <- rep(0, total_vars)
    upper <- rep(Inf, total_vars)

    sol <- lpSolve::lp(direction = 'min', objective.in = obj,
                       const.mat = const.mat, const.dir = dir, const.rhs = rhs,
                       all.int = FALSE, all.bin = FALSE, compute.sens = FALSE)
    status <- if (sol$status == 0) 'Optimal' else 'Infeasible'
    results$status[[o]] <- status
    results$rho[[o]] <- sol$objval
    vals <- sol$solution
    t_val <- vals[total_vars]
    results$t[[o]] <- t_val
    lam_vals <- vals[1:L]
    s_in_p <- vals[L + (1:m)]
    s_g_p <- if (s1 > 0) vals[L + m + (1:s1)] else numeric(0)
    s_b_p <- if (s2 > 0) vals[L + m + s1 + (1:s2)] else numeric(0)
    if (!is.na(t_val) && t_val > eps) {
      results$s_input[[o]] <- s_in_p / t_val
      results$s_good[[o]] <- s_g_p / t_val
      results$s_bad[[o]] <- s_b_p / t_val
    } else {
      results$s_input[[o]] <- rep(NA_real_, m)
      results$s_good[[o]] <- rep(NA_real_, s1)
      results$s_bad[[o]] <- rep(NA_real_, s2)
    }
    results$lambda[[o]] <- as.numeric(lam_vals)
  }
  results
}

compute_targets <- function(x_o, y_g_o, y_b_o, s_in, s_g, s_b) {
  list(
    x_target = x_o - s_in,
    y_g_target = y_g_o + s_g,
    y_b_target = y_b_o - s_b
  )
}

main <- function() {
  excel_path <- 'd:/Desk/MyDeskFiles/Compe&Event/Feng/数据示例超效率SBM.xlsx'
  df <- load_clean_excel(excel_path)
  mapping <- default_mapping(df)
  mats <- build_matrices(df, mapping)
  X <- mats$X; Yg <- mats$Yg; Yb <- mats$Yb; dmu_ids <- mats$dmu_ids

  res_std <- sbm_standard_undesirable(X, Yg, Yb, returns_to_scale = 'VRS')
  # Only compute sup-eff for efficient DMUs (rho_std ~ 1)
  res_sup <- sbm_super_undesirable(X, Yg, Yb, returns_to_scale = 'CRS')

  rows <- list()
  for (i in seq_along(dmu_ids)) {
    is_eff <- (!is.na(res_std$rho[[i]])) && res_std$status[[i]] == 'Optimal' && abs(res_std$rho[[i]] - 1.0) < 1e-3
    src <- if (is_eff && res_sup$status[[i]] == 'Optimal') res_sup else res_std
    targets <- compute_targets(X[i, ], Yg[i, ], Yb[i, ], src$s_input[[i]], src$s_good[[i]], src$s_bad[[i]])
    row <- tibble(
      DMU = dmu_ids[[i]],
      rho_sup = if (is_eff) res_sup$rho[[i]] else NA_real_,
      status_sup = if (is_eff) res_sup$status[[i]] else 'Skipped',
      t_sup = if (is_eff) res_sup$t[[i]] else NA_real_,
      rho_std = res_std$rho[[i]],
      status_std = res_std$status[[i]],
      t_std = res_std$t[[i]]
    )
    # add slacks and targets flattened
    for (idx in seq_len(ncol(X))) {
      row[[paste0('slack_in_', idx-1)]] <- src$s_input[[i]][idx]
      row[[paste0('target_in_', idx-1)]] <- targets$x_target[idx]
    }
    for (idx in seq_len(ncol(Yg))) {
      row[[paste0('slack_g_', idx-1)]] <- src$s_good[[i]][idx]
      row[[paste0('target_g_', idx-1)]] <- targets$y_g_target[idx]
    }
    for (idx in seq_len(ncol(Yb))) {
      row[[paste0('slack_b_', idx-1)]] <- src$s_bad[[i]][idx]
      row[[paste0('target_b_', idx-1)]] <- targets$y_b_target[idx]
    }
    rows[[length(rows) + 1]] <- row
  }
  out <- dplyr::bind_rows(rows)
  readr::write_csv(out, 'd:/Desk/sbm_super_eff_results_R.csv')
  message('Saved results -> d:/Desk/sbm_super_eff_results_R.csv')
}

if (sys.nframe() == 0) {
  # Ensure lpSolve is available
  if (!requireNamespace('lpSolve', quietly = TRUE)) {
    stop('请先安装 lpSolve 包：install.packages("lpSolve")')
  }
  if (!requireNamespace('readr', quietly = TRUE)) {
    stop('请先安装 readr 包：install.packages("readr")')
  }
  main()
}
