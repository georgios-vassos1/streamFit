#!/usr/bin/env Rscript
## =========================================================================
## demo_pe_cox.R — Piecewise-exponential Cox model: streamFit vs timereg
## =========================================================================
##
## Scenario A: Stationary Cox model (constant beta) — correctness validation
## Scenario B: Regime shift (time-varying beta) — streamFit adaptation advantage
##
## Requires: streamFit (installed or devtools::load_all), timereg, survival
## =========================================================================

library(streamFit)

if (!requireNamespace("timereg", quietly = TRUE))
  stop("Install timereg: install.packages('timereg')")
if (!requireNamespace("survival", quietly = TRUE))
  stop("Install survival: install.packages('survival')")

library(timereg)
library(survival)

# ── Data simulation helper ───────────────────────────────────────────────

#' Simulate piecewise-exponential Cox data with possibly time-varying beta
#'
#' @param n        Number of subjects
#' @param beta_fun Function of time returning the coefficient vector (length q)
#' @param baseline_hazard Numeric vector of K baseline hazard values
#' @param breaks   Numeric vector of K+1 cut points
#' @param censor_range Length-2 vector: uniform censoring bounds
#' @param seed     Optional RNG seed
#' @return data.frame with columns: id, start, stop, status, x1, x2, ...
sim_pe_cox <- function(n, beta_fun, baseline_hazard, breaks,
                       censor_range, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  K <- length(breaks) - 1L
  stopifnot(length(baseline_hazard) == K)

  q <- length(beta_fun(0))
  covariates <- matrix(NA_real_, n, q)
  covariates[, 1] <- rnorm(n)
  if (q >= 2) covariates[, 2] <- rbinom(n, 1, 0.5)
  if (q >= 3) for (j in 3:q) covariates[, j] <- rnorm(n)

  event_time <- numeric(n)
  for (i in seq_len(n)) {
    t_event <- Inf
    for (k in seq_len(K)) {
      t_lo  <- breaks[k]
      t_hi  <- breaks[k + 1]
      width <- t_hi - t_lo
      t_mid <- (t_lo + t_hi) / 2
      beta_k   <- beta_fun(t_mid)
      lambda_k <- baseline_hazard[k] * exp(sum(covariates[i, ] * beta_k))
      t_cond   <- rexp(1, rate = lambda_k)
      if (t_cond < width) {
        t_event <- t_lo + t_cond
        break
      }
    }
    event_time[i] <- t_event
  }

  C      <- runif(n, censor_range[1], censor_range[2])
  time   <- pmin(event_time, C)
  status <- as.integer(event_time <= C)

  df <- data.frame(id = seq_len(n), start = rep(0, n),
                   stop = time, status = status)
  for (j in seq_len(q)) df[[paste0("x", j)]] <- covariates[, j]
  df
}


## =====================================================================
##   Scenario A: Stationary Cox model — correctness validation
## =====================================================================

cat("\n====================================================================\n")
cat("  Scenario A: Stationary Cox model — correctness validation\n")
cat("====================================================================\n\n")

n_A   <- 2000
K_A   <- 10
breaks_A       <- seq(0, 5, length.out = K_A + 1)
baseline_haz_A <- rep(0.3, K_A)
beta_true_A    <- c(0.5, -0.3)

dat_A <- sim_pe_cox(n = n_A, beta_fun = function(t) beta_true_A,
                    baseline_hazard = baseline_haz_A,
                    breaks = breaks_A, censor_range = c(2, 5), seed = 42)

cat(sprintf("Simulated %d subjects, %d events (%.1f%% censored)\n",
            n_A, sum(dat_A$status), 100 * (1 - mean(dat_A$status))))

X_A <- as.matrix(dat_A[, c("x1", "x2")])

# 1. pe_cox_fit (streamFit, lambda = 1)
fit_stream_A <- pe_cox_fit(dat_A$start, dat_A$stop, dat_A$status, X_A,
                           breaks_A, method = "rls_glm", lambda = 1.0)

# 2. Batch PE Poisson GLM (gold standard)
pe_A    <- surv_to_pe(dat_A$start, dat_A$stop, dat_A$status, X_A, breaks_A)
fit_glm_A <- glm(pe_A$d ~ pe_A$X_pe - 1, family = poisson(),
                  offset = log(pe_A$e))

# 3. timereg::timecox  (const() → time-constant effects)
fit_tc_A <- timecox(Surv(stop, status) ~ const(x1) + const(x2),
                    data = dat_A, max.time = max(breaks_A) - 0.01)

cat("\n--- Covariate beta estimates ---\n")
cat(sprintf("%-22s  %8s  %8s\n", "Method", "x1", "x2"))
cat(sprintf("%-22s  %8.4f  %8.4f\n", "True",
            beta_true_A[1], beta_true_A[2]))
cat(sprintf("%-22s  %8.4f  %8.4f\n", "Batch GLM (PE)",
            coef(fit_glm_A)[K_A + 1], coef(fit_glm_A)[K_A + 2]))
cat(sprintf("%-22s  %8.4f  %8.4f\n", "streamFit (lam=1)",
            fit_stream_A$beta["x1"], fit_stream_A$beta["x2"]))
cat(sprintf("%-22s  %8.4f  %8.4f\n", "timereg::timecox",
            fit_tc_A$gamma[1, 1], fit_tc_A$gamma[2, 1]))


## =====================================================================
##   Scenario B: Regime shift — online adaptation with forgetting
## =====================================================================
##
## Fully online, single-pass, one-record-at-a-time procedure.
## No oracle knowledge; baseline and covariate effects are jointly estimated.
##
## In survival analysis the natural streaming unit is the calendar-time
## interval.  At the end of each interval [t_{k-1}, t_k] we observe, for
## every subject still at risk, whether an event occurred and the exposure.
## PE records are therefore processed one at a time in interval order:
## all records for interval 1, then interval 2, etc.
##
## The full PE design has K interval-dummy columns (sparse — only column k
## is active when processing interval k) plus q dense covariate columns.
## With forgetting (lambda < 1) the inverse-Fisher diagonal for inactive
## interval parameters inflates as S0 * (1/lambda)^N ("covariance windup").
## Fix: clamp diag(S) at S_max = S0_scale after each update — this caps
## prior uncertainty for dormant parameters, preventing runaway variance.
##
## Two online estimators are compared:
##   1. RLS-GLM with forgetting + S clamping  (O(p^2) per step)
##   2. iSGD with slow learning-rate decay    (O(p)   per step)
##
## Both process the same stream of PE records, one at a time, and jointly
## estimate baseline hazard and covariate effects.
## =====================================================================

cat("\n\n====================================================================\n")
cat("  Scenario B: Regime shift — online adaptation with forgetting\n")
cat("====================================================================\n\n")

n_B   <- 4000
K_B   <- 20
breaks_B       <- seq(0, 10, length.out = K_B + 1)
baseline_haz_B <- rep(0.3, K_B)

beta_fun_B <- function(t) {
  if (t <= 5) c(0.5, -0.3) else c(-0.3, 0.5)
}

dat_B <- sim_pe_cox(n = n_B, beta_fun = beta_fun_B,
                    baseline_hazard = baseline_haz_B,
                    breaks = breaks_B, censor_range = c(5, 10), seed = 123)

cat(sprintf("Simulated %d subjects, %d events (%.1f%% censored)\n",
            n_B, sum(dat_B$status), 100 * (1 - mean(dat_B$status))))

X_B <- as.matrix(dat_B[, c("x1", "x2")])

## Expand to person-interval records
pe_B <- surv_to_pe(dat_B$start, dat_B$stop, dat_B$status, X_B, breaks_B)

p <- K_B + ncol(X_B)                        # total parameters
q <- ncol(X_B)                               # covariate count
cov_cols <- (K_B + 1):(K_B + q)

## Warm start: flat hazard + zero covariate effects
avg_rate <- max(sum(pe_B$d) / sum(pe_B$e), 0.01)
beta_init <- c(rep(log(avg_rate), K_B), rep(0, q))

lambda_forget <- 0.999
S0_scale      <- 1.0
fam           <- stats::poisson()

## ---- Calendar-interval streaming loop ----
## Process records one at a time, interval by interval.
## Uses rls_glm_update() with S_max to prevent covariance windup.

## RLS-GLM state (forgetting + S_max clamping)
beta_rls   <- beta_init
S_rls      <- S0_scale * diag(p)
## RLS-GLM state (no forgetting — control)
beta_rls1  <- beta_init
S_rls1     <- S0_scale * diag(p)
## iSGD state
beta_sgd   <- beta_init
sgd_count  <- 0L
sgd_alpha  <- 0.51     # slowest valid decay, for tracking
sgd_gamma1 <- 1.0

## Storage: beta at end of each interval
beta_by_int_rls  <- matrix(NA_real_, K_B, p)
beta_by_int_rls1 <- matrix(NA_real_, K_B, p)
beta_by_int_sgd  <- matrix(NA_real_, K_B, p)

for (k in seq_len(K_B)) {
  idx_k <- which(pe_B$interval == k)
  ## Shuffle within interval (subjects arrive in random order within period)
  idx_k <- sample(idx_k)

  for (j in idx_k) {
    x_j <- pe_B$X_pe[j, ]
    y_j <- pe_B$d[j]
    off_j <- log(pe_B$e[j])

    ## --- RLS-GLM with forgetting + S_max clamping ---
    out_rls <- rls_glm_update(x_j, y_j, beta_rls, S_rls, fam,
                               lambda = lambda_forget, score_clip = 5,
                               offset = off_j, S_max = S0_scale)
    beta_rls <- out_rls$beta
    S_rls    <- out_rls$S

    ## --- RLS-GLM without forgetting (lambda = 1) ---
    out_rls1 <- rls_glm_update(x_j, y_j, beta_rls1, S_rls1, fam,
                                lambda = 1, score_clip = 5,
                                offset = off_j)
    beta_rls1 <- out_rls1$beta
    S_rls1    <- out_rls1$S

    ## --- iSGD (alpha ~ 0.5 for tracking) ---
    sgd_count <- sgd_count + 1L
    gamma_t   <- sgd_gamma1 * sgd_count^(-sgd_alpha)
    beta_sgd  <- isgd_update(x_j, y_j, beta_sgd, gamma_t, fam, offset = off_j)
  }

  beta_by_int_rls[k, ]  <- beta_rls
  beta_by_int_rls1[k, ] <- beta_rls1
  beta_by_int_sgd[k, ]  <- beta_sgd
}

int_times <- breaks_B[-1]   # K time points: 0.5, 1.0, ..., 10.0

## Print results
cat("--- Covariate beta(t) at end of each interval ---\n")
cat(sprintf("%-10s  %-16s  %-16s  %-16s  %-16s\n",
            "Time", "True", "RLS (lam=.999)", "RLS (lam=1)", "iSGD (a=.51)"))
for (k in seq_len(K_B)) {
  t_mid <- (breaks_B[k] + breaks_B[k + 1]) / 2
  tb <- beta_fun_B(t_mid)
  cat(sprintf("[%4.1f,%4.1f]  (%5.2f,%5.2f)  (%6.3f,%6.3f)  (%6.3f,%6.3f)  (%6.3f,%6.3f)\n",
              breaks_B[k], breaks_B[k + 1], tb[1], tb[2],
              beta_by_int_rls[k, cov_cols[1]], beta_by_int_rls[k, cov_cols[2]],
              beta_by_int_rls1[k, cov_cols[1]], beta_by_int_rls1[k, cov_cols[2]],
              beta_by_int_sgd[k, cov_cols[1]], beta_by_int_sgd[k, cov_cols[2]]))
}

## timereg::timecox — time-varying coefficients (batch)
fit_tc_B <- timecox(Surv(stop, status) ~ x1 + x2,
                    data = dat_B, max.time = max(breaks_B) - 0.1)

## Extract timecox beta(t) via loess-smoothed derivative of B(t)
tc_cum   <- fit_tc_B$cum
tc_times <- tc_cum[, 1]
tc_B_x1  <- tc_cum[, "x1"]
tc_B_x2  <- tc_cum[, "x2"]

local_deriv <- function(times, B_cum, eval_times = times, span = 0.15) {
  lo <- loess(B_cum ~ times, span = span)
  h  <- diff(range(times)) * 1e-4
  f1 <- predict(lo, newdata = data.frame(times = eval_times + h))
  f0 <- predict(lo, newdata = data.frame(times = eval_times - h))
  (f1 - f0) / (2 * h)
}

tc_beta1 <- local_deriv(tc_times, tc_B_x1)
tc_beta2 <- local_deriv(tc_times, tc_B_x2)

## ----- MSE on post-shift intervals (5, 10] -----

beta_true_t <- function(t) if (t <= 5) c(0.5, -0.3) else c(-0.3, 0.5)

post_ints <- which(int_times > 5)
post_times <- int_times[post_ints]
beta_true_post <- t(sapply(post_times, beta_true_t))

# RLS forgetting
mse_rls <- colMeans((beta_by_int_rls[post_ints, cov_cols, drop = FALSE] -
                      beta_true_post)^2)
# RLS no forgetting
mse_rls1 <- colMeans((beta_by_int_rls1[post_ints, cov_cols, drop = FALSE] -
                       beta_true_post)^2)
# iSGD
mse_sgd <- colMeans((beta_by_int_sgd[post_ints, cov_cols, drop = FALSE] -
                      beta_true_post)^2)

# timereg beta(t) at post-shift interval endpoints
eval_grid <- post_times
tc_beta1_grid <- approx(tc_times, tc_beta1, xout = eval_grid)$y
tc_beta2_grid <- approx(tc_times, tc_beta2, xout = eval_grid)$y
mse_tc <- c(mean((tc_beta1_grid - beta_true_post[, 1])^2, na.rm = TRUE),
            mean((tc_beta2_grid - beta_true_post[, 2])^2, na.rm = TRUE))

cat("\n--- Post-shift MSE of beta(t) in (5, 10] ---\n")
cat(sprintf("%-40s  %10s  %10s  %10s\n", "Method", "x1", "x2", "mean"))
cat(sprintf("%-40s  %10.6f  %10.6f  %10.6f\n",
            sprintf("RLS-GLM (lam=%.3f, forgetting)", lambda_forget),
            mse_rls[1], mse_rls[2], mean(mse_rls)))
cat(sprintf("%-40s  %10.6f  %10.6f  %10.6f\n",
            sprintf("iSGD (alpha=%.2f)", sgd_alpha),
            mse_sgd[1], mse_sgd[2], mean(mse_sgd)))
cat(sprintf("%-40s  %10.6f  %10.6f  %10.6f\n", "RLS-GLM (lam=1, no forgetting)",
            mse_rls1[1], mse_rls1[2], mean(mse_rls1)))
cat(sprintf("%-40s  %10.6f  %10.6f  %10.6f\n", "timereg::timecox (batch)",
            mse_tc[1], mse_tc[2], mean(mse_tc)))


## ----- Plot: beta(t) paths — RLS vs iSGD vs timecox -----

tryCatch({
  pdf("pe_cox_regime_shift.pdf", width = 10, height = 8)
  par(mfrow = c(2, 1), mar = c(4, 4.5, 3, 1))

  for (cov_j in 1:2) {
    cov_name <- c("x1", "x2")[cov_j]
    col_j <- cov_cols[cov_j]

    t_grid    <- seq(0, 10, length.out = 500)
    beta_grid <- sapply(t_grid, function(t) beta_true_t(t)[cov_j])

    tc_beta_j <- if (cov_j == 1) tc_beta1 else tc_beta2

    yl <- range(c(beta_grid,
                  quantile(tc_beta_j, c(0.02, 0.98), na.rm = TRUE),
                  beta_by_int_rls[, col_j],
                  beta_by_int_sgd[, col_j]),
                na.rm = TRUE)
    yl <- yl + c(-0.2, 0.2) * diff(yl)

    # True beta(t)
    plot(t_grid, beta_grid, type = "l", col = "black", lwd = 2, lty = 3,
         xlim = c(0, 10), ylim = yl,
         xlab = "Time", ylab = bquote(hat(beta)[.(cov_name)](t)),
         main = paste0("Coefficient ", cov_name,
                       ": online streaming vs timereg::timecox (batch)"))

    # timereg beta(t) path
    lines(tc_times, tc_beta_j, col = "darkgreen", lwd = 1.5, lty = 1)

    # RLS-GLM with forgetting
    lines(int_times, beta_by_int_rls[, col_j],
          type = "s", col = "blue", lwd = 2, lty = 1)

    # iSGD
    lines(int_times, beta_by_int_sgd[, col_j],
          type = "s", col = "purple", lwd = 2, lty = 1)

    # RLS-GLM without forgetting
    lines(int_times, beta_by_int_rls1[, col_j],
          type = "s", col = "red", lwd = 1.5, lty = 2)

    # Regime shift line
    abline(v = 5, col = "grey60", lty = 4)
    text(5.1, yl[2] - 0.05 * diff(yl), "regime shift", adj = 0, cex = 0.8)

    # Place legend away from the line traces:
    # x1: pre-shift at 0.5 (top), post-shift at -0.3 (bottom) => bottomright
    # x2: pre-shift at -0.3 (bottom), post-shift at 0.5 (top) => topleft
    leg_pos <- if (cov_j == 1) "bottomright" else "topleft"
    legend(leg_pos,
           legend = c("true beta(t)",
                       "timereg::timecox (batch)",
                       sprintf("RLS-GLM (lam=%.3f)", lambda_forget),
                       sprintf("iSGD (alpha=%.2f)", sgd_alpha),
                       "RLS-GLM (lam=1, no forgetting)"),
           col = c("black", "darkgreen", "blue", "purple", "red"),
           lty = c(3, 1, 1, 1, 2), lwd = c(2, 1.5, 2, 2, 1.5),
           cex = 0.65, bg = "white")
  }

  par(mfrow = c(1, 1))
  dev.off()
  cat("\nPlot saved to pe_cox_regime_shift.pdf\n")
}, error = function(e) {
  cat("Plotting skipped:", e$message, "\n")
})

cat("\nDone.\n")
