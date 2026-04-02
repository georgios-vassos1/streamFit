## ============================================================================
## Demo: Regime Shift Convergence Visualization
##
## Reproduces the convergence and regime-shift tracking demos from the
## standalone scripts in desgop/utils/ using the streamFit package API.
##
## Sections:
##   1. Logistic convergence — RLS-GLM vs iSGD (L2 distance + beta[2] path)
##   2. Poisson convergence  — same overlay
##   3. Regime shift          — per-coefficient paths (RLS-GLM vs iSGD)
##   4. Forgetting factor     — lambda=1 vs lambda=0.98 (RLS-GLM only)
##
## Run from the package root:
##   source("inst/benchmarks/demo_regime_shift.R")
## or from the shell:
##   Rscript inst/benchmarks/demo_regime_shift.R
## ============================================================================

library(streamFit)

## =========================================================================
## Section 1: Logistic convergence — RLS-GLM vs iSGD
## =========================================================================
set.seed(42)
N <- 3000L; p <- 5L
X <- cbind(1, matrix(rnorm(N * (p - 1L)), nrow = N))
beta_true <- c(-0.5, 1.0, -1.0, 0.5, -0.5)
y <- rbinom(N, 1L, 1 / (1 + exp(-X %*% beta_true)))

## Batch MLE reference
fit_batch <- glm(y ~ X[, -1], family = binomial())
beta_mle  <- unname(coef(fit_batch))

## Online fits
fit_rls  <- rls_glm_fit(X, y, family = binomial(), lambda = 1)
fit_isgd <- isgd_fit(X, y, family = binomial(), gamma1 = 1, alpha = 0.7)

cat("\n--- Section 1: Logistic Regression ---\n")
print(round(cbind(
  true      = beta_true,
  batch_mle = beta_mle,
  rls_glm   = coef(fit_rls),
  isgd      = coef(fit_isgd)
), 4))

## Plot 1: L2 distance to batch MLE
d_rls  <- conv_path(fit_rls$beta_path,  beta_mle)
d_isgd <- conv_path(fit_isgd$beta_path, beta_mle)
ylim1  <- range(c(d_rls, d_isgd), na.rm = TRUE)

plot(d_rls, type = "l", col = "steelblue", ylim = ylim1,
     xlab = "Observation", ylab = "L2 distance to batch MLE",
     main = "Logistic: convergence to MLE")
lines(d_isgd, col = "tomato")
legend("topright",
       legend = c("RLS-GLM (2nd order)", "iSGD (1st order)"),
       col = c("steelblue", "tomato"), lty = 1, cex = 0.8)

## Plot 2: Beta[2] path
ylim2 <- range(c(fit_rls$beta_path[, 2], fit_isgd$beta_path[, 2]),
               na.rm = TRUE)
plot(fit_rls$beta_path[, 2], type = "l", col = "steelblue", ylim = ylim2,
     xlab = "Observation", ylab = expression(beta[2]),
     main = "Logistic: beta[2] path")
lines(fit_isgd$beta_path[, 2], col = "tomato")
abline(h = beta_mle[2], lty = 2, col = "black")
legend("bottomright",
       legend = c("RLS-GLM", "iSGD", "Batch MLE"),
       col = c("steelblue", "tomato", "black"), lty = c(1, 1, 2), cex = 0.8)

## =========================================================================
## Section 2: Poisson convergence — RLS-GLM vs iSGD
## =========================================================================
set.seed(7)
N <- 3000L; p <- 4L
X <- cbind(1, matrix(rnorm(N * (p - 1L)), nrow = N))
beta_true_p <- c(0.5, 0.8, -0.6, 0.4)
y_p <- rpois(N, exp(X %*% beta_true_p))

## Batch MLE reference
fit_batch_p <- glm(y_p ~ X[, -1], family = poisson())
beta_mle_p  <- unname(coef(fit_batch_p))

## Online fits — Poisson needs warm start and score clipping for RLS-GLM
beta_init_p <- c(log(mean(y_p)), rep(0, p - 1L))
fit_rls_p  <- rls_glm_fit(X, y_p, family = poisson(), lambda = 1,
                            S0_scale = 0.1, beta_init = beta_init_p,
                            score_clip = 5 * sqrt(mean(y_p)))
fit_isgd_p <- isgd_fit(X, y_p, family = poisson(), gamma1 = 1, alpha = 0.7)

cat("\n--- Section 2: Poisson Regression ---\n")
print(round(cbind(
  true      = beta_true_p,
  batch_mle = beta_mle_p,
  rls_glm   = coef(fit_rls_p),
  isgd      = coef(fit_isgd_p)
), 4))

## Plot 3: L2 distance to batch MLE
d_rls_p  <- conv_path(fit_rls_p$beta_path,  beta_mle_p)
d_isgd_p <- conv_path(fit_isgd_p$beta_path, beta_mle_p)
ylim3    <- range(c(d_rls_p, d_isgd_p), na.rm = TRUE)

plot(d_rls_p, type = "l", col = "steelblue", ylim = ylim3,
     xlab = "Observation", ylab = "L2 distance to batch MLE",
     main = "Poisson: convergence to MLE")
lines(d_isgd_p, col = "tomato")
legend("topright",
       legend = c("RLS-GLM (2nd order)", "iSGD (1st order)"),
       col = c("steelblue", "tomato"), lty = 1, cex = 0.8)

## Plot 4: Beta[2] path
ylim4 <- range(c(fit_rls_p$beta_path[, 2], fit_isgd_p$beta_path[, 2]),
               na.rm = TRUE)
plot(fit_rls_p$beta_path[, 2], type = "l", col = "steelblue", ylim = ylim4,
     xlab = "Observation", ylab = expression(beta[2]),
     main = "Poisson: beta[2] path")
lines(fit_isgd_p$beta_path[, 2], col = "tomato")
abline(h = beta_mle_p[2], lty = 2, col = "black")
legend("bottomright",
       legend = c("RLS-GLM", "iSGD", "Batch MLE"),
       col = c("steelblue", "tomato", "black"), lty = c(1, 1, 2), cex = 0.8)

## =========================================================================
## Section 3: Regime shift — RLS-GLM vs iSGD
## =========================================================================
set.seed(99)
N       <- 2000L
p       <- 3L
t_break <- 1000L
X <- cbind(1, matrix(rnorm(N * (p - 1L)), nrow = N))

beta_r1 <- c(-0.5,  1.0, -0.8)
beta_r2 <- c( 0.5, -1.0,  0.8)

eta_pw                  <- numeric(N)
eta_pw[1:t_break]       <- X[1:t_break, ]       %*% beta_r1
eta_pw[(t_break + 1):N] <- X[(t_break + 1):N, ] %*% beta_r2
y_shift <- rbinom(N, 1L, 1 / (1 + exp(-eta_pw)))

## RLS-GLM: lambda=0.98, S0_scale=0.01
fit_rls_shift <- rls_glm_fit(X, y_shift, family = binomial(),
                               lambda = 0.98, S0_scale = 0.01)

## iSGD: gamma1=3, alpha=0.51
fit_isgd_shift <- isgd_fit(X, y_shift, family = binomial(),
                             gamma1 = 3, alpha = 0.51)

cat("\n--- Section 3: Regime Shift (logistic) ---\n")
print(round(cbind(
  true_regime2 = beta_r2,
  rls_glm      = coef(fit_rls_shift),
  isgd         = coef(fit_isgd_shift)
), 4))

## Plots 5-7: One per coefficient
for (j in seq_len(p)) {
  rls_j  <- fit_rls_shift$beta_path[, j]
  isgd_j <- fit_isgd_shift$beta_path[, j]
  ylim_j <- range(c(rls_j, isgd_j, beta_r1[j], beta_r2[j]))

  plot(rls_j, type = "l", col = "steelblue", ylim = ylim_j,
       xlab = "Observation", ylab = bquote(beta[.(j)]),
       main = paste0("Regime shift: beta[", j, "] — RLS-GLM vs iSGD"))
  lines(isgd_j, col = "tomato")
  ## Piecewise truth
  lines(c(1, t_break),       rep(beta_r1[j], 2), col = "black", lty = 2, lwd = 1.5)
  lines(c(t_break + 1, N),   rep(beta_r2[j], 2), col = "black", lty = 2, lwd = 1.5)
  abline(v = t_break, lty = 3, col = "darkgray")
  if (j == 1L)
    legend("bottomright",
           legend = c("RLS-GLM (lambda=0.98)",
                      "iSGD (gamma1=3, alpha=0.51)",
                      "true beta"),
           col = c("steelblue", "tomato", "black"),
           lty = c(1, 1, 2), lwd = c(1, 1, 1.5), cex = 0.7)
}

## =========================================================================
## Section 4: Forgetting factor comparison (RLS-GLM only)
## =========================================================================
## Reuse the same regime-shift data from Section 3
fit_rls_fixed  <- rls_glm_fit(X, y_shift, family = binomial(),
                                lambda = 1.00, S0_scale = 0.01)
## fit_rls_shift already has lambda=0.98

cat("\n--- Section 4: Forgetting Factor Comparison ---\n")
print(round(cbind(
  true_regime2  = beta_r2,
  lambda_1.00   = coef(fit_rls_fixed),
  lambda_0.98   = coef(fit_rls_shift)
), 4))

## Plot 8: Beta[2] path — lambda=1.00 vs lambda=0.98
path_fixed  <- fit_rls_fixed$beta_path[, 2]
path_forget <- fit_rls_shift$beta_path[, 2]
ylim8 <- range(c(path_fixed, path_forget, beta_r1[2], beta_r2[2]))

plot(path_fixed, type = "l", col = "steelblue", ylim = ylim8,
     xlab = "Observation", ylab = expression(beta[2]),
     main = "Forgetting factor: beta[2] (lambda=1 vs lambda=0.98)")
lines(path_forget, col = "tomato")
lines(c(1, t_break),       rep(beta_r1[2], 2), col = "black", lty = 2, lwd = 1.5)
lines(c(t_break + 1, N),   rep(beta_r2[2], 2), col = "black", lty = 2, lwd = 1.5)
abline(v = t_break, lty = 3, col = "darkgray")
legend("bottomleft",
       legend = c("lambda = 1.00 (no forgetting)",
                  "lambda = 0.98 (adapts)",
                  "true beta[2]"),
       col = c("steelblue", "tomato", "black"),
       lty = c(1, 1, 2), lwd = c(1, 1, 1.5), cex = 0.8)

cat("\nDone — 8 figures produced.\n")
