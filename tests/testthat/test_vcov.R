# ===================================================================
# These tests replicate the testing philosophy of the R source GLM
# test suite (src/library/stats/tests/{lm-tests.R, reg-tests-*.R}):
#
#   1. Cross-method consistency: two independent code paths must agree
#   2. Online == batch equivalence for the same data
#   3. Internal consistency (summary, vcov, confint are coherent)
#   4. Dispersion/distribution convention (z vs t, est.disp)
#   5. Edge-case non-failure
#   6. Formula-level correctness for accumulators
# ===================================================================


# ===================================================================
# GLM EQUIVALENCE EVIDENCE
#
# For each family (Gaussian, binomial, Poisson), fit the same data
# with both streamFit and stats::glm, then compare every inference
# output: coef, vcov (full matrix), summary coefficient table (SEs,
# test statistics, p-values), confint, and dispersion.
#
# These are the primary tests establishing that streamFit inference
# is as reliable as stats::glm.  Tolerances reflect the O(1/n)
# online-vs-batch discrepancy that is inherent to sequential fitting.
# ===================================================================


# -------------------------------------------------------------------
# Gaussian: rls_fit vs lm / glm
# -------------------------------------------------------------------

test_that("Gaussian: coef(rls) == coef(lm)", {
  set.seed(301)
  n <- 2000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(1, -0.5, 0.8) + rnorm(n, sd = 0.5)
  fit <- rls_fit(X, y, S0_scale = 1e6)
  fit_lm <- lm(y ~ X[, -1])
  expect_equal(as.numeric(coef(fit)), as.numeric(coef(fit_lm)),
               tolerance = 1e-4)
})

test_that("Gaussian: vcov(rls) matches vcov(lm) (full matrix)", {
  set.seed(302)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(1, -0.5, 0.8) + rnorm(n, sd = 0.5)
  V_rls <- vcov(rls_fit(X, y, S0_scale = 1e6))
  V_lm  <- vcov(lm(y ~ X[, -1]))
  ## Full matrix comparison — not just diagonal
  expect_equal(unname(V_rls), unname(V_lm), tolerance = 0.02)
})

test_that("Gaussian: dispersion (sigma^2) matches summary(lm)$sigma^2", {
  set.seed(303)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(1, -0.5, 0.8) + rnorm(n, sd = 2)
  fit <- rls_fit(X, y, S0_scale = 1e6)
  sigma2_rls <- fit$rss / (fit$n_obs - p)
  sigma2_lm  <- summary(lm(y ~ X[, -1]))$sigma^2
  expect_equal(sigma2_rls, sigma2_lm, tolerance = 0.02)
})

test_that("Gaussian: summary(rls) matches summary(lm) coefficient table", {
  set.seed(304)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(1, -0.5, 0.8) + rnorm(n, sd = 0.5)
  s_rls <- summary(rls_fit(X, y, S0_scale = 1e6))
  s_lm  <- summary(lm(y ~ X[, -1]))
  ## All four columns: Estimate, SE, t, p
  expect_equal(unname(s_rls$coefficients[, "Estimate"]),
               unname(s_lm$coefficients[, "Estimate"]), tolerance = 1e-4)
  expect_equal(unname(s_rls$coefficients[, "Std. Error"]),
               unname(s_lm$coefficients[, "Std. Error"]), tolerance = 0.05)
  expect_equal(unname(s_rls$coefficients[, "t value"]),
               unname(s_lm$coefficients[, "t value"]), tolerance = 0.05)
  ## p-values: compare on -log10 scale for stability near zero
  p_rls <- s_rls$coefficients[, "Pr(>|t|)"]
  p_lm  <- s_lm$coefficients[, "Pr(>|t|)"]
  expect_equal(unname(-log10(pmax(p_rls, 1e-300))),
               unname(-log10(pmax(p_lm, 1e-300))), tolerance = 0.05)
})

test_that("Gaussian: confint(rls) close to confint(lm)", {
  set.seed(305)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(1, -0.5, 0.8) + rnorm(n, sd = 0.5)
  ci_rls <- confint(rls_fit(X, y, S0_scale = 1e6), level = 0.95)
  ci_lm  <- confint.default(lm(y ~ X[, -1]), level = 0.95)
  expect_equal(unname(ci_rls), unname(ci_lm), tolerance = 0.05)
})


# -------------------------------------------------------------------
# Binomial: rls_glm_fit vs glm
# -------------------------------------------------------------------

test_that("binomial: coef(rls_glm) close to coef(glm)", {
  set.seed(311)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  beta_true <- c(-0.5, 1, -0.8)
  y <- rbinom(n, 1, plogis(X %*% beta_true))
  fit <- rls_glm_fit(X, y, family = binomial())
  fit_glm <- glm(y ~ X[, -1], family = binomial())
  expect_equal(as.numeric(coef(fit)), as.numeric(coef(fit_glm)),
               tolerance = 0.02)
})

test_that("binomial: vcov(rls_glm) matches vcov(glm) (full matrix)", {
  set.seed(312)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rbinom(n, 1, plogis(X %*% c(-0.5, 1, -0.8)))
  V_online <- vcov(rls_glm_fit(X, y, family = binomial()))
  V_glm    <- vcov(glm(y ~ X[, -1], family = binomial()))
  ## Full matrix comparison, not just diagonal
  expect_equal(unname(V_online), unname(V_glm), tolerance = 0.15)
})

test_that("binomial: summary(rls_glm) matches summary(glm) coefficient table", {
  set.seed(313)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rbinom(n, 1, plogis(X %*% c(-0.5, 1, -0.8)))
  s_online <- summary(rls_glm_fit(X, y, family = binomial()))
  s_glm    <- summary(glm(y ~ X[, -1], family = binomial()))
  ## SEs within 10%
  se_online <- s_online$coefficients[, "Std. Error"]
  se_glm    <- s_glm$coefficients[, "Std. Error"]
  expect_true(all(abs(se_online - se_glm) / se_glm < 0.10))
  ## Both use z-values
  expect_true("z value" %in% colnames(s_online$coefficients))
  expect_true("z value" %in% colnames(s_glm$coefficients))
  ## z-values within 10%
  z_online <- abs(s_online$coefficients[, "z value"])
  z_glm    <- abs(s_glm$coefficients[, "z value"])
  expect_true(all(abs(z_online - z_glm) / z_glm < 0.10))
})

test_that("binomial: confint(rls_glm) close to confint.default(glm)", {
  set.seed(314)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rbinom(n, 1, plogis(X %*% c(-0.5, 1, -0.8)))
  ci_online <- confint(rls_glm_fit(X, y, family = binomial()), level = 0.95)
  ci_glm    <- confint.default(glm(y ~ X[, -1], family = binomial()),
                               level = 0.95)
  expect_equal(unname(ci_online), unname(ci_glm), tolerance = 0.05)
})


# -------------------------------------------------------------------
# Poisson: rls_glm_fit vs glm
# -------------------------------------------------------------------

test_that("Poisson: coef(rls_glm) close to coef(glm)", {
  set.seed(321)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1), sd = 0.5), n, p - 1))
  beta_true <- c(0.5, 0.3, -0.2)
  y <- rpois(n, exp(X %*% beta_true))
  fit <- rls_glm_fit(X, y, family = poisson(),
                     beta_init = c(log(mean(y)), rep(0, p - 1)),
                     score_clip = 5 * sqrt(mean(y)))
  fit_glm <- glm(y ~ X[, -1], family = poisson())
  expect_equal(as.numeric(coef(fit)), as.numeric(coef(fit_glm)),
               tolerance = 0.05)
})

test_that("Poisson: vcov(rls_glm) matches vcov(glm) (full matrix)", {
  set.seed(322)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1), sd = 0.5), n, p - 1))
  y <- rpois(n, exp(X %*% c(0.5, 0.3, -0.2)))
  V_online <- vcov(rls_glm_fit(X, y, family = poisson(),
                                beta_init = c(log(mean(y)), rep(0, p - 1)),
                                score_clip = 5 * sqrt(mean(y))))
  V_glm    <- vcov(glm(y ~ X[, -1], family = poisson()))
  expect_equal(unname(V_online), unname(V_glm), tolerance = 0.15)
})

test_that("Poisson: summary(rls_glm) matches summary(glm) coefficient table", {
  set.seed(323)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1), sd = 0.5), n, p - 1))
  y <- rpois(n, exp(X %*% c(0.5, 0.3, -0.2)))
  s_online <- summary(rls_glm_fit(X, y, family = poisson(),
                                   beta_init = c(log(mean(y)), rep(0, p - 1)),
                                   score_clip = 5 * sqrt(mean(y))))
  s_glm    <- summary(glm(y ~ X[, -1], family = poisson()))
  se_online <- s_online$coefficients[, "Std. Error"]
  se_glm    <- s_glm$coefficients[, "Std. Error"]
  expect_true(all(abs(se_online - se_glm) / se_glm < 0.15))
  ## Both use z-values (Poisson has known dispersion = 1)
  expect_true("z value" %in% colnames(s_online$coefficients))
  expect_true("z value" %in% colnames(s_glm$coefficients))
})

test_that("Poisson: confint(rls_glm) close to confint.default(glm)", {
  set.seed(324)
  n <- 5000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1), sd = 0.5), n, p - 1))
  y <- rpois(n, exp(X %*% c(0.5, 0.3, -0.2)))
  ci_online <- confint(rls_glm_fit(X, y, family = poisson(),
                                    beta_init = c(log(mean(y)), rep(0, p - 1)),
                                    score_clip = 5 * sqrt(mean(y))),
                       level = 0.95)
  ci_glm    <- confint.default(glm(y ~ X[, -1], family = poisson()),
                               level = 0.95)
  expect_equal(unname(ci_online), unname(ci_glm), tolerance = 0.10)
})


# -------------------------------------------------------------------
# Cross-method Gaussian: rls == rls_glm(gaussian) == lm == glm
# -------------------------------------------------------------------

test_that("Gaussian four-way: rls == rls_glm(gaussian), both close to lm/glm", {
  set.seed(331)
  n <- 2000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(1, -0.5, 0.8) + rnorm(n)

  fit_rls     <- rls_fit(X, y, S0_scale = 1e6)
  fit_rls_glm <- rls_glm_fit(X, y, family = gaussian(), S0_scale = 1e6)
  fit_lm      <- lm(y ~ X[, -1])
  fit_glm     <- glm(y ~ X[, -1])

  ## rls == rls_glm exactly
  expect_equal(fit_rls$beta, fit_rls_glm$beta, tolerance = 1e-10)
  expect_equal(fit_rls$S, fit_rls_glm$S, tolerance = 1e-10)

  ## lm == glm (R source lm-tests.R pattern)
  expect_equal(as.numeric(coef(fit_lm)), as.numeric(coef(fit_glm)),
               tolerance = 1e-10)

  ## rls close to lm
  expect_equal(as.numeric(coef(fit_rls)), as.numeric(coef(fit_lm)),
               tolerance = 1e-4)
  expect_equal(unname(vcov(fit_rls)), unname(vcov(fit_lm)), tolerance = 0.05)
})


# -------------------------------------------------------------------
# Streaming RLS-GLM: vcov identical after batch + update vs full batch
# -------------------------------------------------------------------

test_that("streaming RLS-GLM vcov matches full-batch RLS-GLM vcov", {
  set.seed(341)
  n <- 200; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rbinom(n, 1, plogis(X %*% c(0.5, -1)))

  fit_full <- rls_glm_fit(X, y, family = binomial())

  n_init <- 120
  fit <- rls_glm_fit(X[1:n_init, ], y[1:n_init], family = binomial())
  for (i in (n_init + 1):n)
    fit <- update(fit, x = X[i, ], y = y[i])

  expect_equal(fit$beta, fit_full$beta, tolerance = 1e-10)
  expect_equal(fit$S, fit_full$S, tolerance = 1e-10)
  expect_equal(vcov(fit), vcov(fit_full), tolerance = 1e-10)
})


# ===================================================================
# INTERNAL CONSISTENCY, ACCUMULATOR CORRECTNESS, AND EDGE CASES
# ===================================================================


# -------------------------------------------------------------------
# 1. RLS summary$coefficients == lm summary$coefficients
#    (analog of: all.equal(summary(roller.lm0)$coefficients,
#                          summary(roller.lm9)$coefficients) )
# -------------------------------------------------------------------

test_that("rls_fit and rls_glm_fit(gaussian) give same beta and S", {
  set.seed(205)
  n <- 300; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(1, -0.5, 0.8) + rnorm(n)

  fit_rls <- rls_fit(X, y, S0_scale = 100)
  fit_glm <- rls_glm_fit(X, y, family = gaussian(), S0_scale = 100)

  expect_equal(fit_rls$beta, fit_glm$beta, tolerance = 1e-10)
  expect_equal(fit_rls$S, fit_glm$S, tolerance = 1e-10)
})


# -------------------------------------------------------------------
# 5. Dispersion convention: z vs t follows summary.glm rules
#    Known dispersion (binomial, Poisson) → z; estimated (Gaussian) → t
#    (analog of the est.disp logic in summary.glm lines 650-693)
# -------------------------------------------------------------------

test_that("Gaussian RLS uses t-values and pt() (estimated dispersion)", {
  set.seed(206)
  n <- 200; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rnorm(n)

  s <- summary(rls_fit(X, y))
  expect_equal(colnames(s$coefficients),
               c("Estimate", "Std. Error", "t value", "Pr(>|t|)"))
  ## Verify p-values are from t-distribution with n-p df
  df <- n - p
  expected_pval <- 2 * pt(-abs(s$coefficients[, "t value"]), df = df)
  expect_equal(unname(s$coefficients[, "Pr(>|t|)"]),
               unname(expected_pval), tolerance = 1e-12)
})

test_that("binomial RLS-GLM uses z-values and pnorm() (known dispersion)", {
  set.seed(207)
  n <- 500; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rbinom(n, 1, 0.5)

  s <- summary(rls_glm_fit(X, y, family = binomial()))
  expect_equal(colnames(s$coefficients),
               c("Estimate", "Std. Error", "z value", "Pr(>|z|)"))
  expected_pval <- 2 * pnorm(-abs(s$coefficients[, "z value"]))
  expect_equal(unname(s$coefficients[, "Pr(>|z|)"]),
               unname(expected_pval), tolerance = 1e-12)
})

test_that("iSGD uses z-values and pnorm() for binomial", {
  set.seed(208)
  n <- 1000; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rbinom(n, 1, plogis(X %*% c(0.5, -1)))

  s <- summary(isgd_fit(X, y, family = binomial(), compute_vcov = TRUE))
  expect_equal(colnames(s$coefficients),
               c("Estimate", "Std. Error", "z value", "Pr(>|z|)"))
  expected_pval <- 2 * pnorm(-abs(s$coefficients[, "z value"]))
  expect_equal(unname(s$coefficients[, "Pr(>|z|)"]),
               unname(expected_pval), tolerance = 1e-12)
})


# -------------------------------------------------------------------
# 6. Internal consistency: SE == sqrt(diag(vcov))
#    (analog of how summary.glm builds s.err from var.cf from covmat)
# -------------------------------------------------------------------

test_that("summary SEs == sqrt(diag(vcov)) for all three methods", {
  set.seed(209)
  n <- 500; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))

  ## RLS
  y_gauss <- X %*% c(1, -0.5, 0.8) + rnorm(n)
  fit_rls <- rls_fit(X, y_gauss)
  s <- summary(fit_rls)
  expect_equal(unname(s$coefficients[, "Std. Error"]),
               unname(sqrt(diag(vcov(fit_rls)))), tolerance = 1e-12)

  ## RLS-GLM
  y_bin <- rbinom(n, 1, plogis(X %*% c(-0.5, 1, -0.8)))
  fit_glm <- rls_glm_fit(X, y_bin, family = binomial())
  s2 <- summary(fit_glm)
  expect_equal(unname(s2$coefficients[, "Std. Error"]),
               unname(sqrt(diag(vcov(fit_glm)))), tolerance = 1e-12)

  ## iSGD
  fit_isgd <- isgd_fit(X, y_bin, family = binomial(), compute_vcov = TRUE)
  s3 <- summary(fit_isgd)
  expect_equal(unname(s3$coefficients[, "Std. Error"]),
               unname(sqrt(diag(vcov(fit_isgd)))), tolerance = 1e-12)
})


# -------------------------------------------------------------------
# 7. confint == beta +/- q * SE, and consistent with vcov
#    (analog of: confint.default applied to glm)
# -------------------------------------------------------------------

test_that("confint matches confint.default logic for RLS-GLM (binomial)", {
  set.seed(210)
  n <- 2000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rbinom(n, 1, plogis(X %*% c(-0.5, 1, -0.8)))

  fit <- rls_glm_fit(X, y, family = binomial())
  ci  <- confint(fit, level = 0.95)
  se  <- sqrt(diag(vcov(fit)))
  q   <- qnorm(0.975)

  ## Lower and upper bounds must equal beta +/- q*se exactly
  expect_equal(unname(ci[, 1]), fit$beta - q * se, tolerance = 1e-12)
  expect_equal(unname(ci[, 2]), fit$beta + q * se, tolerance = 1e-12)
})

test_that("confint uses qt (not qnorm) for RLS Gaussian", {
  set.seed(211)
  n <- 100; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rnorm(n)

  fit <- rls_fit(X, y)
  ci  <- confint(fit, level = 0.95)
  se  <- sqrt(diag(vcov(fit)))
  q_t <- qt(0.975, df = n - p)
  q_z <- qnorm(0.975)

  ## Must use t, which is wider than z
  expect_equal(unname(ci[, 1]), fit$beta - q_t * se, tolerance = 1e-12)
  expect_true(q_t > q_z)  # t quantile > z quantile for finite df
})


# -------------------------------------------------------------------
# 8. Streaming == batch: identical results for deterministic methods
#    (analog of the R source principle: different code paths, same answer)
# -------------------------------------------------------------------

test_that("streaming RLS (partial + update) gives identical vcov to full fit", {
  set.seed(212)
  n <- 100; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rnorm(n)

  fit_full <- rls_fit(X, y)

  n_init <- 60
  fit <- rls_fit(X[1:n_init, ], y[1:n_init])
  for (i in (n_init + 1):n)
    fit <- update(fit, x = X[i, ], y = y[i])

  expect_equal(fit$beta, fit_full$beta, tolerance = 1e-10)
  expect_equal(fit$S,    fit_full$S,    tolerance = 1e-10)
  expect_equal(fit$rss,  fit_full$rss,  tolerance = 1e-10)
  expect_equal(fit$n_obs, fit_full$n_obs)
  expect_equal(vcov(fit), vcov(fit_full), tolerance = 1e-10)
})

test_that("streaming iSGD sandwich matches full fit exactly (Gaussian)", {
  ## For Gaussian the beta path is deterministic regardless of split point.
  set.seed(213)
  n <- 200; p <- 2
  X <- cbind(1, rnorm(n))
  y <- X %*% c(0.5, -1) + rnorm(n)

  fit_full <- isgd_fit(X, y, family = gaussian(), gamma1 = 0.5, alpha = 0.7,
                       compute_vcov = TRUE)

  n_init <- 120
  fit <- isgd_fit(X[1:n_init, ], y[1:n_init], family = gaussian(),
                  gamma1 = 0.5, alpha = 0.7, compute_vcov = TRUE)
  for (i in (n_init + 1):n)
    fit <- update(fit, x = X[i, ], y = y[i])

  expect_equal(fit$beta,  fit_full$beta,  tolerance = 1e-10)
  expect_equal(fit$A_hat, fit_full$A_hat, tolerance = 1e-10)
  expect_equal(fit$B_hat, fit_full$B_hat, tolerance = 1e-10)
  expect_equal(vcov(fit), vcov(fit_full), tolerance = 1e-10)
})

test_that("streaming RLS-GLM n_obs is correct after update", {
  set.seed(214)
  n <- 100; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rbinom(n, 1, 0.5)

  fit <- rls_glm_fit(X[1:60, ], y[1:60], family = binomial())
  for (i in 61:n)
    fit <- update(fit, x = X[i, ], y = y[i])

  expect_equal(fit$n_obs, n)
  ## beta_path is NOT extended by update() (to avoid O(n^2) copy cost),
  ## so it retains the initial batch size.
  expect_equal(nrow(fit$beta_path), 60L)
})


# -------------------------------------------------------------------
# 9. Accumulator correctness: A_hat, B_hat, rss from first principles
#    (tests the accumulation loop, not vcov formula)
# -------------------------------------------------------------------

test_that("iSGD A_hat == X'X/n for Gaussian (w_i = 1 for all i)", {
  set.seed(215)
  n <- 1000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(1, -0.5, 0.8) + rnorm(n)

  fit <- isgd_fit(X, y, family = gaussian(), gamma1 = 0.5, alpha = 0.7,
                  compute_vcov = TRUE)

  ## For gaussian(): mu.eta = 1, variance = 1 ⟹ w = 1.
  ## A_hat = (1/n) sum_i x_i x_i' = X'X / n.
  expect_equal(fit$A_hat, crossprod(X) / n, tolerance = 1e-10)
})

test_that("iSGD B_hat matches manual score-outer-product accumulation", {
  ## Replay the fitting loop, compute (y_i - x_i'beta_pre)^2 * x_i x_i'
  ## manually from the stored beta path.  This catches bugs where the
  ## accumulator uses post-update instead of pre-update beta, or where
  ## the family functions are applied incorrectly.
  set.seed(216)
  n <- 200; p <- 2
  X <- cbind(1, rnorm(n))
  y <- X %*% c(0.5, -1) + rnorm(n)

  fit <- isgd_fit(X, y, family = gaussian(), gamma1 = 0.5, alpha = 0.7,
                  compute_vcov = TRUE)

  B_manual <- matrix(0, p, p)
  beta_prev <- numeric(p)
  for (i in seq_len(n)) {
    x_i     <- X[i, ]
    score_i <- y[i] - sum(x_i * beta_prev)
    B_manual <- B_manual + score_i^2 * tcrossprod(x_i)
    beta_prev <- fit$beta_path[i, ]
  }
  B_manual <- B_manual / n

  expect_equal(fit$B_hat, B_manual, tolerance = 1e-10)
})

test_that("iSGD B_hat manual replay for binomial (non-trivial family)", {
  ## Same idea but with binomial, where mu.eta and variance are nontrivial.
  set.seed(217)
  n <- 300; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rbinom(n, 1, plogis(X %*% c(0.5, -1)))
  fam <- binomial()

  fit <- isgd_fit(X, y, family = fam, gamma1 = 1, alpha = 0.7,
                  compute_vcov = TRUE)

  A_manual <- matrix(0, p, p)
  B_manual <- matrix(0, p, p)
  beta_prev <- numeric(p)
  for (i in seq_len(n)) {
    x_i   <- X[i, ]
    eta_i <- sum(x_i * beta_prev)
    mu_i  <- fam$linkinv(eta_i)
    dmu_i <- fam$mu.eta(eta_i)
    V_i   <- fam$variance(mu_i)
    w_i   <- dmu_i^2 / V_i
    score_i <- (y[i] - mu_i) * dmu_i / V_i
    xx_i    <- tcrossprod(x_i)
    A_manual <- A_manual + w_i * xx_i
    B_manual <- B_manual + score_i^2 * xx_i
    beta_prev <- fit$beta_path[i, ]
  }
  A_manual <- A_manual / n
  B_manual <- B_manual / n

  expect_equal(fit$A_hat, A_manual, tolerance = 1e-10)
  expect_equal(fit$B_hat, B_manual, tolerance = 1e-10)
})

test_that("RLS rss matches sum of post-update squared residuals", {
  set.seed(218)
  n <- 100; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rnorm(n)

  fit <- rls_fit(X, y, S0_scale = 1e6)

  ## Recompute RSS from the beta path: at step i the post-update beta is
  ## beta_path[i,], so the post-update residual is y[i] - x_i' beta_path[i,].
  rss_manual <- 0
  for (i in seq_len(n))
    rss_manual <- rss_manual + (y[i] - sum(X[i, ] * fit$beta_path[i, ]))^2

  expect_equal(fit$rss, rss_manual, tolerance = 1e-10)
})


# -------------------------------------------------------------------
# 10. iSGD sandwich SEs same order of magnitude as batch glm SEs
#     (not an exact match — the sandwich converges slower than Fisher)
# -------------------------------------------------------------------

test_that("iSGD sandwich SEs within factor of 3 of batch glm SEs", {
  set.seed(219)
  n <- 10000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  beta_true <- c(-0.5, 1, -0.8)
  y <- rbinom(n, 1, plogis(X %*% beta_true))

  se_isgd <- sqrt(diag(vcov(
    isgd_fit(X, y, family = binomial(), gamma1 = 1, alpha = 0.7,
             compute_vcov = TRUE)
  )))
  se_batch <- sqrt(diag(vcov(glm(y ~ X[, -1], family = binomial()))))

  ratio <- se_isgd / se_batch
  expect_true(all(ratio > 1/3 & ratio < 3),
              info = paste("SE ratios:", paste(round(ratio, 3), collapse = ", ")))
})


# -------------------------------------------------------------------
# 11. vcov SPD and symmetric for all methods
# -------------------------------------------------------------------

test_that("vcov is symmetric positive definite for all three methods", {
  set.seed(220)
  n <- 500; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))

  check_spd <- function(V, label) {
    expect_equal(V, t(V), tolerance = 1e-12, info = paste(label, "not symmetric"))
    evals <- eigen(V, symmetric = TRUE, only.values = TRUE)$values
    expect_true(all(evals > 0), info = paste(label, "not positive definite"))
  }

  y_gauss <- X %*% c(1, -0.5, 0.8) + rnorm(n)
  check_spd(vcov(rls_fit(X, y_gauss)), "RLS")

  y_bin <- rbinom(n, 1, plogis(X %*% c(-0.5, 1, -0.8)))
  check_spd(vcov(rls_glm_fit(X, y_bin, family = binomial())), "RLS-GLM")
  check_spd(vcov(isgd_fit(X, y_bin, family = binomial(),
                           compute_vcov = TRUE)), "iSGD")
})


# -------------------------------------------------------------------
# 12. Edge cases and error handling
#     (analog of R source: PR#8720 zero-weight summary, PR#10494 confint
#      on rank-deficient, etc.)
# -------------------------------------------------------------------

test_that("vcov errors on iSGD fit without sandwich matrices", {
  set.seed(221)
  X <- cbind(1, rnorm(50))
  y <- rbinom(50, 1, 0.5)
  fit <- isgd_fit(X, y, family = binomial(), compute_vcov = FALSE)
  expect_error(vcov(fit), "sandwich matrices")
  expect_null(fit$A_hat)
  expect_null(fit$B_hat)
})

test_that("confint parm subsetting returns correct rows", {
  set.seed(222)
  n <- 200; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rnorm(n)

  fit <- rls_fit(X, y)
  ci_full <- confint(fit)
  ci_sub  <- confint(fit, parm = 2)
  expect_equal(nrow(ci_sub), 1L)
  expect_equal(ci_sub[1, ], ci_full[2, ])

  ci_multi <- confint(fit, parm = c(1, 3))
  expect_equal(nrow(ci_multi), 2L)
  expect_equal(ci_multi[1, ], ci_full[1, ])
  expect_equal(ci_multi[2, ], ci_full[3, ])
})

test_that("summary without vcov (iSGD, compute_vcov=FALSE) returns Estimate only", {
  set.seed(223)
  X <- cbind(1, rnorm(100))
  y <- rbinom(100, 1, 0.5)
  fit <- isgd_fit(X, y, family = binomial(), compute_vcov = FALSE)
  s <- summary(fit)
  expect_s3_class(s, "summary.stream_fit")
  expect_equal(ncol(s$coefficients), 1L)
  expect_equal(colnames(s$coefficients), "Estimate")
})

test_that("print.summary.stream_fit runs without error", {
  set.seed(224)
  fit <- rls_fit(cbind(1, rnorm(200)), rnorm(200))
  s <- summary(fit)
  expect_output(print(s), "Stream fit")
  expect_output(print(s), "RLS")
  expect_output(print(s), "200")
  expect_output(print(s), "Coefficients:")
})


# -------------------------------------------------------------------
# 13. CI coverage and width scaling
# -------------------------------------------------------------------

test_that("confint 99% CI covers true beta (RLS, n=2000)", {
  set.seed(225)
  n <- 2000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  beta_true <- c(1, -0.5, 0.8)
  y <- X %*% beta_true + rnorm(n, sd = 0.5)

  ci <- confint(rls_fit(X, y, S0_scale = 1e6), level = 0.99)
  for (j in seq_len(p))
    expect_true(ci[j, 1] < beta_true[j] && beta_true[j] < ci[j, 2],
                info = paste("beta[", j, "] =", beta_true[j],
                             "not in [", round(ci[j, 1], 4), ",",
                             round(ci[j, 2], 4), "]"))
})

test_that("confint width ratio approximates sqrt(n1/n2) for 10x more data", {
  set.seed(226)
  p <- 2; beta_true <- c(0.5, -1)

  make_fit <- function(nn) {
    X <- cbind(1, rnorm(nn))
    y <- X %*% beta_true + rnorm(nn)
    rls_fit(X, y, S0_scale = 1e6)
  }

  w1 <- diff(t(confint(make_fit(500))))
  w2 <- diff(t(confint(make_fit(5000))))

  expected_ratio <- sqrt(5000 / 500)
  actual_ratios  <- w1 / w2
  expect_true(all(abs(actual_ratios - expected_ratio) / expected_ratio < 0.20),
              info = paste("width ratios:", paste(round(actual_ratios, 2),
                           collapse = ", "),
                           "expected ~", round(expected_ratio, 2)))
})
