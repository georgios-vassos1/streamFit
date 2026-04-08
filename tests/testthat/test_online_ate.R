## Tests for online_ate_fit() and update.online_ate()
##
## Reference implementations consulted:
##   - tlverse/tmle3, tests/testthat/test-ATE.R and test-ATT.R,
##     https://github.com/tlverse/tmle3
##     Strategy: point estimates and SE match a reference package at 1/sqrt(n).
##   - yqzhong7/AIPW, tests/testthat/test-nuisance_input.R,
##     https://github.com/yqzhong7/AIPW
##     Strategy: verify the EIF computation path with user-supplied nuisance values.
##   - DoubleML/doubleml-for-r, tests/testthat/helper-10-dml_irm.R,
##     https://github.com/DoubleML/doubleml-for-r
##     Strategy: exact numerical agreement between the package and a hand-coded
##     reference EIF (used as the primary oracle for Test 2).
##   - DoubleML/doubleml-for-r, tests/testthat/test-double_ml_irm.R
##     Strategy: ATE and ATTE scores tested separately, tolerance 1e-8 vs
##     reference; inspired double-robustness DGP in Test 4.
##
## Properties tested:
##   1. Convergence to known ATE in a confounded linear DGP.
##      [tmle3 / AIPW / DoubleML all use controlled-DGP convergence checks]
##   2. psi and SE equal the manually computed mean and SD of EIF contributions
##      (formula correctness + Welford accumulator).
##      [DoubleML: exact agreement with hand-coded EIF in helper-10-dml_irm.R]
##   3. ATE and ATT estimates differ under confounding with effect modification.
##      [tmle3: ATE and ATT tested against reference at 1/sqrt(n)]
##   4. Double robustness: misspecified Q with correct g yields consistent ATE.
##      [van der Laan & Rose (2011) Sec. 2; not present in reference test suites
##       but is the core semiparametric guarantee of the estimator]
##   5. Propensity clamping yields a finite estimate under near-degenerate g.
##      [AIPW g.bound validation; DoubleML trimming_threshold tests]
##
## Step-5 audit:
##   1. Optimiser / gradient direction: Test 1 — wrong EIF sign → |psi−1| > 0.20.
##   2. Prediction feature order: Test 2 — cbind(1,W) vs cbind(W,1) gives
##      different Q1, so manual EIF diverges from package psi.
##   3. Edge case: Test 5 — |W2| = 10, near-deterministic treatment assignment.
##   4. Family / link function: g is hardcoded to binomial() in online_ate_fit;
##      outcome family exercised via gaussian() default in all tests.
##      TODO: add a binomial-outcome test to exercise the link function in Q.


## ---- Shared fixture --------------------------------------------------------

## A single RLS-GLM learner that supports online updates.
make_lib_rls <- function() {
  list(
    rls = make_learner(
      fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
      predict = function(model, newdata, ...) predict(model, newdata),
      update  = function(model, x, y, family, ...) update(model, x = x, y = y)
    )
  )
}


## ---- 1. Convergence --------------------------------------------------------
## Linear DGP: Y = 1*A + 0.5*W2 + N(0, 0.5), A ~ Bernoulli(plogis(0.4*W2)).
## True ATE = 1.  After n = 500 batch + 300 stream the estimate must satisfy
## |psi - 1| < 0.20.

## Would fail if: the EIF has the wrong sign, Q1-Q0 is omitted, or the H
## correction term contributes with incorrect magnitude.

test_that("ATE converges to true value 1 in a confounded linear DGP", {
  set.seed(42L)
  n <- 500L; p <- 2L
  W <- cbind(1, matrix(rnorm(n * p), n, p))
  A <- rbinom(n, 1L, plogis(0.4 * W[, 2L]))
  Y <- 1.0 * A + 0.5 * W[, 2L] + rnorm(n, sd = 0.5)

  lib <- make_lib_rls()
  ate <- online_ate_fit(W, A, Y, Q_library = lib, g_library = lib)

  set.seed(7L)
  for (i in seq_len(300L)) {
    w_i <- c(1, rnorm(p))
    a_i <- rbinom(1L, 1L, plogis(0.4 * w_i[2L]))
    y_i <- 1.0 * a_i + 0.5 * w_i[2L] + rnorm(1L, sd = 0.5)
    ate  <- update(ate, w = w_i, a = a_i, y = y_i)
  }

  expect_lt(abs(coef(ate) - 1.0), 0.20)
})


## ---- 2. EIF formula exactness (DoubleML oracle approach) -------------------
## Re-derive the ATE and its SE manually from the fitted nuisance models stored
## in the returned object.  The package psi must equal mean(eif_manual) to
## machine precision, and sqrt(eif_var / n) must equal sd(eif_manual) / sqrt(n).
##
## This mirrors the DoubleML strategy: the package output is compared against a
## hand-coded reference implementation of the Neyman-orthogonal score
## (helper-10-dml_irm.R: theta = mean(g1 - g0 + d*u1/m - (1-d)*u0/(1-m))).

## Would fail if: H uses the wrong sign (A/g replaced by g/A), the counterfactual
## feature matrices are constructed as cbind(W, 1) instead of cbind(1, W), the
## Welford mean has an off-by-one error, or eif_var is not the sample variance.

test_that("psi and SE equal the manual mean and SD of EIF contributions", {
  set.seed(99L)
  n <- 300L
  W <- cbind(1, matrix(rnorm(n * 2L), n, 2L))
  A <- rbinom(n, 1L, plogis(0.5 * W[, 2L]))
  Y <- 1.5 * A + 0.8 * W[, 2L] + rnorm(n, sd = 0.5)

  lib <- make_lib_rls()
  ate <- online_ate_fit(W, A, Y, Q_library = lib, g_library = lib)

  ## Hand-coded EIF using the nuisance models extracted from the object.
  ## Matches DoubleML formula: theta = mean(g1 - g0 + d*u1/m - (1-d)*u0/(1-m))
  ## where u1 = Y - g1, u0 = Y - g0.  Our Qa-based form is equivalent.
  Q1  <- predict(ate$Q_sl, cbind(1, W))
  Q0  <- predict(ate$Q_sl, cbind(0, W))
  Qa  <- predict(ate$Q_sl, cbind(A, W))
  g_c <- pmin(pmax(predict(ate$g_sl, W), ate$g_clamp), 1 - ate$g_clamp)
  H   <- A / g_c - (1 - A) / (1 - g_c)
  eif <- H * (Y - Qa) + Q1 - Q0

  expect_equal(as.numeric(coef(ate)), mean(eif),            tolerance = 1e-8)
  expect_equal(sqrt(ate$eif_var / ate$n_obs), sd(eif) / sqrt(n), tolerance = 1e-8)
})


## ---- 3. Target dispatch: ATE ≠ ATT under confounding -----------------------
## DGP with effect modification: Y = A*(1 + W2) + N(0, 0.5),
## W2 ~ N(0, 1), A ~ Bernoulli(plogis(1.5*W2)).
## True ATE = E[1 + W2] = 1;  True ATT = E[1 + W2 | A=1] > 1 (high-W2 units
## are more likely treated).  The two estimates must differ by > 2 * SE_ATE.

## Would fail if: update.online_ate ignores the target argument and always
## computes the ATE EIF, or the ATT normalisation by sum(A) is dropped.

test_that("ATE and ATT differ by more than 2*SE under confounding with HTE", {
  set.seed(2L)
  n <- 600L
  W <- cbind(1, rnorm(n))
  A <- rbinom(n, 1L, plogis(1.5 * W[, 2L]))
  Y <- A * (1.0 + W[, 2L]) + rnorm(n, sd = 0.5)

  lib <- make_lib_rls()
  ate_obj <- online_ate_fit(W, A, Y,
                            Q_library = lib, g_library = lib,
                            target    = "ATE")
  att_obj <- online_ate_fit(W, A, Y,
                            Q_library = lib, g_library = lib,
                            target    = "ATT")

  psi_ate <- coef(ate_obj)
  psi_att <- coef(att_obj)
  se_ate  <- sqrt(ate_obj$eif_var / ate_obj$n_obs)

  expect_gt(abs(psi_att - psi_ate), 2 * se_ate)
})


## ---- 4. Double robustness: misspecified Q, correct g -----------------------
## When the outcome model Q is deliberately misspecified (intercept-only, ignores
## W2) but the propensity score g is correctly specified (logistic RLS-GLM),
## the AIPW IPW correction restores consistency: psi → ATE via the
## Horvitz-Thompson identity mean(H*Y) = E[Y(1)] - E[Y(0)] under correct g.
##
## The misspecified Q uses column 2 of cbind(A, W) — the all-ones intercept
## column from W — so Q1 = Q0 = Qa ≈ mean(Y), and psi is driven entirely by
## the H*(Y - mean(Y)) correction term.

## Would fail if: the H*(Y-Qa) term is omitted from the EIF or multiplied by
## zero; or if psi = mean(Q1-Q0) only (no IPW correction).

test_that("ATE is consistent when Q is misspecified but g is correctly specified", {
  set.seed(15L)
  n <- 600L
  W <- cbind(1, rnorm(n))          # p = 2: intercept column + one covariate
  A <- rbinom(n, 1L, plogis(0.5 * W[, 2L]))
  Y <- 1.0 * A + 1.5 * W[, 2L] + rnorm(n, sd = 0.5)

  ## Intercept-only Q: X[, 2] is the all-ones column of cbind(A, W) because
  ## W already has an intercept in its first column.
  lib_Q_wrong <- list(
    intercept = make_learner(
      fit     = function(X, y, family, ...) rls_glm_fit(X[, 2L, drop = FALSE], y, family = family),
      predict = function(model, newdata, ...) predict(model, newdata[, 2L, drop = FALSE]),
      update  = function(model, x, y, family, ...) update(model, x = x[2L], y = y)
    )
  )
  ## Correctly specified g (logistic RLS-GLM on W).
  lib_g <- make_lib_rls()

  ate <- online_ate_fit(W, A, Y, Q_library = lib_Q_wrong, g_library = lib_g)

  expect_lt(abs(coef(ate) - 1.0), 0.30)
})


## ---- 5. Propensity clamping prevents Inf / NaN ----------------------------
## Introduce 20 observations with extreme covariate values (|W2| = 10) so
## that the predicted propensity is near 0 or 1 for those units.  Without
## clamping, H = A/g - (1-A)/(1-g) overflows.  With g_clamp = 0.01 the
## estimate and its streaming updates must remain finite.

## Would fail if: .g_clamp() is not applied to the g prediction before H is
## computed in either online_ate_fit or update.online_ate.

test_that("propensity clamping yields a finite estimate with near-degenerate g", {
  set.seed(6L)
  n <- 200L
  W <- cbind(1, c(rnorm(n - 20L), rep(10, 10L), rep(-10, 10L)))
  A <- rbinom(n, 1L, plogis(2.0 * W[, 2L]))
  Y <- 1.0 * A + W[, 2L] + rnorm(n, sd = 0.5)

  lib <- make_lib_rls()
  ate <- online_ate_fit(W, A, Y,
                        Q_library = lib, g_library = lib,
                        g_clamp   = 0.01)

  expect_true(is.finite(coef(ate)))

  ## Streaming updates with extreme observations must also stay finite.
  ate <- update(ate, w = c(1, 10), a = 1L, y = 5)
  ate <- update(ate, w = c(1, -10), a = 0L, y = -4)
  expect_true(is.finite(coef(ate)))
})
