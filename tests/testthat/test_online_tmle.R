## Tests for online_tmle_fit() and update.online_tmle()
##
## Reference implementations consulted:
##   - tlverse/tmle3, tests/testthat/test-ATE.R,
##     https://github.com/tlverse/tmle3/blob/master/tests/testthat/test-ATE.R
##     Strategy: ATE psi and SE match the classic `tmle` package within 1/sqrt(n).
##   - tlverse/tmle3, tests/testthat/test-ATT.R,
##     https://github.com/tlverse/tmle3/blob/master/tests/testthat/test-ATT.R
##     Strategy: ATT estimates come from the ATT slot (not ATE slot) of the
##     reference; ATE and ATT are implicitly shown to differ.
##   - tlverse/tmle3, tests/testthat/test-basic_intevention.R,
##     https://github.com/tlverse/tmle3/blob/master/tests/testthat/test-basic_intevention.R
##     Strategy: Qstar, psi, SE, and epsilon all compared against the classic
##     `tmle` package in the same test — the most complete targeting-step test.
##     Epsilon extracted as `updater$epsilons[[1]]$Y`.
##   - cran/ltmle, tests/testthat/test-EffectMeasures.R,
##     https://github.com/cran/ltmle/blob/master/tests/testthat/test-EffectMeasures.R
##     Strategy: ATE SE verified as sqrt(Var(IC_1 - IC_0) / n) where IC is the
##     influence curve / EIF contribution — direct oracle for our Welford SE.
##   - cran/ltmle, tests/testthat/test-EstimateVariance.R,
##     https://github.com/cran/ltmle/blob/master/tests/testthat/test-EstimateVariance.R
##     Strategy: TMLE variance > IC variance under near-positivity-violation;
##     ordering of variance estimators under degenerate propensity.
##   - DoubleML/doubleml-for-r, tests/testthat/helper-10-dml_irm.R,
##     https://github.com/DoubleML/doubleml-for-r
##     Strategy: exact numerical agreement between package output and a hand-coded
##     Neyman-orthogonal score (our oracle in Test 2).
##
## Step 2 — Classification of reference tests:
##
##   | Test from references                        | Class                | Keep |
##   |---------------------------------------------|----------------------|------|
##   | psi matches classic tmle package (1/sqrt(n))| Empirical / oracle   | Yes  |
##   | SE matches classic tmle package              | Oracle / asymptotic  | Yes  |
##   | epsilon matches classic tmle package         | Oracle               | Yes  |
##   | Qstar matches classic tmle Qstar             | Oracle               | Yes  |
##   | g bounds enforced pre/post targeting         | Edge case            | Yes  |
##   | pooled ATE = weighted mean of strata ATEs    | Theoretical identity | No   |
##   | ATE SE = sqrt(Var(IC_1 - IC_0) / n)         | Oracle               | Yes  |
##   | TMLE var > IC var under positivity violation | Empirical            | Yes  |
##
## Properties tested (with reference):
##   1. Batch score equation satisfied exactly for Gaussian outcome.
##      [ltmle targeting guarantee; tmle3/test-basic_intevention.R]
##   2. psi and SE equal the manual mean and SD of targeted EIF to 1e-8.
##      [DoubleML oracle; ltmle/test-EffectMeasures.R SE formula]
##   3. TMLE psi converges to true ATE with tolerance 0.15.
##      [tmle3/test-ATE.R 1/sqrt(n) tolerance strategy]
##   4. ATE and ATT differ under confounding with HTE.
##      [tmle3/test-ATT.R ATE vs ATT distinction]
##   5. epsilon is small when Q is correctly specified.
##      [tmle3/test-basic_intevention.R epsilon diagnostic]
##
## Step 5 audit:
##   1. Optimiser / gradient direction: Test 1 (wrong linkfun or sign →
##      |score| >> 0) and Test 3 (wrong epsilon sign → psi diverges).
##   2. Prediction function: Test 2 recomputes predictions from the stored
##      Q_sl / g_sl — wrong ensemble weights → manual EIF ≠ package psi.
##   3. Edge case: ## TODO: add propensity clamping test (near-deterministic g)
##      analogous to test_online_ate.R Test 5.
##   4. Family / link function: Tests 1 and 2 exercise the Gaussian linkfun
##      (identity). ## TODO: add a binomial-outcome test to exercise qlogis
##      offset and expit Q* in the targeting step.


## ---- Shared fixture --------------------------------------------------------

make_lib_rls <- function() {
  list(
    rls = make_learner(
      fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
      predict = function(model, newdata, ...) predict(model, newdata),
      update  = function(model, x, y, family, ...) update(model, x = x, y = y)
    )
  )
}


## ---- 1. Batch score equation solved exactly --------------------------------
## After online_tmle_fit() with a Gaussian outcome, the 1-D targeting
## regression is equivalent to exact OLS (RLS with lambda = 1 and identity
## link).  The normal equations guarantee mean(H * (Y - Q*a)) = 0 exactly.
##
## Inspired by: tlverse/tmle3/test-basic_intevention.R (Q* and epsilon match
## classic tmle, which also satisfies the score equation by construction) and
## the ltmle targeting guarantee (Gruber & van der Laan, 2012).
##
## Would fail if: linkfun(Q) offset is omitted from the 1-D regression so that
## epsilon absorbs the raw mean of Y rather than a residual correction, making
## Q*a = linkinv(eps*H) instead of linkinv(linkfun(Q) + eps*H).

test_that("batch score equation |mean(H*(Y-Q*a))| < 1e-8 for Gaussian outcome", {
  set.seed(11L)
  n <- 300L
  W <- cbind(1, matrix(rnorm(n * 2L), n, 2L))
  A <- rbinom(n, 1L, plogis(0.4 * W[, 2L]))
  Y <- 1.0 * A + 0.5 * W[, 2L] + rnorm(n, sd = 0.5)

  lib  <- make_lib_rls()
  tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib)

  eps <- tmle$epsilon
  fam <- tmle$family

  Q1  <- predict(tmle$Q_sl, cbind(1, W))
  Q0  <- predict(tmle$Q_sl, cbind(0, W))
  Qa  <- predict(tmle$Q_sl, cbind(A, W))
  g_c <- pmin(pmax(predict(tmle$g_sl, W), tmle$g_clamp), 1 - tmle$g_clamp)

  H      <- A / g_c - (1 - A) / (1 - g_c)
  Qa_str <- fam$linkinv(fam$linkfun(Qa) + eps * H)

  expect_lt(abs(mean(H * (Y - Qa_str))), 1e-8)
})


## ---- 2. TMLE psi and SE equal manual targeted EIF --------------------------
## Oracle check: re-derive psi and SE using stored nuisance models and epsilon.
## Matches the strategy in DoubleML helper-10-dml_irm.R (exact hand-coded EIF
## agreement) and ltmle/test-EffectMeasures.R (SE = sqrt(Var(IC)/n)).
##
## Would fail if: H1 uses H0's formula (-1/(1-g) instead of 1/g), making Q1*
## drift away from the correctly targeted value so that mean(EIF_manual)
## diverges from the stored psi.

test_that("psi and SE equal the manual mean and SD of targeted EIF to 1e-8", {
  set.seed(77L)
  n <- 300L
  W <- cbind(1, matrix(rnorm(n * 2L), n, 2L))
  A <- rbinom(n, 1L, plogis(0.5 * W[, 2L]))
  Y <- 1.5 * A + 0.8 * W[, 2L] + rnorm(n, sd = 0.5)

  lib  <- make_lib_rls()
  tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib)

  eps <- tmle$epsilon
  fam <- tmle$family

  Q1  <- predict(tmle$Q_sl, cbind(1, W))
  Q0  <- predict(tmle$Q_sl, cbind(0, W))
  Qa  <- predict(tmle$Q_sl, cbind(A, W))
  g_c <- pmin(pmax(predict(tmle$g_sl, W), tmle$g_clamp), 1 - tmle$g_clamp)

  H      <- A / g_c - (1 - A) / (1 - g_c)
  H1     <-  1 / g_c
  H0     <- -1 / (1 - g_c)
  Qa_str <- fam$linkinv(fam$linkfun(Qa) + eps * H)
  Q1_str <- fam$linkinv(fam$linkfun(Q1) + eps * H1)
  Q0_str <- fam$linkinv(fam$linkfun(Q0) + eps * H0)

  ## Hand-coded TMLE EIF (ltmle/test-EffectMeasures.R IC formula).
  eif_manual <- H * (Y - Qa_str) + Q1_str - Q0_str

  expect_equal(as.numeric(coef(tmle)), mean(eif_manual),          tolerance = 1e-8)
  expect_equal(sqrt(tmle$eif_var / n), sd(eif_manual) / sqrt(n), tolerance = 1e-8)
})


## ---- 3. TMLE psi converges to true ATE within 1/sqrt(n) tolerance ---------
## Same DGP as test_online_ate.R Test 1 (seed 42, n = 500 batch + 300 stream,
## true ATE = 1).  Tolerance 0.15 mirrors the tmle3/test-ATE.R strategy of
## comparing within 1/sqrt(n) ≈ 0.045 at n = 500; the total n = 800 gives
## 1/sqrt(800) ≈ 0.035, so 0.15 is a conservative 4-sigma bound.
##
## Would fail if: epsilon is updated with the wrong sign during the streaming
## phase so the TMLE EIF accumulates a systematic bias opposite to the truth.

test_that("TMLE psi converges to true ATE = 1 within tolerance 0.15", {
  set.seed(42L)
  n <- 500L; p <- 2L
  W <- cbind(1, matrix(rnorm(n * p), n, p))
  A <- rbinom(n, 1L, plogis(0.4 * W[, 2L]))
  Y <- 1.0 * A + 0.5 * W[, 2L] + rnorm(n, sd = 0.5)

  lib  <- make_lib_rls()
  tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib)

  set.seed(7L)
  for (i in seq_len(300L)) {
    w_i <- c(1, rnorm(p))
    a_i <- rbinom(1L, 1L, plogis(0.4 * w_i[2L]))
    y_i <- 1.0 * a_i + 0.5 * w_i[2L] + rnorm(1L, sd = 0.5)
    tmle <- update(tmle, w = w_i, a = a_i, y = y_i)
  }

  expect_lt(abs(coef(tmle) - 1.0), 0.15)
})


## ---- 4. Target dispatch: ATE ≠ ATT under confounding with HTE -------------
## DGP with effect modification: Y = A*(1 + W2) + N(0, 0.5),
## W2 ~ N(0, 1), A ~ Bernoulli(plogis(1.5*W2)).
## True ATE = E[1 + W2] = 1; True ATT = E[1 + W2 | A = 1] > 1.
## Mirrors tlverse/tmle3/test-ATT.R which verifies ATT comes from the ATT
## (not ATE) slot of the reference package.
##
## Would fail if: the ATT normalisation by sum_A is dropped, or the target
## argument is ignored and the ATE EIF is always used regardless of `target`.

test_that("TMLE ATE and ATT differ by more than 2*SE under confounding with HTE", {
  set.seed(2L)
  n <- 600L
  W <- cbind(1, rnorm(n))
  A <- rbinom(n, 1L, plogis(1.5 * W[, 2L]))
  Y <- A * (1.0 + W[, 2L]) + rnorm(n, sd = 0.5)

  lib     <- make_lib_rls()
  ate_obj <- online_tmle_fit(W, A, Y,
                             Q_library = lib, g_library = lib,
                             target    = "ATE")
  att_obj <- online_tmle_fit(W, A, Y,
                             Q_library = lib, g_library = lib,
                             target    = "ATT")

  psi_ate <- coef(ate_obj)
  psi_att <- coef(att_obj)
  se_ate  <- sqrt(ate_obj$eif_var / ate_obj$n_obs)

  expect_gt(abs(psi_att - psi_ate), 2 * se_ate)
})


## ---- 5. epsilon is small when Q is correctly specified ---------------------
## With a correctly specified linear outcome model (Gaussian RLS-GLM on the
## true features), the initial Q already satisfies the score equation
## approximately.  The targeting correction should be negligible: |epsilon| < 0.5.
## Inspired by tmle3/test-basic_intevention.R where epsilon is extracted and
## compared to the classic tmle package (which also gives near-zero epsilon for
## well-specified nuisance models).
##
## Would fail if: the linkfun(Q) offset is omitted from the epsilon regression
## so epsilon absorbs mean(Y - 0) instead of the residual mean(Y - Q),
## producing a large spurious correction even when Q is correct.

test_that("epsilon is small when Q is correctly specified", {
  set.seed(55L)
  n <- 500L
  W <- cbind(1, matrix(rnorm(n * 2L), n, 2L))
  A <- rbinom(n, 1L, plogis(0.4 * W[, 2L]))
  ## Linear Gaussian DGP: the RLS-GLM outcome model is correctly specified.
  Y <- 1.0 * A + 0.5 * W[, 2L] + rnorm(n, sd = 0.5)

  lib  <- make_lib_rls()
  tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib)

  expect_lt(abs(tmle$epsilon), 0.5)
})
