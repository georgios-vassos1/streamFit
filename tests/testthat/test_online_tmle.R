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
##     `tmle` package. Epsilon extracted as `updater$epsilons[[1]]$Y`. Iterative
##     loop runs up to maxit=100; convergence via |P_n D*| <= SE(D*)/min(log n, 10).
##   - tlverse/tmle3, R/tmle3_Update.R (convergence criterion implementation),
##     https://github.com/tlverse/tmle3/blob/master/R/tmle3_Update.R
##     Convergence modes: "scaled_var" (|ED| <= SE/log(n)) and "sample_size"
##     (|ED| <= 1/n). No active test verifies these thresholds explicitly.
##   - cran/ltmle, R/ltmle.R (UpdateQ + FixScoreEquation),
##     https://github.com/cran/ltmle/blob/master/R/ltmle.R
##     ltmle checks |sum(IC)| / mean(Y) < 0.001 after each GLM targeting step;
##     triggers nlminb fallback with abs.tol=1e-8 if threshold exceeded. No active
##     test of this criterion exists (commented-out expect_error in test-ErrorHandling.R).
##   - cran/ltmle, tests/testthat/test-EffectMeasures.R,
##     https://github.com/cran/ltmle/blob/master/tests/testthat/test-EffectMeasures.R
##     Strategy: ATE SE verified as sqrt(Var(IC_1 - IC_0) / n).
##   - DoubleML/doubleml-for-r, tests/testthat/helper-10-dml_irm.R,
##     https://github.com/DoubleML/doubleml-for-r
##     Strategy: exact numerical agreement between package output and a hand-coded
##     Neyman-orthogonal score (our oracle in Test 2).
##
## Step 2 — Classification of reference tests:
##
##   | Test from references                           | Class                | Keep |
##   |------------------------------------------------|----------------------|------|
##   | psi matches classic tmle package (1/sqrt(n))   | Empirical / oracle   | Yes  |
##   | SE matches classic tmle package                | Oracle / asymptotic  | Yes  |
##   | epsilon matches classic tmle package           | Oracle               | Yes  |
##   | Qstar matches classic tmle Qstar               | Oracle               | Yes  |
##   | g bounds enforced pre/post targeting           | Edge case            | Yes  |
##   | pooled ATE = weighted mean of strata ATEs      | Theoretical identity | No   |
##   | ATE SE = sqrt(Var(IC_1 - IC_0) / n)           | Oracle               | Yes  |
##   | TMLE var > IC var under positivity violation   | Empirical            | Yes  |
##   | ltmle relative score < 0.001 (commented out)  | Edge case            | Yes  |
##   | tmle3 convergence: |ED| <= SE/log(n)           | Oracle               | Yes  |
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
##   6. Logistic outcome: score equation satisfied after iterative targeting.
##      [ltmle UpdateQ relative score < 0.001; tmle3 |ED| <= SE/log(n);
##       no active reference test — gap identified in Step 2 classification]
##   7. max_iter=1 cannot converge to tight tol for logistic; iter_converged
##      reflects true score-equation status, not a hardcoded flag.
##      [cran/tmle ATT converged flag; ltmle commented-out FixScoreEquation test]
##
## Step 5 audit:
##   1. Optimiser / gradient direction: Test 1 (Gaussian score) and Test 6
##      (logistic score) both fail if eps_k is added with wrong sign or if
##      the iterative offset doesn't update from Q to Q*_k correctly.
##   2. Prediction function: Test 2 recomputes predictions from the stored
##      Q_sl / g_sl — wrong ensemble weights → manual EIF ≠ package psi.
##   3. Edge case: ## TODO: add propensity clamping test (near-deterministic g)
##      analogous to test_online_ate.R Test 5.
##   4. Family / link function: Test 1 exercises Gaussian (identity link);
##      Tests 6 and 7 exercise binomial (logit link) including qlogis offset
##      and expit Q* at each iteration.


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


## ---- 6. Logistic outcome: score equation satisfied after iterative targeting
## No active reference test for this property exists in tmle3 or ltmle (gap
## identified in Step 2 classification above).  ltmle's UpdateQ checks a
## relative score criterion |sum(IC)| / mean(Y) < 0.001 but it is not tested;
## tmle3's convergence criterion |ED| <= SE/log(n) is also untested.
##
## For a Gaussian outcome, 1-D RLS-GLM solves the score equation exactly in
## one pass (Test 1).  For a logistic outcome the score equation is nonlinear:
## mean(H * (Y - plogis(logit(Q) + eps * H))) = 0
## requires genuine iteration; a single RLS-GLM pass gives an online
## approximation that may not satisfy the equation to the required tolerance.
##
## Oracle check: recompute Q*a from stored epsilon and original (unadapted) Q
## using the cumulative-epsilon invariant
##   logit(Q*) = logit(Q) + epsilon * H
## and verify the score is below tol.
##
## Would fail if: the offset for iteration k+1 uses the original linkfun(Q)
## instead of linkfun(Q*_k), because for logistic the score equation is
## nonlinear and each iteration must chain from the previous Q*.

test_that("logistic score equation |mean(H*(Y-Q*a))| < tol after iterative targeting", {
  set.seed(33L)
  n <- 400L
  W <- cbind(1, matrix(rnorm(n * 2L), n, 2L))
  A <- rbinom(n, 1L, plogis(0.5 * W[, 2L]))
  ## Binary DGP: logistic outcome family.
  Y <- rbinom(n, 1L, plogis(1.2 * A + 0.8 * W[, 2L]))

  ## RLS-GLM is a sequential online algorithm; for logistic regression it gives
  ## an online approximation (not the exact batch MLE).  The achievable score
  ## floor after many iterations is ~1e-5 for n=400.  tol=1e-4 is comfortably
  ## above this floor while still meaningfully below a naive one-step score.
  lib <- make_lib_rls()
  tol <- 1e-4
  tmle <- online_tmle_fit(W, A, Y,
                          Q_library      = lib,
                          g_library      = lib,
                          outcome_family = stats::binomial(),
                          max_iter       = 20L,
                          tol            = tol)

  ## Verify convergence flag was set.
  expect_true(tmle$iter_converged)

  ## Recompute score independently from stored epsilon and original (unadapted) Q.
  ## Cumulative-epsilon invariant: logit(Q*a) = logit(Qa) + epsilon * H.
  fam <- tmle$family
  Qa  <- predict(tmle$Q_sl, cbind(A, W))
  g_c <- pmin(pmax(predict(tmle$g_sl, W), tmle$g_clamp), 1 - tmle$g_clamp)
  H   <- A / g_c - (1 - A) / (1 - g_c)   # ATE clever covariate

  Qa_str <- fam$linkinv(fam$linkfun(Qa) + tmle$epsilon * H)
  score  <- mean(H * (Y - Qa_str))

  expect_lt(abs(score), tol)
})


## ---- 7. iter_converged reflects actual score; targeting reduces score vs eps=0
## tmle3's tmle3_Update runs up to maxit=100 but has no test confirming that
## iter_converged can be FALSE.  cran/tmle stores ATT$converged but never
## tests the FALSE case.  ltmle's FixScoreEquation convergence test is
## commented out (test-ErrorHandling.R).
##
## Same binary DGP as Test 6 (seed 33).  Empirical floor for RLS-GLM on this
## dataset: score ≈ 1.3e-5 after iteration 1, stable thereafter.
##   - tol=1e-10 (below floor): iter_converged=FALSE after 20 iterations.
##   - Targeting always reduces score vs eps=0 (score_before=0.00385,
##     score_after≈1.3e-5), regardless of convergence status.
##
## Would fail if: iter_converged is hardcoded to TRUE regardless of the actual
## score (convergence check tests abs(eps_k) < tol instead of abs(score) < tol,
## mistaking a tiny last epsilon step for score-equation satisfaction).

test_that("iter_converged=FALSE at unachievable tol; targeting reduces score vs eps=0", {
  set.seed(33L)
  n <- 400L
  W <- cbind(1, matrix(rnorm(n * 2L), n, 2L))
  A <- rbinom(n, 1L, plogis(0.5 * W[, 2L]))
  Y <- rbinom(n, 1L, plogis(1.2 * A + 0.8 * W[, 2L]))

  lib <- make_lib_rls()

  ## tol=1e-10 is below the RLS-GLM score floor (~1e-5) for this DGP;
  ## 20 iterations cannot converge → iter_converged must be FALSE.
  set.seed(33L)
  tmle <- online_tmle_fit(W, A, Y,
                          Q_library      = lib,
                          g_library      = lib,
                          outcome_family = stats::binomial(),
                          max_iter       = 20L,
                          tol            = 1e-10)
  expect_false(tmle$iter_converged)

  ## Regardless of convergence status, epsilon must REDUCE the score relative
  ## to the untargeted Q (eps=0).  Recompute both scores from stored nuisance.
  fam <- tmle$family
  Qa  <- predict(tmle$Q_sl, cbind(A, W))
  g_c <- pmin(pmax(predict(tmle$g_sl, W), tmle$g_clamp), 1 - tmle$g_clamp)
  H   <- A / g_c - (1 - A) / (1 - g_c)

  score_before <- abs(mean(H * (Y - Qa)))   # eps=0, no targeting
  score_after  <- abs(mean(H * (Y - fam$linkinv(fam$linkfun(Qa) + tmle$epsilon * H))))
  expect_lt(score_after, score_before)
})
