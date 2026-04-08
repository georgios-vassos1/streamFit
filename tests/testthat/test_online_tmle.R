## Tests for online_tmle_fit() and update.online_tmle()
##
## Reference implementations consulted (Items 1–7 / original tests):
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
## Reference implementations consulted (Items 8–11 / stage-specific g and martingale CLT):
##   - achambaz/tsml.cara.rct,
##     https://github.com/achambaz/tsml.cara.rct
##     Reference CARA TMLE package: simulates adaptive RCTs where G_t changes at
##     each stage. No unit test suite; the package itself is the authoritative
##     CARA implementation. Test 8 mirrors its adaptive-design simulation.
##   - Chambaz & van der Laan (2011), Int. J. Biostatistics 7(1),
##     DOI: 10.2202/1557-4679.1247 (Theoretical) and 10.2202/1557-4679.1310 (Simulation)
##     Key result: TMLE is consistent and asymptotically Gaussian under CARA when H_i
##     uses the stage-i mechanism G_i. Section 4 simulation verifies convergence to
##     ATE = 1 — our oracle for Tests 3 and 8.
##   - Chambaz, Zheng & van der Laan (2017), Ann. Statist. 45(6): 2537–2564,
##     DOI: 10.1214/16-AOS1534
##     Martingale CLT for sequential TMLE; asymptotic variance V_t = (1/t) sum D*(P_hat_i)^2.
##     No active test suite (theoretical paper). Tests 9–10 verify the formula oracle.
##     Gap: no R package implements the martingale variance; Test 10 fills this gap.
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
##   | sequential g differs from pooled (CARA DGP)   | Empirical            | Yes  |
##   | eif_sq_sum = sum(EIF^2) batch oracle           | Oracle / exactness   | Yes  |
##   | SE_martingale uses uncentered 2nd moment       | Oracle               | Yes  |
##   | eif_sq_sum updates correctly in streaming      | Oracle / exactness   | Yes  |
##   | martingale SE ≈ i.i.d. SE for large n          | Asymptotic identity  | No   |
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
##   8. sequential_init=TRUE produces different EIF (and psi) from pooled g
##      under a two-phase CARA DGP (mechanism shift at obs 201).
##      [achambaz/tsml.cara.rct adaptive simulation; Chambaz & van der Laan 2011 §4]
##   9. eif_sq_sum = sum(EIF_i^2) in the batch phase to 1e-8.
##      [Chambaz, Zheng & van der Laan 2017 — V_t = (1/t) sum D*(P_hat_i)^2]
##  10. SE_martingale = sqrt(eif_sq_sum)/n; strictly greater than SE_iid
##      because eif_sq_sum/n^2 = eif_var/n + psi^2/n > eif_var/n for psi != 0.
##      [Chambaz, Zheng & van der Laan 2017 — no active test in reference packages]
##  11. eif_sq_sum is correctly incremented in update.online_tmle:
##      new_eif_sq_sum = old_eif_sq_sum + eif_stream^2 to machine precision.
##      [Chambaz, Zheng & van der Laan 2017 — online accumulation of V_t]
##
## Step 5 audit:
##   1. Optimiser / gradient direction: Test 1 (Gaussian score) and Test 6
##      (logistic score) both fail if eps_k is added with wrong sign or if
##      the iterative offset doesn't update from Q to Q*_k correctly.
##      Test 8 fails if the sequential g replay processes obs in wrong order.
##   2. Prediction function: Test 2 recomputes predictions from the stored
##      Q_sl / g_sl — wrong ensemble weights → manual EIF ≠ package psi.
##      Test 9 and Test 11 also verify the EIF computation chain.
##   3. Edge case: ## TODO: add propensity clamping test (near-deterministic g)
##      analogous to test_online_ate.R Test 5.
##      ## TODO: add sequential_init=TRUE with very small n (n near k_burn).
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


## ---- 8. Stage-specific g gives different EIF than pooled g (Item 2) --------
## DGP: CARA-like two-phase mechanism.  First 200 obs: A ~ Bern(plogis(0.2*W2));
## next 200 obs: A ~ Bern(plogis(1.5*W2)).  Both phases share the same Q DGP.
## Under sequential_init=TRUE the propensity score used for H_i is the model
## trained on obs 1..(i-1), capturing the phase-shift.  Under sequential_init=FALSE
## a pooled G fitted on all 400 obs is used, smearing both phases together.
## The two strategies give different weighted H_i values → different psi.
##
## References: achambaz/tsml.cara.rct (stage-specific allocation design);
##             Chambaz & van der Laan (2011) §4 simulation validates CARA TMLE.
##
## Would fail if: sequential_init=TRUE internally falls back to the pooled g_sl
## (e.g., the sequential replay loop is short-circuited), making both estimates
## numerically equal even under a two-phase DGP.

test_that("sequential_init uses stage-specific g, giving different EIF from pooled g", {
  set.seed(101L)
  n  <- 400L
  W  <- cbind(1, rnorm(n))

  ## Two-phase mechanism: different confounding strength in each half.
  g_true <- c(plogis(0.2 * W[seq_len(200L),    2L]),
              plogis(1.5 * W[seq_len(200L) + 200L, 2L]))
  A <- rbinom(n, 1L, g_true)
  Y <- 1.0 * A + 0.8 * W[, 2L] + rnorm(n, sd = 0.5)

  lib <- make_lib_rls()

  set.seed(101L)
  tmle_pool <- online_tmle_fit(W, A, Y,
                               Q_library      = lib,
                               g_library      = lib,
                               sequential_init = FALSE)

  set.seed(101L)
  tmle_seq  <- online_tmle_fit(W, A, Y,
                               Q_library      = lib,
                               g_library      = lib,
                               sequential_init = TRUE)

  ## Stage-specific H_i values differ from pooled H_i under a two-phase DGP,
  ## so the EIF means — and therefore psi — must differ.
  expect_false(isTRUE(all.equal(coef(tmle_pool), coef(tmle_seq), tolerance = 1e-4)))
})


## ---- 9. eif_sq_sum = sum(EIF_i^2) batch oracle (Item 3) --------------------
## Verify that the running sum of squared TMLE EIF contributions stored in
## eif_sq_sum equals sum(eif_i^2) recomputed from the stored nuisance models.
## This is the sample estimator of V_t in Chambaz, Zheng & van der Laan (2017):
##   V_t = (1/t) sum_{i=1}^{t} [D*(P_hat_i)(O_i)]^2
##
## For the batch phase: eif_i = H_i*(Y_i - Q*a_i) + Q1*_i - Q0*_i (un-centered)
## is computed from the pooled nuisance models (sequential_init=FALSE here).
## The oracle recomputes the same vector using the batch link-function identity
##   Q*a = linkinv( linkfun(Q) + eps * H )
## and verifies sum(eif^2) matches eif_sq_sum to 1e-8.
##
## Would fail if: eif_sq_sum is initialised from eif_M2 (the Welford centered
## sum-of-squares) instead of sum(psi_i^2), making it smaller by n*psi^2.

test_that("eif_sq_sum equals sum of squared EIF contributions in batch to 1e-8", {
  set.seed(88L)
  n <- 300L
  W <- cbind(1, matrix(rnorm(n * 2L), n, 2L))
  A <- rbinom(n, 1L, plogis(0.4 * W[, 2L]))
  Y <- 1.0 * A + 0.5 * W[, 2L] + rnorm(n, sd = 0.5)

  lib  <- make_lib_rls()
  tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib)

  ## Re-derive EIF manually from stored nuisance (same oracle as Test 2).
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
  eif    <- H * (Y - Qa_str) + Q1_str - Q0_str

  expect_equal(tmle$eif_sq_sum, sum(eif^2), tolerance = 1e-8)
})


## ---- 10. Martingale SE uses uncentered 2nd moment; > i.i.d. SE for psi ≠ 0
## Formula: SE_martingale = sqrt(eif_sq_sum) / n_obs.
## Since eif_sq_sum = eif_M2 + n*psi^2 = (n-1)*eif_var + n*psi^2:
##   SE_martingale^2 = (eif_var*(n-1)/n + psi^2) / n
##   SE_iid^2        = eif_var / n
## For any non-zero psi (which is always true for ATE ≠ 0):
##   SE_martingale > SE_iid.
##
## References: Chambaz, Zheng & van der Laan (2017) V_t formula (no active test).
##
## Would fail if: variance_type="martingale" secretly uses eif_var (the i.i.d.
## formula), making the two SEs identical and the expect_gt assertion false.

test_that("SE_martingale = sqrt(eif_sq_sum)/n and exceeds SE_iid for non-zero ATE", {
  set.seed(55L)
  n <- 400L
  W <- cbind(1, matrix(rnorm(n * 2L), n, 2L))
  A <- rbinom(n, 1L, plogis(0.4 * W[, 2L]))
  Y <- 1.0 * A + 0.5 * W[, 2L] + rnorm(n, sd = 0.5)

  lib      <- make_lib_rls()
  tmle_iid <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib,
                              variance_type = "iid")
  tmle_mrt <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib,
                              variance_type = "martingale")

  ## Both objects have the same nuisance models and eif_sq_sum (same DGP + seed).
  se_mart_direct <- sqrt(tmle_mrt$eif_sq_sum) / tmle_mrt$n_obs
  se_iid_direct  <- sqrt(tmle_iid$eif_var / tmle_iid$n_obs)

  ## SE from confint() must match the direct formula.
  ci_mart <- confint(tmle_mrt)
  se_from_confint <- (ci_mart[1L, 2L] - ci_mart[1L, 1L]) / (2 * qnorm(0.975))
  expect_equal(se_from_confint, se_mart_direct, tolerance = 1e-10)

  ## Martingale SE > i.i.d. SE when psi ≠ 0 (ATE ≈ 1 in this DGP).
  expect_gt(se_mart_direct, se_iid_direct)
})


## ---- 11. eif_sq_sum is correctly incremented during streaming (Item 3) ------
## After one streaming update with new observation (w_new, a_new, y_new), the
## new eif_sq_sum must equal old eif_sq_sum + eif_stream^2, where eif_stream
## is computed from the PRE-update nuisance models (predict-then-update).
##
## Oracle: replicate the streaming EIF formula inline using the pre-update Q_sl,
## g_sl, and epsilon stored in the object before the update.
##
## References: Chambaz, Zheng & van der Laan (2017) — online accumulation of V_t.
##
## Would fail if: the line `object$eif_sq_sum <- object$eif_sq_sum + eif^2` is
## absent from update.online_tmle(), causing eif_sq_sum to stay constant while
## the Welford mean and variance continue to evolve.

test_that("eif_sq_sum is incremented by eif_stream^2 after one streaming update", {
  set.seed(200L)
  n <- 200L
  W <- cbind(1, rnorm(n))
  A <- rbinom(n, 1L, plogis(0.3 * W[, 2L]))
  Y <- 1.0 * A + 0.5 * W[, 2L] + rnorm(n, sd = 0.5)

  lib  <- make_lib_rls()
  tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib)

  ## Fixed new observation — values chosen to give a non-trivial EIF.
  w_new <- c(1, 0.7); a_new <- 1L; y_new <- 1.8

  ## Replicate the streaming EIF computation using pre-update nuisance.
  ## (Mirrors the logic in update.online_tmle; no internal-function dependency.)
  eps   <- tmle$epsilon
  fam   <- tmle$family
  mA    <- tmle$sum_A / tmle$n_obs       # mean_A before update

  Q1_s  <- predict(tmle$Q_sl, matrix(c(1, w_new),     nrow = 1L))
  Q0_s  <- predict(tmle$Q_sl, matrix(c(0, w_new),     nrow = 1L))
  Qa_s  <- predict(tmle$Q_sl, matrix(c(a_new, w_new), nrow = 1L))
  g_s   <- pmin(pmax(predict(tmle$g_sl, matrix(w_new, nrow = 1L)),
                     tmle$g_clamp), 1 - tmle$g_clamp)

  H_s   <- a_new / g_s - (1 - a_new) / (1 - g_s)   # ATE clever covariate
  H1_s  <-  1 / g_s
  H0_s  <- -1 / (1 - g_s)

  Qa_str <- fam$linkinv(fam$linkfun(Qa_s) + eps * H_s)
  Q1_str <- fam$linkinv(fam$linkfun(Q1_s) + eps * H1_s)
  Q0_str <- fam$linkinv(fam$linkfun(Q0_s) + eps * H0_s)
  eif_s  <- H_s * (y_new - Qa_str) + Q1_str - Q0_str

  sq_before <- tmle$eif_sq_sum
  tmle_upd  <- update(tmle, w = w_new, a = a_new, y = y_new)

  expect_equal(tmle_upd$eif_sq_sum, sq_before + eif_s^2, tolerance = 1e-10)
})
