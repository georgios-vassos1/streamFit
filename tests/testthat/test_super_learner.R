## Tests for super_learner_fit()
##
## Reference implementations consulted:
##   - ecpolley/SuperLearner, tests/testthat/, https://github.com/ecpolley/SuperLearner
##   - benkeser/onlinesl, https://github.com/benkeser/onlinesl
##   - tlverse/origami (folds_rolling_origin), https://github.com/tlverse/origami
##   - Benkeser, Ju, Lendle & van der Laan (2018), Statistics in Medicine, PMC5671383
##   - Cerqueira, Torgo & Mozetič (2020), Machine Learning, arXiv:1905.11744
##
## Properties tested:
##   1. Oracle inequality: sl_cv_risk <= min(cv_risk) [binomial, gaussian, poisson]
##      [source: van der Laan, Polley & Hubbard (2007); clamping bug history]
##   2. Single-learner degenerate case: weight = 1 and sl_cv_risk = cv_risk
##      [source: ecpolley/SuperLearner L=1 branch]
##   3. Held-out generalisation (k-fold): ensemble log-loss <= worst learner
##      [source: Benkeser et al. (2018) simulation design]
##   4. Sequential CV oracle inequality (binomial): sl_cv_risk <= min(cv_risk)
##      [source: Benkeser et al. (2018), online oracle inequality]
##   5. Sequential CV oracle inequality (gaussian): catches family-clamping regression
##      [source: clamping bug history]
##   6. No data leakage: flipping y[t] does not change Z[t,], but does change Z[t+1,]
##      [source: Dawid (1984) prequential principle; origami temporal ordering contract]
##   7. Sequential vs k-fold disagreement: distinct Z matrices on trended data
##      [source: Cerqueira et al. (2020) prequential vs CV comparison]
##   8. Sequential CV held-out generalisation: predict() with prequential weights
##      beats intercept-only on fresh data
##      [source: Benkeser et al. (2018) holdout validation design]

## ---- shared fixtures -------------------------------------------------------

make_data <- function(n, p, family, seed = 1L) {
  set.seed(seed)
  X     <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
  beta0 <- c(-0.5, 1, -1, 0.5)[seq_len(p)]
  eta   <- as.vector(X %*% beta0)
  y <- switch(family$family,
    gaussian = eta + rnorm(n, sd = 0.5),
    binomial = rbinom(n, 1L, family$linkinv(eta)),
    poisson  = rpois(n,  family$linkinv(eta)),
    stop("unsupported family in fixture")
  )
  list(X = X, y = y)
}

lrn_rls <- make_learner(
  fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
  predict = function(model, newdata, ...) predict(model, newdata)
)

lrn_isgd <- make_learner(
  fit     = function(X, y, family, ...) isgd_fit(X, y, family = family),
  predict = function(model, newdata, ...) predict(model, newdata)
)

lrn_intercept <- make_learner(
  fit     = function(X, y, family, ...) rls_glm_fit(X[, 1L, drop = FALSE],
                                                     y, family = family),
  predict = function(model, newdata, ...) predict(model, newdata[, 1L, drop = FALSE])
)

## Online-capable RLS learner (used in sequential CV tests)
lrn_rls_online <- make_learner(
  fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
  predict = function(model, newdata, ...) predict(model, newdata),
  update  = function(model, x, y, family, ...) update(model, x = x, y = y)
)

## Online-capable intercept-only learner (used in sequential CV tests)
lrn_intercept_online <- make_learner(
  fit     = function(X, y, family, ...) rls_glm_fit(X[, 1L, drop = FALSE],
                                                     y, family = family),
  predict = function(model, newdata, ...) predict(model,
                                                   newdata[, 1L, drop = FALSE]),
  update  = function(model, x, y, family, ...) update(model, x = x[1L], y = y)
)

## Log-loss for held-out evaluation (binomial only)
logloss <- function(y, p) {
  p <- pmax(pmin(p, 1 - 1e-15), 1e-15)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

## ---- Oracle inequality -----------------------------------------------------
## sl_cv_risk <= min(cv_risk): the individual learner risks are feasible
## vertices of the simplex the meta-learner minimises over, so the optimised
## combination can only be at least as good.  Tests optimizer convergence and
## family-consistent clamping (a clamping bug caused this to fail for Gaussian).

test_that("oracle inequality: sl_cv_risk <= min(cv_risk) for binomial", {
  d  <- make_data(400L, 4L, binomial())
  sl <- super_learner_fit(d$X, d$y, family = binomial(),
                          library = list(rls = lrn_rls, isgd = lrn_isgd),
                          k = 5L)
  expect_lte(sl$sl_cv_risk, min(sl$cv_risk) + 1e-6)
})

test_that("oracle inequality: sl_cv_risk <= min(cv_risk) for gaussian", {
  d  <- make_data(400L, 4L, gaussian())
  sl <- super_learner_fit(d$X, d$y, family = gaussian(),
                          library = list(rls = lrn_rls, isgd = lrn_isgd),
                          k = 5L)
  expect_lte(sl$sl_cv_risk, min(sl$cv_risk) + 1e-6)
})

test_that("oracle inequality: sl_cv_risk <= min(cv_risk) for poisson", {
  d  <- make_data(400L, 4L, poisson())
  sl <- super_learner_fit(d$X, d$y, family = poisson(),
                          library = list(rls = lrn_rls, isgd = lrn_isgd),
                          k = 5L)
  expect_lte(sl$sl_cv_risk, min(sl$cv_risk) + 1e-6)
})

## ---- Single-learner degenerate case ----------------------------------------
## With one learner, weight must be 1 and the ensemble CV risk must equal that
## learner's CV risk.  Tests the L = 1 branch of .sl_meta_weights and that
## sl_cv_risk is computed from Z %*% weights, not from some other path.

test_that("single-learner library: weight = 1 and sl_cv_risk = cv_risk", {
  d  <- make_data(200L, 4L, binomial())
  sl <- super_learner_fit(d$X, d$y, family = binomial(),
                          library = list(rls = lrn_rls),
                          k = 3L)
  expect_equal(sl$weights[["rls"]], 1, tolerance = 1e-8)
  expect_equal(sl$sl_cv_risk, sl$cv_risk[["rls"]], tolerance = 1e-8)
})

## ---- Held-out generalisation -----------------------------------------------
## On a test set unseen during CV weight fitting, the ensemble should achieve
## lower log-loss than the worst individual learner in the library.  Tests that
## CV-optimised weights transfer to fresh data and that predict() applies them
## correctly.  Uses a library where one learner (GLM) is clearly better than
## another (intercept-only) to make the gap reliable in finite samples.

test_that("ensemble test log-loss <= worst individual learner on held-out data", {
  d    <- make_data(600L, 4L, binomial(), seed = 11L)
  n_tr <- 400L
  X_tr <- d$X[seq_len(n_tr), ];  y_tr <- d$y[seq_len(n_tr)]
  X_te <- d$X[-seq_len(n_tr), ]; y_te <- d$y[-seq_len(n_tr)]

  sl <- super_learner_fit(X_tr, y_tr, family = binomial(),
                          library = list(glm  = lrn_rls,
                                         null = lrn_intercept),
                          k = 5L)

  p_glm  <- lrn_rls$predict(sl$learners[["glm"]],  X_te)
  p_null <- lrn_intercept$predict(sl$learners[["null"]],
                                   X_te[, 1L, drop = FALSE])
  p_sl   <- predict(sl, X_te)

  expect_lte(logloss(y_te, p_sl),
             max(logloss(y_te, p_glm), logloss(y_te, p_null)))
})

## ---- Sequential CV: oracle inequality --------------------------------------
## The oracle inequality sl_cv_risk <= min(cv_risk) must hold under sequential
## CV just as under k-fold.  Here it verifies that (a) the meta-learner
## optimises correctly over prequential predictions and (b) .sl_clamp is
## applied consistently in both cv_risk and sl_cv_risk computations.

## Would fail if: .sl_meta_weights used a different clamping path than the
## cv_risk computation, or if the meta-learner objective diverged on the
## prequential Z matrix.
test_that("sequential CV: oracle inequality sl_cv_risk <= min(cv_risk)", {
  d  <- make_data(400L, 4L, binomial(), seed = 3L)
  sl <- super_learner_fit(d$X, d$y, family = binomial(),
                          library = list(rls = lrn_rls_online,
                                         null = lrn_intercept),
                          cv_type = "sequential", n_min = 30L)
  expect_lte(sl$sl_cv_risk, min(sl$cv_risk) + 1e-6)
})

## ---- Sequential CV: no data leakage ----------------------------------------
## Under prequential CV the prediction at time t must be formed using only
## observations 1..(t-1).  Test: flip y[t0] and refit.  Z[t0,] must be
## unchanged (the model at t0 has not seen y[t0]).  Z[t0+1,] must change
## (the model for t0+1 was updated with y[t0]).
## This directly tests the test-before-update contract of .sl_seq_predictions.

## Would fail if: the loop updated the model with y[t] before recording Z[t,],
## i.e. predict came after update rather than before.
test_that("sequential CV: Z[t0,] unchanged when y[t0] is flipped, Z[t0+1,] changes", {
  n_min <- 20L; t0 <- 50L
  d <- make_data(120L, 4L, binomial(), seed = 21L)

  sl1 <- super_learner_fit(d$X, d$y, family = binomial(),
                            library = list(rls = lrn_rls_online),
                            cv_type = "sequential", n_min = n_min)

  d2       <- d
  d2$y[t0] <- 1L - d$y[t0]   # flip the binary outcome at t0
  sl2 <- super_learner_fit(d2$X, d2$y, family = binomial(),
                            library = list(rls = lrn_rls_online),
                            cv_type = "sequential", n_min = n_min)

  ## Prediction at t0: model has not seen y[t0] — must be identical
  expect_equal(sl1$Z[t0, ], sl2$Z[t0, ], tolerance = 1e-12)

  ## Prediction at t0+1: model was updated with y[t0] — must differ
  expect_false(isTRUE(all.equal(sl1$Z[t0 + 1L, ], sl2$Z[t0 + 1L, ])))
})

## ---- Sequential CV: oracle inequality for gaussian -------------------------
## Catches a regression to the family-clamping bug: if Gaussian predictions
## were clamped to (eps, 1-eps) in the prequential loop, the deviance is
## distorted and sl_cv_risk can exceed min(cv_risk).

## Would fail if: .sl_clamp applied binomial-style clamping to Gaussian
## predictions inside .sl_seq_predictions or in the meta-weight objective.
test_that("sequential CV: oracle inequality holds for gaussian", {
  d  <- make_data(400L, 4L, gaussian(), seed = 23L)
  sl <- super_learner_fit(d$X, d$y, family = gaussian(),
                          library = list(rls  = lrn_rls_online,
                                         null = lrn_intercept_online),
                          cv_type = "sequential", n_min = 30L)
  expect_lte(sl$sl_cv_risk, min(sl$cv_risk) + 1e-6)
})

## ---- Sequential CV: held-out generalisation --------------------------------
## After fitting with prequential CV, predict() with the sequential weights
## should achieve lower log-loss than the intercept-only learner on fresh data.
## Tests that (a) the sequential meta-weights are optimised correctly and
## (b) predict.super_learner_fit applies them to new observations.

## Would fail if: predict() ignored sl$weights and used equal weights, or if
## the sequential meta-weights were driven to the intercept-only vertex.
test_that("sequential CV: ensemble held-out log-loss < intercept-only", {
  d    <- make_data(700L, 4L, binomial(), seed = 29L)
  n_tr <- 400L
  X_tr <- d$X[seq_len(n_tr), ]; y_tr <- d$y[seq_len(n_tr)]
  X_te <- d$X[-seq_len(n_tr), ]; y_te <- d$y[-seq_len(n_tr)]

  sl <- super_learner_fit(X_tr, y_tr, family = binomial(),
                          library = list(glm  = lrn_rls_online,
                                         null = lrn_intercept_online),
                          cv_type = "sequential", n_min = 30L)

  p_sl   <- predict(sl, X_te)
  p_null <- lrn_intercept_online$predict(sl$learners[["null"]],
                                          X_te[, 1L, drop = FALSE])

  expect_lt(logloss(y_te, p_sl), logloss(y_te, p_null))
})

## ---- Sequential vs k-fold: different predictions ---------------------------
## On data with a strong temporal trend, prequential and k-fold CV produce
## different out-of-sample predictions.  If they were identical, one of the
## two code paths would be dead.

## Would fail if: the sequential path silently fell back to k-fold (e.g.
## by ignoring cv_type), or if .sl_seq_predictions initialised models on
## the full data instead of the burn-in prefix.
test_that("sequential and k-fold CV produce different Z matrices", {
  ## Generate data with a structural time trend so the ordering matters
  set.seed(7L)
  n <- 300L; p <- 4L
  t_idx <- seq_len(n)
  X     <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
  eta   <- as.vector(X %*% c(-0.5, 1, -1, 0.5)) + 0.01 * t_idx
  y     <- rbinom(n, 1L, 1 / (1 + exp(-eta)))

  sl_kf <- super_learner_fit(X, y, family = binomial(),
                             library = list(rls = lrn_rls_online),
                             cv_type = "kfold", k = 5L)
  sl_sq <- super_learner_fit(X, y, family = binomial(),
                             library = list(rls = lrn_rls_online),
                             cv_type = "sequential", n_min = 30L)

  ok <- complete.cases(sl_sq$Z)
  expect_false(isTRUE(all.equal(sl_kf$Z[ok, ], sl_sq$Z[ok, ])))
})
