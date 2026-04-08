## Tests for super_learner_fit()
##
## Each test targets a specific nontrivial property:
##   - Oracle inequality [van der Laan 2007]: sl_cv_risk <= min(cv_risk).
##     Holds by construction (individual learners are feasible simplex
##     vertices), but tests that (a) the optimizer converges and (b) the
##     clamping is applied consistently across families.
##   - Single-learner degenerate case: weight = 1 and sl_cv_risk = cv_risk.
##   - Held-out generalisation: ensemble test deviance <= worst learner,
##     testing that CV-fitted weights transfer to fresh data.

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
