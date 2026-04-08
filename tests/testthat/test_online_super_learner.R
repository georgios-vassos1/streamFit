## Tests for online_sl_fit() and update.online_sl()
##
## Each test targets a specific nontrivial property:
##   - Base-learner dispatch: update function called when provided, skipped
##     when NULL.
##   - SGD directional convergence: when a small-n batch gives non-saturated
##     weights, streaming from the GLM DGP should increase the GLM weight.
##     Uses a known seed where the batch SL assigns modest weight to the GLM
##     (0.18), confirming the gradient pushes in the correct direction.
##   - Single-learner equivalence: with one base learner the ensemble
##     prediction must equal that learner's prediction exactly.
##   - Held-out generalisation: after online updates the ensemble achieves
##     lower log-loss than the intercept-only learner on fresh data.

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

lrn_rls_online <- make_learner(
  fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
  predict = function(model, newdata, ...) predict(model, newdata),
  update  = function(model, x, y, family, ...) update(model, x = x, y = y)
)

lrn_rls_static <- make_learner(
  fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
  predict = function(model, newdata, ...) predict(model, newdata)
)

lrn_intercept <- make_learner(
  fit     = function(X, y, family, ...) rls_glm_fit(X[, 1L, drop = FALSE],
                                                     y, family = family),
  predict = function(model, newdata, ...) predict(model,
                                                   newdata[, 1L, drop = FALSE]),
  update  = function(model, x, y, family, ...) update(model, x = x[1L], y = y)
)

logloss <- function(y, p) {
  p <- pmax(pmin(p, 1 - 1e-15), 1e-15)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

## ---- Base-learner dispatch -------------------------------------------------
## Tests that the update slot is called when provided and skipped when NULL.
## Uses nobs() as a proxy: an updated stream_fit increments its counter,
## a static one does not.

test_that("base learner n_obs increments when update function is provided", {
  d   <- make_data(200L, 4L, binomial())
  osl <- online_sl_fit(d$X[1:100, ], d$y[1:100], family = binomial(),
                       library = list(rls = lrn_rls_online), k = 3L)

  n_before <- nobs(osl$learners[["rls"]])
  for (i in 101:120)
    osl <- update(osl, x = d$X[i, ], y = d$y[i])

  expect_equal(nobs(osl$learners[["rls"]]), n_before + 20L)
})

test_that("base learner is held fixed when update function is NULL", {
  d   <- make_data(200L, 4L, binomial())
  osl <- online_sl_fit(d$X[1:100, ], d$y[1:100], family = binomial(),
                       library = list(rls = lrn_rls_static), k = 3L)

  n_before <- nobs(osl$learners[["rls"]])
  for (i in 101:120)
    osl <- update(osl, x = d$X[i, ], y = d$y[i])

  expect_equal(nobs(osl$learners[["rls"]]), n_before)
})

## ---- SGD directional convergence -------------------------------------------
## With a small initial batch (n = 60, seed = 17) the batch SL assigns a
## modest weight to the GLM (0.18) because the GLM can overfit small folds.
## After streaming 600 observations from the same GLM DGP, the GLM consistently
## makes better predictions than the intercept-only, so the SGD gradient
## should drive its weight up.  Tests both the direction (weight increases from
## batch init) and final dominance.

test_that("streaming from GLM DGP increases GLM weight above batch init", {
  p     <- 4L
  beta0 <- c(-0.5, 1, -1, 0.5)

  set.seed(17L)
  n_batch <- 60L
  X_batch <- cbind(1, matrix(rnorm(n_batch * (p - 1L)), n_batch, p - 1L))
  y_batch <- rbinom(n_batch, 1L, 1 / (1 + exp(-X_batch %*% beta0)))

  osl        <- online_sl_fit(X_batch, y_batch, family = binomial(),
                              library = list(glm  = lrn_rls_online,
                                             null = lrn_intercept),
                              k = 5L)
  w_glm_init <- osl$weights[["glm"]]   # 0.18 with this seed

  set.seed(99L)
  n_stream <- 600L
  X_str <- cbind(1, matrix(rnorm(n_stream * (p - 1L)), n_stream, p - 1L))
  y_str <- rbinom(n_stream, 1L, 1 / (1 + exp(-X_str %*% beta0)))

  for (i in seq_len(n_stream))
    osl <- update(osl, x = X_str[i, ], y = y_str[i])

  expect_gt(osl$weights[["glm"]], w_glm_init)            # directional increase
  expect_gt(osl$weights[["glm"]], osl$weights[["null"]]) # final dominance
})

## ---- Single-learner equivalence --------------------------------------------
## With exactly one base learner the weight is always 1, so the ensemble
## prediction must equal the base-learner prediction exactly after any number
## of online updates.  Would fail if the weighting arithmetic in
## predict.online_sl or the theta update corrupts the single-learner case.

test_that("single-learner online SL predictions equal the base learner", {
  d   <- make_data(200L, 4L, binomial(), seed = 9L)
  osl <- online_sl_fit(d$X[1:100, ], d$y[1:100], family = binomial(),
                       library = list(rls = lrn_rls_online), k = 3L)

  for (i in 101:200)
    osl <- update(osl, x = d$X[i, ], y = d$y[i])

  expect_equal(predict(osl, d$X), predict(osl$learners[["rls"]], d$X),
               tolerance = 1e-8)
})

## ---- Held-out generalisation after streaming -------------------------------
## After online updates the ensemble should achieve lower log-loss than the
## intercept-only learner on a held-out test set.  Tests that the full
## pipeline (batch init + SGD weight updates + base-learner updates) produces
## predictions that are genuinely useful on unseen data.

test_that("online ensemble test log-loss < intercept-only after streaming", {
  d    <- make_data(800L, 4L, binomial(), seed = 13L)
  n_tr <- 400L; n_st <- 200L
  X_tr <- d$X[seq_len(n_tr), ]
  y_tr <- d$y[seq_len(n_tr)]
  X_st <- d$X[n_tr + seq_len(n_st), ]
  y_st <- d$y[n_tr + seq_len(n_st)]
  X_te <- d$X[n_tr + n_st + seq_len(200L), ]
  y_te <- d$y[n_tr + n_st + seq_len(200L)]

  osl <- online_sl_fit(X_tr, y_tr, family = binomial(),
                       library = list(glm  = lrn_rls_online,
                                      null = lrn_intercept),
                       k = 3L)

  for (i in seq_len(n_st))
    osl <- update(osl, x = X_st[i, ], y = y_st[i])

  p_sl   <- predict(osl, X_te)
  p_null <- lrn_intercept$predict(osl$learners[["null"]],
                                   X_te[, 1L, drop = FALSE])

  expect_lt(logloss(y_te, p_sl), logloss(y_te, p_null))
})
