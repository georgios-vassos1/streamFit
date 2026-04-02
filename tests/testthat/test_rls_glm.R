test_that("rls_glm_update: S shrinks after each observation (Fisher info accumulates)", {
  ## One RLS-GLM step must reduce tr(S): observing a data point gains information.
  p   <- 3
  S0  <- diag(p)
  out <- rls_glm_update(c(1, 0.5, -0.3), 1L, numeric(p), S0, binomial())
  expect_lt(sum(diag(out$S)), sum(diag(S0)))
})

test_that("rls_glm_fit (Gaussian) matches rls_fit numerically", {
  set.seed(1)
  n <- 500; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(0.5, -1, 0.8) + rnorm(n)
  expect_equal(rls_fit(X, y)$beta,
               rls_glm_fit(X, y, family = gaussian())$beta,
               tolerance = 1e-10)
})

test_that("rls_glm_fit (logistic) converges close to batch glm", {
  set.seed(42)
  n <- 2000; p <- 4
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  beta0 <- c(-0.5, 1, -1, 0.5)
  y <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta0)))
  fit_online <- rls_glm_fit(X, y, family = binomial())
  fit_batch  <- stats::glm(y ~ X[, -1], family = binomial())
  expect_equal(fit_online$beta, unname(coef(fit_batch)), tolerance = 0.05)
})

test_that("rls_glm_fit (Poisson) converges with score clipping", {
  set.seed(7)
  n <- 2000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rpois(n, exp(X %*% c(0.5, 0.8, -0.6)))
  fit <- rls_glm_fit(X, y, family = poisson(),
                     S0_scale   = 0.1,
                     beta_init  = c(log(mean(y)), rep(0, p - 1)),
                     score_clip = 5 * sqrt(mean(y)))
  fit_batch <- stats::glm(y ~ X[, -1], family = poisson())
  expect_equal(fit$beta, unname(coef(fit_batch)), tolerance = 0.05)
})

test_that("rls_glm_update input validation: bad lambda", {
  p <- 2
  expect_error(
    rls_glm_update(c(1, 0), 1, numeric(p), diag(p), binomial(), lambda = 0),
    "'lambda' must be a scalar in"
  )
  expect_error(
    rls_glm_update(c(1, 0), 1, numeric(p), diag(p), binomial(), lambda = 2),
    "'lambda' must be a scalar in"
  )
})

test_that("rls_glm_update input validation: mismatched dims", {
  expect_error(
    rls_glm_update(c(1, 0, 0), 1, numeric(2), diag(2), binomial()),
    "length of 'x' must equal length of 'beta'"
  )
  expect_error(
    rls_glm_update(c(1, 0), 1, numeric(2), diag(3), binomial()),
    "'S' must be a square matrix"
  )
})

test_that("rls_glm_fit input validation", {
  X <- cbind(1, rnorm(10)); y <- rbinom(10, 1, 0.5)
  expect_error(rls_glm_fit(X, y[1:8], family = binomial()),
               "nrow\\(X\\) must equal length\\(y\\)")
  expect_error(rls_glm_fit(X, y, family = binomial(), lambda = 1.5),
               "'lambda' must be a scalar in")
  expect_error(rls_glm_fit(X, y, family = binomial(), S0_scale = -1),
               "'S0_scale' must be a positive scalar")
  expect_error(rls_glm_fit(X, y, family = binomial(), eta_clip = 0),
               "'eta_clip' must be a positive scalar")
  expect_error(rls_glm_fit(X, y, family = binomial(), score_clip = -2),
               "'score_clip' must be a positive scalar")
})

test_that("update.stream_fit: streaming matches rls_glm_fit on full data (logistic)", {
  ## Core mathematical property: fit on 60 obs then stream 20 more must give
  ## the same beta as fitting directly on all 80 observations.
  set.seed(5)
  n <- 80; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rbinom(n, 1, 0.4)
  fit <- rls_glm_fit(X[1:60, ], y[1:60], family = binomial())
  for (i in 61:n) fit <- update(fit, x = X[i, ], y = y[i])
  fit_full <- rls_glm_fit(X, y, family = binomial())
  expect_equal(fit$beta, fit_full$beta, tolerance = 1e-10)
})

test_that("predict: link and response predictions related by logistic for binomial", {
  ## Verifies that predict correctly applies the inverse link function:
  ## response = 1 / (1 + exp(-link)) for the logit link.
  set.seed(6)
  n <- 100; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rbinom(n, 1, 0.5)
  fit  <- rls_glm_fit(X, y, family = binomial())
  eta  <- predict(fit, X[1:5, ], type = "link")
  prob <- predict(fit, X[1:5, ], type = "response")
  expect_equal(prob, 1 / (1 + exp(-eta)), tolerance = 1e-10)
})

test_that("print.stream_fit handles NULL score_clip without error (regression)", {
  ## Regression test: NULL score_clip previously caused unlist() to silently
  ## drop the element and then round() crashed on a 0-length vector.
  set.seed(7)
  n <- 30; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rbinom(n, 1, 0.5)
  fit <- rls_glm_fit(X, y, family = binomial(), score_clip = NULL)
  expect_output(print(fit), "Stream fit")
})


# -------------------------------------------------------------------
# S-diagonal clamping (S_max)
# -------------------------------------------------------------------

test_that("rls_glm_update: S_max clamps inflated diagonals on sparse x", {
  ## A single sparse update with lambda < 1 inflates unobserved dimensions.
  ## Hand-computable: S0 = 5*I, x = (1,0,0,0), lambda = 0.95.
  ## For dims 2-4 (x_j = 0): S_new[j,j] = S0[j,j] / lambda = 5/0.95 ≈ 5.263.
  ## With S_max = 3.0, these must be rescaled to exactly 3.0.
  ## For dim 1 (x_j = 1): Sherman-Morrison shrinks it below 5, then /lambda,
  ## but it should still end up < S_max because it received information.
  p <- 4
  S0 <- 5 * diag(p)
  fam <- poisson()
  out_clamped   <- rls_glm_update(c(1,0,0,0), 1L, numeric(p), S0, fam,
                                   lambda = 0.95, S_max = 3.0)
  out_unclamped <- rls_glm_update(c(1,0,0,0), 1L, numeric(p), S0, fam,
                                   lambda = 0.95)
  ## Unclamped: dims 2-4 should be 5/0.95 ≈ 5.263
  expect_equal(diag(out_unclamped$S)[2:4], rep(5 / 0.95, 3), tolerance = 1e-10)
  ## Clamped: all diagonals bounded
  expect_true(all(diag(out_clamped$S) <= 3.0 + 1e-10))
  ## Clamped: dims 2-4 should be exactly S_max (they were over)
  expect_equal(diag(out_clamped$S)[2:4], rep(3.0, 3), tolerance = 1e-10)
  ## Dim 1 got information, so it should be smaller than unclamped dim 1
  ## (and also smaller than S_max since it was shrunk by the update)
  expect_lt(diag(out_clamped$S)[1], 3.0)
})

test_that("rls_glm_fit: S_max prevents covariance windup on PE-like sparse design", {
  ## Realistic PE scenario: 10 interval dummies + 1 covariate.
  ## Each observation activates exactly one interval dummy.
  ## With lambda < 1, the 9 inactive dummies per step get inflated by 1/lambda.
  ## With S_max, they stay bounded. We verify: (a) clamped S is bounded,
  ## (b) unclamped S has at least one large diagonal, (c) both fits produce
  ## finite coefficients (the clamped one doesn't diverge).
  set.seed(88)
  K <- 10; q <- 1; p <- K + q
  n <- 500
  X <- matrix(0, n, p)
  y <- integer(n)
  log_e <- numeric(n)
  for (i in seq_len(n)) {
    k <- ((i - 1L) %% K) + 1L
    X[i, k] <- 1
    X[i, K + 1] <- rnorm(1)
    log_e[i] <- log(runif(1, 0.5, 1.5))
    rate <- exp(-1 + 0.3 * X[i, K + 1] + log_e[i])
    y[i] <- rpois(1, rate)
  }
  fit_no_clamp <- rls_glm_fit(X, y, family = poisson(), lambda = 0.98,
                               S0_scale = 1, offset = log_e)
  fit_clamped <- rls_glm_fit(X, y, family = poisson(), lambda = 0.98,
                              S0_scale = 1, offset = log_e, S_max = 1.0)
  ## Clamped: all diagonals bounded
  expect_true(all(diag(fit_clamped$S) <= 1.0 + 1e-10))
  ## Unclamped: at least one diagonal should have blown up
  expect_true(max(diag(fit_no_clamp$S)) > 1.0)
  ## Both fits should produce finite coefficients
  expect_true(all(is.finite(fit_clamped$beta)))
  expect_true(all(is.finite(fit_no_clamp$beta)))
})

test_that("rls_glm_fit: S_max does not degrade accuracy on dense designs", {
  ## When the design is dense (no sparse dummies), S_max should never fire
  ## and the fit should be numerically identical to the unclamped fit.
  set.seed(90)
  n <- 500; p <- 3
  X <- cbind(1, matrix(rnorm(n * 2), n, 2))
  beta_true <- c(0.5, 0.8, -0.6)
  y <- rpois(n, exp(X %*% beta_true))
  fit_plain <- rls_glm_fit(X, y, family = poisson(), lambda = 0.99,
                            S0_scale = 1, beta_init = c(log(mean(y)), 0, 0),
                            score_clip = 5)
  fit_smax  <- rls_glm_fit(X, y, family = poisson(), lambda = 0.99,
                            S0_scale = 1, beta_init = c(log(mean(y)), 0, 0),
                            score_clip = 5, S_max = 10.0)
  ## With S_max = 10 on a dense design, S diags never reach 10,
  ## so the results should be bitwise identical
  expect_equal(fit_smax$beta, fit_plain$beta, tolerance = 1e-12)
  expect_equal(fit_smax$S, fit_plain$S, tolerance = 1e-12)
})

test_that("streaming with S_max on sparse design: batch == fit + update", {
  ## The mathematical equivalence property: fitting on all n observations
  ## at once must give the same beta as fitting on the first n1 then
  ## streaming the rest — even when S_max is active.
  set.seed(91)
  K <- 5; q <- 1; p <- K + q
  n <- 100
  X <- matrix(0, n, p)
  y <- integer(n)
  log_e <- numeric(n)
  for (i in seq_len(n)) {
    k <- ((i - 1L) %% K) + 1L
    X[i, k] <- 1
    X[i, K + 1] <- rnorm(1)
    log_e[i] <- log(runif(1, 0.5, 1.5))
    y[i] <- rpois(1, exp(-1 + 0.3 * X[i, K + 1] + log_e[i]))
  }
  ## Batch: fit all at once
  fit_full <- rls_glm_fit(X, y, family = poisson(), lambda = 0.98,
                           S0_scale = 1, offset = log_e, S_max = 1.0)
  ## Streaming: fit first 60, stream remaining 40
  fit_stream <- rls_glm_fit(X[1:60, ], y[1:60], family = poisson(),
                             lambda = 0.98, S0_scale = 1,
                             offset = log_e[1:60], S_max = 1.0)
  for (i in 61:n) {
    fit_stream <- update(fit_stream, x = X[i, ], y = y[i],
                         offset = log_e[i])
  }
  expect_equal(fit_stream$beta, fit_full$beta, tolerance = 1e-10)
  expect_equal(fit_stream$S, fit_full$S, tolerance = 1e-10)
})



# -------------------------------------------------------------------
# Gamma: streaming pearson_ss accumulation
# -------------------------------------------------------------------

test_that("update.stream_fit: Gamma streaming matches full batch", {
  set.seed(54)
  n <- 500; p <- 2
  X <- cbind(1, rnorm(n, sd = 0.3))
  y <- rgamma(n, shape = 5, rate = 5 / exp(X %*% c(1, 0.2)))
  fam <- Gamma(link = "log")
  b0 <- c(log(mean(y)), 0)
  fit_full <- rls_glm_fit(X, y, family = fam,
                          beta_init = b0,
                          score_clip = 5, S0_scale = 1)
  n_init <- 300
  fit <- rls_glm_fit(X[1:n_init, ], y[1:n_init], family = fam,
                     beta_init = b0,
                     score_clip = 5, S0_scale = 1)
  for (i in (n_init + 1):n)
    fit <- update(fit, x = X[i, ], y = y[i])
  expect_equal(fit$beta, fit_full$beta, tolerance = 1e-10)
  expect_equal(fit$pearson_ss, fit_full$pearson_ss, tolerance = 1e-8)
})
