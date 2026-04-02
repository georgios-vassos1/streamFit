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
