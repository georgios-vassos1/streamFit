test_that("isgd_update: implicit step self-limits for Poisson extreme count", {
  ## Key iSGD property: even when y >> mu, the implicit equation bounds |xi|,
  ## so the update is far smaller than an explicit gradient step would be.
  ## This is the principal advantage of iSGD over explicit SGD for Poisson.
  set.seed(11)
  p     <- 3
  beta  <- c(0.5, 0, 0)     # mu = exp(0.5) ≈ 1.65
  x     <- c(1, 0.1, -0.1)
  y     <- 200L              # extreme count — explicit SGD would diverge
  gamma <- 0.5
  beta_new <- isgd_update(x, y, beta, gamma, poisson())
  expect_true(all(is.finite(beta_new)))
  ## Explicit SGD step on the intercept would be: gamma * (y - mu) * x[1] ≈ 99
  explicit_step <- gamma * (y - exp(sum(x * beta))) * x[1]
  implicit_step <- beta_new[1] - beta[1]
  expect_lt(abs(implicit_step), abs(explicit_step))
})

test_that("isgd_fit (logistic) converges close to batch glm", {
  set.seed(42)
  n <- 3000; p <- 4
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  beta0 <- c(-0.5, 1, -1, 0.5)
  y <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta0)))
  fit_online <- isgd_fit(X, y, family = binomial(), gamma1 = 1, alpha = 0.7)
  fit_batch  <- stats::glm(y ~ X[, -1], family = binomial())
  expect_equal(fit_online$beta, unname(coef(fit_batch)), tolerance = 0.1)
})

test_that("isgd_fit (Gaussian) recovers true coefficients", {
  set.seed(1)
  n <- 2000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% c(0.5, -1, 0.8) + rnorm(n)
  fit <- isgd_fit(X, y, family = gaussian(), gamma1 = 0.5, alpha = 0.7)
  expect_equal(fit$beta, c(0.5, -1, 0.8), tolerance = 0.1)
})

test_that("conv_path: distance to truth decreases substantially for well-specified model", {
  ## For a well-specified model with enough data, the final distance to the
  ## true parameter must be far smaller than the initial distance.
  ## This tests that conv_path measures actual convergence, not just a norm.
  set.seed(9)
  n <- 500; p <- 3
  beta_true <- c(1, -1, 0.5)
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- X %*% beta_true + rnorm(n, sd = 0.5)
  fit <- rls_fit(X, y)
  d <- conv_path(fit$beta_path, beta_true)
  expect_length(d, n)
  ## Final distance must be at least 90 % smaller than the initial distance
  expect_lt(d[n], 0.1 * d[1])
})

test_that("isgd_update input validation: bad gamma", {
  expect_error(
    isgd_update(c(1, 0), 1, numeric(2), gamma = 0, binomial()),
    "'gamma' must be a positive scalar"
  )
  expect_error(
    isgd_update(c(1, 0), 1, numeric(2), gamma = -1, binomial()),
    "'gamma' must be a positive scalar"
  )
})

test_that("isgd_update input validation: mismatched x and beta", {
  expect_error(
    isgd_update(c(1, 0, 0), 1, numeric(2), gamma = 0.5, binomial()),
    "length of 'x' must equal length of 'beta'"
  )
})

test_that("isgd_fit input validation: bad alpha", {
  X <- cbind(1, rnorm(20)); y <- rbinom(20, 1, 0.5)
  expect_error(isgd_fit(X, y, family = binomial(), alpha = 0.5),
               "'alpha' must be in")
  expect_error(isgd_fit(X, y, family = binomial(), alpha = 1.1),
               "'alpha' must be in")
  expect_error(isgd_fit(X, y, family = binomial(), alpha = 0.3),
               "'alpha' must be in")
})

test_that("isgd_fit input validation: bad gamma1", {
  X <- cbind(1, rnorm(20)); y <- rbinom(20, 1, 0.5)
  expect_error(isgd_fit(X, y, family = binomial(), gamma1 = 0),
               "'gamma1' must be a positive scalar")
  expect_error(isgd_fit(X, y, family = binomial(), gamma1 = -1),
               "'gamma1' must be a positive scalar")
})

test_that("update.stream_fit: streaming matches isgd_fit on full data", {
  ## Core mathematical property: fit on 50 obs then stream 10 more one-at-a-time
  ## must give exactly the same beta as fitting directly on all 60 observations.
  set.seed(8)
  n <- 60; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rbinom(n, 1, 0.4)
  fit <- isgd_fit(X[1:50, ], y[1:50], family = binomial())
  for (i in 51:n) fit <- update(fit, x = X[i, ], y = y[i])
  fit_full <- isgd_fit(X, y, family = binomial())
  expect_equal(fit$beta, fit_full$beta, tolerance = 1e-10)
})
