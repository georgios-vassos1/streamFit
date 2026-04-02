test_that("rls_update: sequential steps recover exact OLS solution", {
  ## After processing n observations from a flat prior, beta must equal
  ## (X'X)^{-1} X'y exactly (up to floating point).  This validates the
  ## Sherman-Morrison recursion.  With n >> p the (1/S0) regularisation
  ## term is negligible and the identity holds to machine precision.
  set.seed(10)
  n <- 20; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  beta_ols <- as.vector(solve(t(X) %*% X) %*% t(X) %*% y)
  S0  <- 1e6 * diag(p)
  beta_t <- numeric(p)
  S      <- S0
  for (i in seq_len(n)) {
    out    <- rls_update(X[i, ], y[i], beta_t, S)
    beta_t <- out$beta
    S      <- out$S
  }
  expect_equal(beta_t, beta_ols, tolerance = 1e-4)
})

test_that("rls_update: S shrinks after each observation (Fisher info accumulates)", {
  p   <- 3
  S0  <- diag(p)
  out <- rls_update(c(1, 0.5, -0.3), 2.1, numeric(p), S0)
  expect_lt(sum(diag(out$S)), sum(diag(S0)))
})

test_that("rls_fit recovers true coefficients (OLS, lambda=1)", {
  set.seed(42)
  n <- 1000; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  beta_true <- c(1.0, -0.5, 0.8)
  y   <- X %*% beta_true + rnorm(n, sd = 0.5)
  fit <- rls_fit(X, y)
  expect_s3_class(fit, "stream_fit")
  expect_equal(fit$beta, beta_true, tolerance = 0.05)
})

test_that("rls_fit with lambda < 1 tracks a regime shift better than lambda = 1", {
  set.seed(99)
  n <- 1000; t_break <- 500
  X <- matrix(1, nrow = n, ncol = 1)
  y <- c(rnorm(t_break, mean = 0), rnorm(n - t_break, mean = 5))
  fit_forget <- rls_fit(X, y, lambda = 0.95)
  fit_fixed  <- rls_fit(X, y, lambda = 1.0)
  expect_lt(abs(fit_forget$beta - 5), abs(fit_fixed$beta - 5))
})

test_that("rls_fit returns stream_fit with correct structure", {
  fit <- rls_fit(cbind(1, rnorm(50)), rnorm(50))
  expect_named(fit, c("beta", "beta_path", "S", "family", "method",
                       "call", "hyperparams", "A_hat", "B_hat",
                       "rss", "n_obs", "pearson_ss"))
  expect_equal(nrow(fit$beta_path), 50)
  expect_equal(ncol(fit$beta_path), 2)
})

test_that("rls_update input validation: bad lambda", {
  p <- 2
  expect_error(rls_update(c(1, 0), 1, numeric(p), diag(p), lambda = 0),
               "'lambda' must be a scalar in")
  expect_error(rls_update(c(1, 0), 1, numeric(p), diag(p), lambda = 1.5),
               "'lambda' must be a scalar in")
  expect_error(rls_update(c(1, 0), 1, numeric(p), diag(p), lambda = -0.1),
               "'lambda' must be a scalar in")
})

test_that("rls_update input validation: mismatched dims", {
  expect_error(rls_update(c(1, 0, 0), 1, numeric(2), diag(2)),
               "length of 'x' must equal length of 'beta'")
  expect_error(rls_update(c(1, 0), 1, numeric(2), diag(3)),
               "'S' must be a square matrix")
})

test_that("coef() auto-generates names; print() shows method and observation count", {
  ## coef.stream_fit adds names (beta[1], beta[2], ...) when X has no column
  ## names.  That is the non-trivial behaviour of the S3 method.
  set.seed(1)
  fit <- rls_fit(cbind(1, rnorm(30)), rnorm(30))
  cf  <- coef(fit)
  expect_equal(names(cf), paste0("beta[", 1:2, "]"))
  ## print() must identify the method and observation count
  expect_output(print(fit), "Stream fit")
  expect_output(print(fit), "RLS")
  expect_output(print(fit), "30")
})

test_that("predict: fitted values numerically close to true conditional mean", {
  set.seed(2)
  n <- 500; p <- 2
  beta_true <- c(1.5, -0.8)
  X <- cbind(1, rnorm(n))
  y <- X %*% beta_true + rnorm(n, sd = 0.3)
  fit <- rls_fit(X, y)
  Xnew <- cbind(1, c(-1, 0, 1))
  expected <- as.vector(Xnew %*% beta_true)
  preds <- predict(fit, Xnew, type = "response")
  expect_equal(preds, expected, tolerance = 0.1)
})

test_that("update.stream_fit: streaming matches rls_fit on full data exactly", {
  ## Core mathematical property: fit on first 50, stream 51-60 one-at-a-time,
  ## must give exactly the same beta as fitting directly on all 60.
  set.seed(3)
  n <- 60; p <- 2
  X <- cbind(1, rnorm(n))
  y <- rnorm(n)
  fit <- rls_fit(X[1:50, ], y[1:50])
  for (i in 51:n) fit <- update(fit, x = X[i, ], y = y[i])
  fit_full <- rls_fit(X, y)
  expect_equal(fit$beta, fit_full$beta, tolerance = 1e-10)
})
