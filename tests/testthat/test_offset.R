## tests/testthat/test_offset.R
## Tests for offset support across all fitting functions

## -----------------------------------------------------------------------
## 1. Analytic single-step ground truth
##    Values computed by hand; independent of any R solver.
## -----------------------------------------------------------------------

test_that("rls_glm_update: single Poisson step with offset matches hand calculation", {
  ## Setup: x=1, y=3, beta=0, S=100, family=poisson(), offset=log(2), lambda=1
  ##
  ## eta   = 0 + log(2)          => mu  = 2,  dmu = 2,  vmu = 2,  w = 2
  ## Sx    = 100
  ## denom = 1 + w * x'Sx = 201  => gain = 100/201
  ## score = (y - mu)*dmu/vmu  = (3-2)*2/2 = 1
  ## beta_new = 0 + (100/201)*1 = 100/201

  out <- rls_glm_update(x = 1, y = 3, beta = 0, S = matrix(100),
                         family = poisson(), offset = log(2))
  expect_equal(out$beta, 100 / 201, tolerance = 1e-12)

  ## Without offset: eta=0, mu=1, w=1, denom=101, score=2  => beta_new = 200/101
  ## Must differ from the offset case.
  out0 <- rls_glm_update(x = 1, y = 3, beta = 0, S = matrix(100),
                          family = poisson(), offset = 0)
  expect_equal(out0$beta, 200 / 101, tolerance = 1e-12)

  expect_false(isTRUE(all.equal(out$beta, out0$beta)))
})

test_that("isgd_update: single Gaussian step with offset matches hand calculation", {
  ## Setup: x=1, y=3, beta=0, gamma=0.5, family=gaussian(), offset=1.5
  ##
  ## eta = 0 + 1.5 = 1.5  =>  mu = 1.5
  ## r   = 0.5 * (3 - 1.5) = 0.75
  ## Closed-form Gaussian: xi = r / (1 + gamma*norm_x2) = 0.75/1.5 = 0.5
  ## beta_new = 0 + 0.5

  beta_new <- isgd_update(x = 1, y = 3, beta = 0, gamma = 0.5,
                           family = gaussian(), offset = 1.5)
  expect_equal(beta_new, 0.5, tolerance = 1e-12)

  ## Without offset: eta=0, mu=0, r=1.5, xi=1.5/1.5=1 => beta_new=1
  beta_new0 <- isgd_update(x = 1, y = 3, beta = 0, gamma = 0.5,
                            family = gaussian(), offset = 0)
  expect_equal(beta_new0, 1.0, tolerance = 1e-12)

  expect_false(isTRUE(all.equal(beta_new, beta_new0)))
})

## -----------------------------------------------------------------------
## 2. Convergence: cold-start, intercept-only model, glm() as oracle
##    Intercept-only Poisson has a closed-form MLE: log(sum(y)/sum(exposure)).
##    Using glm() as the oracle (not beta_true) avoids confounding algorithm
##    bias with finite-sample noise.
## -----------------------------------------------------------------------

test_that("rls_fit with offset matches glm() (Gaussian)", {
  set.seed(42)
  n <- 500; p <- 3
  X   <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  off <- rnorm(n, 0, 0.5)
  y   <- rnorm(n, X %*% c(1, -0.5, 0.8) + off)

  expect_equal(
    unname(coef(rls_fit(X, y, offset = off))),
    unname(coef(glm(y ~ X - 1, family = gaussian(), offset = off))),
    tolerance = 1e-4
  )
})

test_that("rls_glm_fit with offset matches glm() oracle — intercept-only Poisson, cold start", {
  ## Intercept-only so Fisher information is well-conditioned and RLS-GLM
  ## converges from beta=0 without any initialization help.
  ## MLE is log(sum(y)/sum(exposure)) — computed by glm() independently.
  set.seed(42)
  n        <- 1000
  exposure <- runif(n, 1, 3)
  y        <- rpois(n, exposure * exp(0.4))

  fit_glm    <- glm(y ~ 1, family = poisson(), offset = log(exposure))
  fit_stream <- rls_glm_fit(matrix(1, n, 1), y, family = poisson(),
                             offset = log(exposure),
                             S0_scale = 10, eta_clip = 5)
  expect_equal(as.numeric(coef(fit_stream)), as.numeric(coef(fit_glm)),
               tolerance = 0.01)
})

test_that("isgd_fit with offset matches glm() oracle — intercept-only Poisson, cold start", {
  ## Same setup as above; gamma1 calibrated to ~ 1/(mean_weight * p) = 1/3.
  set.seed(42)
  n        <- 2000
  exposure <- runif(n, 1, 3)
  y        <- rpois(n, exposure * exp(0.4))

  fit_glm    <- glm(y ~ 1, family = poisson(), offset = log(exposure))
  fit_stream <- isgd_fit(matrix(1, n, 1), y, family = poisson(),
                          gamma1 = 0.3, alpha = 0.7,
                          offset = log(exposure), compute_vcov = FALSE)
  expect_equal(as.numeric(coef(fit_stream)), as.numeric(coef(fit_glm)),
               tolerance = 0.02)
})

## -----------------------------------------------------------------------
## 3. Algebraic identities — independent of convergence arguments
## -----------------------------------------------------------------------

test_that("rls_fit: offset absorbed into response gives identical coefficients (Gaussian identity)", {
  ## y = X*beta + o + eps  is the same model as  (y-o) = X*beta + eps.
  ## Both fits must return identical coefficients.
  set.seed(7)
  n <- 500; p <- 3
  X   <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  off <- rnorm(n, 0, 0.5)
  y   <- as.vector(X %*% c(1, -0.5, 0.8)) + off + rnorm(n)

  expect_equal(
    coef(rls_fit(X, y,       offset = off)),
    coef(rls_fit(X, y - off, offset = NULL))
  )
})

test_that("predict: Poisson multiplicative identity — offset is inside the link, not outside", {
  ## For Poisson log-link: linkinv(eta + log(e)) = exp(eta)*e.
  ## So predict(fit, X, offset=log(e)) must equal predict(fit, X) * e.
  ## If offset were applied after linkinv this identity would fail.
  set.seed(3)
  n <- 500; p <- 3
  X   <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y   <- rpois(n, exp(X %*% c(0.3, -0.2, 0.3)))
  fit <- rls_glm_fit(X, y, family = poisson(),
                     beta_init = c(log(mean(y)), rep(0, p - 1)),
                     S0_scale = 1, score_clip = 5)

  n_new    <- 20
  X_new    <- cbind(1, matrix(rnorm(n_new * (p - 1)), n_new, p - 1))
  exposure <- runif(n_new, 0.5, 3.0)

  expect_equal(
    predict(fit, X_new, offset = log(exposure)),
    predict(fit, X_new) * exposure
  )
})

## -----------------------------------------------------------------------
## 4. update() plumbing — offset forwarded to underlying *_update functions
## -----------------------------------------------------------------------

test_that("update.stream_fit passes offset to rls_glm_update", {
  set.seed(1)
  n <- 100; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rpois(n, exp(X %*% c(0.3, -0.2, 0.3)))
  fit <- rls_glm_fit(X, y, family = poisson(),
                     beta_init = c(log(mean(y)), rep(0, p - 1)),
                     S0_scale = 1, score_clip = 5)

  x_new <- c(1, rnorm(p - 1)); y_new <- 5L; off <- log(2.5)
  hp <- fit$hyperparams

  expected <- rls_glm_update(x_new, y_new, fit$beta, fit$S, poisson(),
                              lambda     = hp$lambda,
                              eta_clip   = hp$eta_clip,
                              score_clip = hp$score_clip,
                              offset     = off)$beta
  expect_equal(unname(coef(update(fit, x = x_new, y = y_new, offset = off))),
               expected)
})

test_that("update.stream_fit passes offset to rls_update", {
  set.seed(2)
  n <- 100; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- as.numeric(X %*% c(1, -0.5, 0.8) + rnorm(n))
  fit <- rls_fit(X, y)

  x_new <- c(1, rnorm(p - 1)); y_new <- 2.0; off <- 0.7
  hp    <- fit$hyperparams

  expected <- rls_update(x_new, y_new, fit$beta, fit$S,
                          lambda = hp$lambda, offset = off)$beta
  expect_equal(unname(coef(update(fit, x = x_new, y = y_new, offset = off))),
               expected)
})

test_that("update.stream_fit passes offset to isgd_update", {
  set.seed(3)
  n <- 100; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rpois(n, exp(X %*% c(0.3, -0.2, 0.3)))
  fit <- isgd_fit(X, y, family = poisson(), gamma1 = 0.5, compute_vcov = FALSE)

  x_new <- c(1, rnorm(p - 1)); y_new <- 5L; off <- log(2.5)
  hp    <- fit$hyperparams
  gamma <- hp$gamma1 * (fit$n_obs + 1L)^(-hp$alpha)

  expected <- isgd_update(x_new, y_new, fit$beta, gamma, poisson(), offset = off)
  expect_equal(unname(coef(update(fit, x = x_new, y = y_new, offset = off))),
               expected)
})

## -----------------------------------------------------------------------
## 5. Input validation
## -----------------------------------------------------------------------

test_that("wrong-length offset vector errors", {
  set.seed(1)
  n <- 100; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rnorm(n)

  expect_error(rls_fit(X, y,     offset = rep(0, n + 1)), "offset.*length n")
  expect_error(rls_glm_fit(X, y, family = gaussian(),
                            offset = rep(0, n - 1)),       "offset.*length n")
  expect_error(isgd_fit(X, y,    family = gaussian(),
                         offset = rep(0, 5)),              "offset.*length n")
})
