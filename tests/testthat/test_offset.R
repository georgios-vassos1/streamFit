## tests/testthat/test_offset.R
## Tests for offset support across all fitting functions

test_that("rls_glm_fit with Poisson offset converges to true beta", {
  set.seed(42)
  n <- 2000; p <- 3
  beta_true <- c(0.3, -0.2, 0.3)
  X <- cbind(1, matrix(rnorm(n * (p - 1), sd = 0.5), n, p - 1))
  exposure <- runif(n, 1, 2)
  eta <- as.vector(X %*% beta_true) + log(exposure)
  y <- rpois(n, exp(eta))

  fit <- rls_glm_fit(X, y, family = poisson(),
                     offset = log(exposure),
                     beta_init = c(log(mean(y / exposure)), rep(0, p - 1)),
                     score_clip = 3, S0_scale = 1, eta_clip = 5)
  expect_equal(unname(coef(fit)), beta_true, tolerance = 0.15)
})

test_that("isgd_fit with Poisson offset converges to true beta", {
  set.seed(42)
  n <- 3000; p <- 3
  beta_true <- c(0.3, -0.2, 0.3)
  X <- cbind(1, matrix(rnorm(n * (p - 1), sd = 0.5), n, p - 1))
  exposure <- runif(n, 1, 2)
  eta <- as.vector(X %*% beta_true) + log(exposure)
  y <- rpois(n, exp(eta))

  fit <- isgd_fit(X, y, family = poisson(),
                  gamma1 = 0.5, alpha = 0.7,
                  offset = log(exposure), compute_vcov = FALSE)
  expect_equal(unname(coef(fit)), beta_true, tolerance = 0.25)
})

test_that("rls_fit with offset converges to true beta", {
  set.seed(42)
  n <- 1000; p <- 3
  beta_true <- c(1, -0.5, 0.8)
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  off <- rnorm(n, 0, 0.5)
  y <- as.vector(X %*% beta_true) + off + rnorm(n)

  fit <- rls_fit(X, y, offset = off)
  expect_equal(unname(coef(fit)), beta_true, tolerance = 0.1)
})

test_that("rls_fit with offset matches batch glm", {
  set.seed(42)
  n <- 500; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  off <- rnorm(n, 0, 0.5)
  y <- rnorm(n, X %*% c(1, -0.5, 0.8) + off)

  fit_stream <- rls_fit(X, y, offset = off)
  fit_glm    <- glm(y ~ X - 1, family = gaussian(), offset = off)
  expect_equal(unname(coef(fit_stream)), unname(coef(fit_glm)), tolerance = 1e-4)
})

test_that("offset = NULL matches no-offset behaviour exactly", {
  set.seed(99)
  n <- 200; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rbinom(n, 1, 1 / (1 + exp(-X %*% c(-0.5, 1, -0.8))))

  fit_null   <- rls_glm_fit(X, y, family = binomial(), offset = NULL)
  fit_zero   <- rls_glm_fit(X, y, family = binomial(), offset = rep(0, n))
  expect_equal(coef(fit_null), coef(fit_zero))

  fit_null_rls <- rls_fit(X, y, offset = NULL)
  fit_zero_rls <- rls_fit(X, y, offset = rep(0, n))
  expect_equal(coef(fit_null_rls), coef(fit_zero_rls))

  fit_null_isgd <- isgd_fit(X, y, family = binomial(), offset = NULL,
                             compute_vcov = FALSE)
  fit_zero_isgd <- isgd_fit(X, y, family = binomial(), offset = rep(0, n),
                             compute_vcov = FALSE)
  expect_equal(coef(fit_null_isgd), coef(fit_zero_isgd))
})

test_that("predict with offset works correctly", {
  set.seed(7)
  n <- 500; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rpois(n, exp(X %*% c(0.3, -0.2, 0.3)))

  fit <- rls_glm_fit(X, y, family = poisson(),
                     beta_init = c(log(mean(y)), rep(0, p - 1)),
                     score_clip = 5, S0_scale = 1)

  n_new <- 10
  X_new <- cbind(1, matrix(rnorm(n_new * (p - 1)), n_new, p - 1))
  exposure_new <- runif(n_new, 0.5, 3.0)

  pred <- predict(fit, X_new, offset = log(exposure_new))
  expected <- poisson()$linkinv(as.vector(X_new %*% coef(fit)) + log(exposure_new))
  expect_equal(pred, expected)

  pred_link <- predict(fit, X_new, type = "link", offset = log(exposure_new))
  expected_link <- as.vector(X_new %*% coef(fit)) + log(exposure_new)
  expect_equal(pred_link, expected_link)

  # NULL offset should give same as no offset
  pred_null <- predict(fit, X_new, offset = NULL)
  pred_none <- predict(fit, X_new)
  expect_equal(pred_null, pred_none)
})

test_that("update.stream_fit with offset changes beta", {
  set.seed(1)
  n <- 100; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rpois(n, exp(X %*% c(0.3, -0.2, 0.3)))

  fit <- rls_glm_fit(X, y, family = poisson(),
                     beta_init = c(log(mean(y)), rep(0, p - 1)),
                     score_clip = 5, S0_scale = 1)
  beta_before <- coef(fit)
  x_new <- c(1, rnorm(p - 1))
  y_new <- 5
  fit2 <- update(fit, x = x_new, y = y_new, offset = log(2.0))
  expect_false(all(coef(fit2) == beta_before))

  # RLS branch
  fit_rls <- rls_fit(X, as.numeric(y))
  beta_rls_before <- coef(fit_rls)
  fit_rls2 <- update(fit_rls, x = x_new, y = 3.0, offset = 0.5)
  expect_false(all(coef(fit_rls2) == beta_rls_before))

  # iSGD branch
  fit_isgd <- isgd_fit(X, y, family = poisson(), gamma1 = 0.5,
                        compute_vcov = FALSE)
  beta_isgd_before <- coef(fit_isgd)
  fit_isgd2 <- update(fit_isgd, x = x_new, y = y_new, offset = log(2.0))
  expect_false(all(coef(fit_isgd2) == beta_isgd_before))
})

test_that("wrong-length offset vector errors", {
  set.seed(1)
  n <- 100; p <- 3
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  y <- rnorm(n)

  expect_error(rls_fit(X, y, offset = rep(0, n + 1)),
               "offset.*length n")
  expect_error(rls_glm_fit(X, y, family = gaussian(), offset = rep(0, n - 1)),
               "offset.*length n")
  expect_error(isgd_fit(X, y, family = gaussian(), offset = rep(0, 5)),
               "offset.*length n")
})
