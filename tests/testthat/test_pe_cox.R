## tests/testthat/test_pe_cox.R
## Unit tests for piecewise-exponential Cox model functions

# ── surv_to_pe: basic expansion correctness ──────────────────────────────

test_that("surv_to_pe produces correct records for a single subject", {
  # 1 subject, event at time 2.5, breaks at 0, 1, 2, 3
  pe <- surv_to_pe(
    start  = 0,
    stop   = 2.5,
    status = 1L,
    X      = matrix(0.5, nrow = 1, ncol = 1),
    breaks = c(0, 1, 2, 3)
  )
  expect_equal(length(pe$d), 3L)
  expect_equal(pe$e, c(1, 1, 0.5))
  expect_equal(pe$d, c(0L, 0L, 1L))
  expect_equal(pe$interval, c(1L, 2L, 3L))

  # interval dummies: one-hot per interval
  expect_equal(pe$X_pe[1, 1:3], c(1, 0, 0), ignore_attr = TRUE)
  expect_equal(pe$X_pe[2, 1:3], c(0, 1, 0), ignore_attr = TRUE)
  expect_equal(pe$X_pe[3, 1:3], c(0, 0, 1), ignore_attr = TRUE)

  # covariate column replicated
  expect_equal(pe$X_pe[, 4], c(0.5, 0.5, 0.5))
})


test_that("surv_to_pe handles censored subject correctly", {
  # censored at 1.5 => spans intervals 1 and 2 only, no event
  pe <- surv_to_pe(
    start  = 0,
    stop   = 1.5,
    status = 0L,
    X      = matrix(1, nrow = 1, ncol = 1),
    breaks = c(0, 1, 2, 3)
  )
  expect_equal(length(pe$d), 2L)
  expect_equal(pe$e, c(1, 0.5))
  expect_equal(pe$d, c(0L, 0L))
})


test_that("surv_to_pe handles multiple subjects", {
  pe <- surv_to_pe(
    start  = c(0, 0),
    stop   = c(1.5, 2.5),
    status = c(0L, 1L),
    X      = matrix(c(1, 2), nrow = 2, ncol = 1),
    breaks = c(0, 1, 2, 3)
  )
  # subject 1: 2 records, subject 2: 3 records

expect_equal(length(pe$d), 5L)
})


# ── surv_to_pe: edge cases ──────────────────────────────────────────────

test_that("surv_to_pe: subject censored before first interval produces no records", {
  pe <- surv_to_pe(
    start  = 0,
    stop   = 0,
    status = 0L,
    X      = matrix(1, nrow = 1, ncol = 1),
    breaks = c(0, 1, 2)
  )
  expect_equal(length(pe$d), 0L)
})


test_that("surv_to_pe: event exactly on a break boundary", {
  # event at time 2.0 exactly, breaks at 0, 1, 2, 3
  # the event falls in interval 2 (breaks[2], breaks[3]) = (1, 2]
  pe <- surv_to_pe(
    start  = 0,
    stop   = 2.0,
    status = 1L,
    X      = matrix(0, nrow = 1, ncol = 1),
    breaks = c(0, 1, 2, 3)
  )
  expect_equal(length(pe$d), 2L)
  expect_equal(pe$e, c(1, 1))
  # event in interval 2 because stop <= breaks[3] is TRUE,
  # but also stop <= breaks[2+1=3]... let's check: stop=2, breaks[3]=3, so

  # stop <= breaks[k+1]: for k=1: 2<=1? No -> d=0; for k=2: 2<=2? Yes -> d=1
  expect_equal(pe$d, c(0L, 1L))
})


test_that("surv_to_pe: counting process (delayed entry)", {
  # subject enters at time 1.5, event at 2.5, breaks at 0, 1, 2, 3
  pe <- surv_to_pe(
    start  = 1.5,
    stop   = 2.5,
    status = 1L,
    X      = matrix(0, nrow = 1, ncol = 1),
    breaks = c(0, 1, 2, 3)
  )
  # interval 1: max(1.5,0)=1.5, min(2.5,1)=1 => 1>1.5? No => skip
  # interval 2: max(1.5,1)=1.5, min(2.5,2)=2 => 2>1.5? Yes => e=0.5
  # interval 3: max(1.5,2)=2, min(2.5,3)=2.5 => 2.5>2? Yes => e=0.5
  expect_equal(length(pe$d), 2L)
  expect_equal(pe$e, c(0.5, 0.5))
  expect_equal(pe$d, c(0L, 1L))
  expect_equal(pe$interval, c(2L, 3L))
})


test_that("surv_to_pe validates inputs", {
  expect_error(surv_to_pe(0, c(1, 2), 1, matrix(1), c(0, 1)),
               "same length")
  expect_error(surv_to_pe(0, 1, 1, matrix(c(1, 2), ncol = 1), c(0, 1)),
               "nrow")
  expect_error(surv_to_pe(0, 1, 1, matrix(1), c(2, 1)),
               "sorted")
})


# ── pe_cox_fit: matches batch glm on PE data ────────────────────────────

test_that("pe_cox_fit matches glm() on piecewise-exponential data", {
  skip_on_cran()
  set.seed(123)
  n <- 800
  x1 <- rnorm(n)
  x2 <- rbinom(n, 1, 0.5)
  X <- cbind(x1 = x1, x2 = x2)

  # simple exponential survival
  lam0 <- 0.5
  beta_true <- c(0.5, -0.3)
  T_event <- rexp(n, rate = lam0 * exp(X %*% beta_true))
  C <- runif(n, 2, 5)
  time <- pmin(T_event, C)
  status <- as.integer(T_event <= C)

  breaks <- seq(0, max(time) + 0.01, length.out = 6)

  # pe_cox_fit
  fit_pe <- pe_cox_fit(rep(0, n), time, status, X, breaks,
                       method = "rls_glm", lambda = 1)

  # batch glm on same PE expansion
  pe <- surv_to_pe(rep(0, n), time, status, X, breaks)
  fit_glm <- glm(pe$d ~ pe$X_pe - 1, family = poisson(),
                  offset = log(pe$e))

  # coefficients should match within tolerance
  # (RLS-GLM is single-pass, so some difference is expected)
  expect_equal(fit_pe$beta, coef(fit_glm)[6:7], tolerance = 0.1,
               ignore_attr = TRUE)
})


# ── pe_cox_fit: exponential special case (K=1) ──────────────────────────

test_that("pe_cox_fit recovers exponential model with K=1", {
  skip_on_cran()
  set.seed(42)
  n <- 1000
  x1 <- rnorm(n)
  X <- matrix(x1, ncol = 1, dimnames = list(NULL, "x1"))
  beta_true <- 0.5
  T_event <- rexp(n, rate = 0.3 * exp(beta_true * x1))
  C <- runif(n, 3, 8)
  time <- pmin(T_event, C)
  status <- as.integer(T_event <= C)

  # single interval covering the whole range
  breaks <- c(0, max(time) + 0.01)

  fit <- pe_cox_fit(rep(0, n), time, status, X, breaks,
                    method = "rls_glm", lambda = 1)

  # covariate beta should be close to 0.5
  expect_equal(as.numeric(fit$beta), beta_true, tolerance = 0.15)

  # baseline should be close to log(0.3)
  expect_equal(as.numeric(fit$baseline_log_hazard), log(0.3),
               tolerance = 0.3)
})


# ── pe_cox_update: streaming produces comparable results ─────────────────

test_that("pe_cox_update produces sensible streaming estimates", {
  skip_on_cran()
  set.seed(99)
  n <- 300
  x1 <- rnorm(n)
  X <- matrix(x1, ncol = 1, dimnames = list(NULL, "x1"))
  T_event <- rexp(n, rate = 0.5 * exp(0.4 * x1))
  C <- runif(n, 2, 5)
  time <- pmin(T_event, C)
  status <- as.integer(T_event <= C)
  breaks <- c(0, 1, 2, 3, 4, max(time) + 0.01)

  # fit on first 200, stream remaining 100
  fit <- pe_cox_fit(rep(0, 200), time[1:200], status[1:200],
                    X[1:200, , drop = FALSE], breaks,
                    method = "rls_glm", lambda = 1)

  for (i in 201:n) {
    fit <- pe_cox_update(fit, 0, time[i], status[i], X[i, ])
  }

  # beta should be in a reasonable range around 0.4
  expect_true(abs(fit$beta - 0.4) < 0.3)
})



# ── pe_cox_fit: agreement with timereg::timecox (stationary) ─────────────

test_that("pe_cox_fit agrees with timereg::timecox on stationary Cox model", {
  skip_on_cran()
  skip_if_not_installed("timereg")
  skip_if_not_installed("survival")

  set.seed(77)
  n <- 1500
  x1 <- rnorm(n)
  x2 <- rbinom(n, 1, 0.5)
  X <- cbind(x1 = x1, x2 = x2)

  beta_true <- c(0.5, -0.3)
  lam0 <- 0.4
  T_event <- rexp(n, rate = lam0 * exp(X %*% beta_true))
  C <- runif(n, 2, 6)
  time <- pmin(T_event, C)
  status <- as.integer(T_event <= C)

  breaks <- seq(0, max(time) + 0.01, length.out = 11)

  # streamFit
  fit_pe <- pe_cox_fit(rep(0, n), time, status, X, breaks,
                       method = "rls_glm", lambda = 1)

  # timereg::timecox with const() for time-constant effects
  # const() must be called unqualified (timecox parses the formula symbolically)
  dat <- data.frame(time = time, status = status, x1 = x1, x2 = x2)
  const <- timereg::const
  fit_tc <- timereg::timecox(
    survival::Surv(time, status) ~ const(x1) + const(x2),
    data = dat, max.time = max(breaks) - 0.1
  )
  tc_beta <- fit_tc$gamma[, 1]  # extract point estimates

  # Both should be close to the truth and to each other
  # streamFit vs timecox: within 0.15 (single-pass vs kernel-smoothed batch)
  expect_equal(as.numeric(fit_pe$beta), tc_beta, tolerance = 0.15,
               ignore_attr = TRUE)

  # Both should be within 0.15 of truth
  expect_equal(as.numeric(fit_pe$beta), beta_true, tolerance = 0.15,
               ignore_attr = TRUE)
  expect_equal(tc_beta, beta_true, tolerance = 0.15,
               ignore_attr = TRUE)
})


# ── pe_cox_fit: batch GLM on PE data matches timereg::timecox ────────────

test_that("batch PE Poisson GLM matches timereg::timecox on constant-beta data", {
  skip_on_cran()
  skip_if_not_installed("timereg")
  skip_if_not_installed("survival")

  set.seed(55)
  n <- 1200
  x1 <- rnorm(n)
  X <- cbind(x1 = x1)

  beta_true <- 0.6
  T_event <- rexp(n, rate = 0.3 * exp(beta_true * x1))
  C <- runif(n, 2, 6)
  time <- pmin(T_event, C)
  status <- as.integer(T_event <= C)

  breaks <- seq(0, max(time) + 0.01, length.out = 8)

  # Batch PE Poisson GLM
  pe <- surv_to_pe(rep(0, n), time, status, X, breaks)
  fit_glm <- glm(pe$d ~ pe$X_pe - 1, family = poisson(), offset = log(pe$e))
  glm_beta <- coef(fit_glm)[length(breaks)]  # last coef = x1

  # timereg::timecox
  dat <- data.frame(time = time, status = status, x1 = x1)
  const <- timereg::const
  fit_tc <- timereg::timecox(
    survival::Surv(time, status) ~ const(x1),
    data = dat, max.time = max(breaks) - 0.1
  )
  tc_beta <- fit_tc$gamma[1, 1]

  # PE Poisson GLM and timecox should agree closely (both are batch methods)
  expect_equal(as.numeric(glm_beta), tc_beta, tolerance = 0.1)
})
