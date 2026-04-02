# streamFit

Online sequential parameter estimation for streaming data.

`streamFit` implements two complementary algorithms for fitting generalised
linear models (GLMs) observation-by-observation, without ever storing or
refitting on the full dataset.  Both algorithms support a forgetting factor
or adaptive learning rate for tracking **time-varying parameters**.

| Algorithm | Cost / step | Storage | Notes |
|-----------|-------------|---------|-------|
| RLS-GLM   | O(p^2)      | O(p^2)  | Second-order; Sherman-Morrison Fisher update |
| iSGD      | O(p)        | O(p)    | First-order; implicit proximal update |

On top of these core estimators, `streamFit` provides a **piecewise-exponential
Cox model** layer (`pe_cox_fit` / `pe_cox_update`) that recasts survival
analysis as a Poisson GLM with offset, enabling fully online estimation of
hazard ratios and baseline hazards.

---

## Installation

```r
# From local source
install.packages("devtools")
devtools::install("~/streamFit")
```

---

## Quick start

```r
library(streamFit)

set.seed(42)
n <- 1000; p <- 4
X     <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta0 <- c(-0.5, 1, -1, 0.5)
y     <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta0)))

## Second-order online fit (RLS-GLM) on first 800 observations
fit_rls <- rls_glm_fit(X[1:800, ], y[1:800], family = binomial())
coef(fit_rls)

## Stream in the remaining 200 observations one at a time
for (i in 801:n) {
  fit_rls <- update(fit_rls, x = X[i, ], y = y[i])
}
coef(fit_rls)

## Predict on new data
Xnew <- cbind(1, matrix(rnorm(5 * (p - 1)), 5, p - 1))
predict(fit_rls, Xnew, type = "response")   # probabilities

## First-order online fit (iSGD)
fit_isgd <- isgd_fit(X, y, family = binomial(), gamma1 = 1, alpha = 0.7)
coef(fit_isgd)

## Compare convergence to truth
plot(fit_rls,  ref = beta0, col = "steelblue", type = "l")
lines(conv_path(fit_isgd$beta_path, beta0), col = "tomato")
```

---

## Functions

### Linear model (Gaussian)

```r
# Single-step update
out <- rls_update(x, y, beta, S, lambda = 1, offset = 0)
# out$beta  -- updated coefficients
# out$S     -- updated inverse covariance

# Full dataset
fit <- rls_fit(X, y, lambda = 1, S0_scale = 100, beta_init = NULL,
               offset = NULL)
```

### Generalised linear models

```r
# Single-step update -- works with any R family object
out <- rls_glm_update(x, y, beta, S, family,
                      lambda = 1, eta_clip = 10, score_clip = NULL,
                      offset = 0, S_max = NULL)

# Full dataset
fit <- rls_glm_fit(X, y, family = gaussian(),
                   lambda     = 1,       # forgetting factor
                   S0_scale   = 100,     # initial inverse Fisher scale
                   eta_clip   = 10,      # linear predictor clipping
                   beta_init  = NULL,    # starting coefficients
                   score_clip = NULL,    # score clipping (Poisson)
                   offset     = NULL,    # e.g. log(exposure)
                   S_max      = NULL)    # covariance clamping bound
```

### Implicit SGD

```r
# Single-step update
beta <- isgd_update(x, y, beta, gamma, family, offset = 0)

# Full dataset
fit <- isgd_fit(X, y, family = gaussian(),
                gamma1 = 1,    # initial learning rate
                alpha  = 0.7,  # decay exponent, must be in (0.5, 1]
                offset = NULL)
```

### Piecewise-exponential Cox model

```r
# Expand survival data into person-interval records
pe <- surv_to_pe(start, stop, status, X, breaks)

# Fit an online Cox model (uses Poisson GLM with offset = log(exposure))
fit <- pe_cox_fit(start, stop, status, X, breaks,
                  method = "rls_glm", lambda = 1, ...)

# Stream a single new observation
fit <- pe_cox_update(fit, start_new, stop_new, status_new, x_new)
```

### Utilities

```r
# L2 distance from coefficient path to a reference vector
d <- conv_path(fit$beta_path, ref = beta0)
plot(d, type = "l")
```

### S3 methods on `stream_fit` objects

```r
print(fit)              # method, family, final coefficients, hyperparams
coef(fit)               # final coefficient vector
nobs(fit)               # number of observations processed
vcov(fit)               # variance-covariance matrix of coefficients
summary(fit)            # coefficient table with SEs, z/t values, p-values
confint(fit)            # Wald confidence intervals
plot(fit)               # coefficient paths over observations
plot(fit, ref = beta0)  # L2 convergence to reference
predict(fit, Xnew)      # predicted responses for new data
update(fit, x, y)       # single-step streaming update with one new observation
```

---

## Supported families

Any R `family` object works:

```r
rls_glm_fit(X, y, family = binomial())   # logistic regression
rls_glm_fit(X, y, family = poisson())    # Poisson regression
rls_glm_fit(X, y, family = gaussian())   # linear regression (= rls_fit)
rls_glm_fit(X, y, family = Gamma(link = "log"))
```

---

## Offset support

All fitting functions accept an `offset` parameter (a numeric vector of length
`n` added to the linear predictor before applying the link). Common uses:

```r
# Poisson rate model with known exposure
fit <- rls_glm_fit(X, y, family = poisson(), offset = log(exposure))

# Streaming with offset
fit <- update(fit, x = x_new, y = y_new, offset = log(exposure_new))

# Prediction with offset
predict(fit, Xnew, offset = log(exposure_new))
```

---

## Tracking time-varying parameters

### RLS-GLM: forgetting factor

```r
# lambda < 1 discounts old observations
# effective memory window = 1 / (1 - lambda)
fit <- rls_glm_fit(X, y, family = binomial(),
                   lambda   = 0.98,   # window ~ 50 observations
                   S0_scale = 0.01)   # small S0 makes convergence arc visible
```

### iSGD: slow decay schedule

```r
# For tracking, alpha close to 0.5 keeps the learning rate large at late steps.
# Calibrate gamma1 ~ 1 / (w * p) where w is the expected IRLS weight:
#   binomial (balanced): w ~ 0.25  =>  gamma1 ~ 4/p
#   poisson:             w ~ mean(y) =>  gamma1 ~ 1/(p * mean(y))
fit <- isgd_fit(X, y, family = binomial(),
                gamma1 = 3,     # ~ 1/w for p=3
                alpha  = 0.51)  # slowest valid decay
```

### S-diagonal clamping (`S_max`)

When using `lambda < 1` with sparse design matrices (e.g. interval dummies in
piecewise-exponential models), the inverse Fisher information `S` can blow up
for parameters that are not regularly updated ("covariance windup"). The `S_max`
parameter prevents this by clamping the diagonal of `S`:

```r
fit <- rls_glm_fit(X, y, family = poisson(),
                   lambda = 0.99, S_max = 1.0)
```

After each update, any `diag(S)[j] > S_max` is rescaled symmetrically back to
`S_max` while preserving the correlation structure.

---

## Poisson-specific notes

Poisson responses are unbounded. For RLS-GLM, two modifications prevent
divergence from extreme count observations:

```r
fit <- rls_glm_fit(X, y, family = poisson(),
                   S0_scale   = 0.1,                          # small initial step
                   beta_init  = c(log(mean(y)), rep(0, p-1)), # intercept warm start
                   score_clip = 5 * sqrt(mean(y)))            # Huber-type clipping
```

iSGD handles extreme counts automatically via the implicit equation and
requires no special modification.

---

## Online piecewise-exponential Cox model

A Cox proportional hazards model with piecewise-constant baseline hazard can be
recast as a Poisson GLM. `streamFit` provides three functions for this:

- `surv_to_pe()` -- expands counting-process `(start, stop, status)` data into
  person-interval records with interval dummies and `offset = log(exposure)`
- `pe_cox_fit()` -- convenience wrapper that expands + fits in one call
- `pe_cox_update()` -- streams a single new subject through the fitted model

```r
set.seed(42)
n <- 500
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.5)
X <- cbind(x1, x2)
T_event <- rexp(n, rate = 0.5 * exp(0.5 * x1 - 0.3 * x2))
C <- runif(n, 2, 5)
time <- pmin(T_event, C); status <- as.integer(T_event <= C)
breaks <- seq(0, max(time) + 0.01, length.out = 6)

# Batch fit
fit <- pe_cox_fit(rep(0, n), time, status, X, breaks,
                  method = "rls_glm", lambda = 1)
fit$beta                  # covariate effects (log hazard-ratios)
fit$baseline_log_hazard   # log baseline hazard per interval

# Streaming with forgetting (tracks time-varying coefficients)
fit <- pe_cox_fit(rep(0, 400), time[1:400], status[1:400],
                  X[1:400, ], breaks,
                  method = "rls_glm", lambda = 0.999, S_max = 1.0)
for (i in 401:n) {
  fit <- pe_cox_update(fit, 0, time[i], status[i], X[i, ])
}
fit$beta
```

With `lambda < 1`, the online estimator tracks time-varying regression
coefficients, analogous to `timereg::timecox()` in batch mode. Use `S_max`
to prevent covariance windup on the sparse interval-dummy columns.

---

## Running the tests

```r
devtools::test("~/streamFit")
```

Or from the shell:

```bash
R CMD check --no-manual ~/streamFit
```

---

## References

- Fahrmeir, L. (1992). Posterior mode estimation by extended Kalman filtering
  for multivariate dynamic generalised linear models. *JASA*, 87(418), 501-509.
- Toulis, P. and Airoldi, E. (2017). Asymptotic and finite-sample properties
  of estimators based on stochastic gradients. *Annals of Statistics*, 45(4),
  1694-1727.
- Amari, S.-I. (1998). Natural gradient works efficiently in learning.
  *Neural Computation*, 10(2), 251-276.
- Friedman, M. (1982). Piecewise exponential models for survival data with
  covariates. *Annals of Statistics*, 10(1), 101-113.
- Martinussen, T. and Scheike, T.H. (2006). *Dynamic Regression Models for
  Survival Data*. Springer.
