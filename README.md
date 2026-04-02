# streamFit

Online sequential parameter estimation for streaming data.

`streamFit` implements two complementary algorithms for fitting generalised
linear models (GLMs) observation-by-observation, without ever storing or
refitting on the full dataset.  Both algorithms support a forgetting factor
or adaptive learning rate for tracking **time-varying parameters**.

| Algorithm | Cost / step | Storage | Notes |
|-----------|-------------|---------|-------|
| RLS-GLM   | O(p²)       | O(p²)   | Second-order; Sherman-Morrison Fisher update |
| iSGD      | O(p)        | O(p)    | First-order; implicit proximal update |

The package is designed for extension to other streaming estimation problems
(survival models, state-space models, etc.).

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
out <- rls_update(x, y, beta, S, lambda = 1)
# out$beta  — updated coefficients
# out$S     — updated inverse covariance

# Full dataset
fit <- rls_fit(X, y, lambda = 1, S0_scale = 100, beta_init = NULL)
```

### Generalised linear models

```r
# Single-step update — works with any R family object
out <- rls_glm_update(x, y, beta, S, family,
                      lambda = 1, eta_clip = 10, score_clip = NULL)

# Full dataset
fit <- rls_glm_fit(X, y, family = gaussian(),
                   lambda     = 1,       # forgetting factor
                   S0_scale   = 100,     # initial inverse Fisher scale
                   eta_clip   = 10,      # linear predictor clipping
                   beta_init  = NULL,    # starting coefficients
                   score_clip = NULL)    # score clipping (Poisson)
```

### Implicit SGD

```r
# Single-step update
beta <- isgd_update(x, y, beta, gamma, family)

# Full dataset
fit <- isgd_fit(X, y, family = gaussian(),
                gamma1 = 1,    # initial learning rate
                alpha  = 0.7)  # decay exponent, must be in (0.5, 1]
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

## Running the tests

```r
devtools::test("~/streamFit")
```

Or from the shell:

```bash
R CMD check --no-manual ~/streamFit
```

---

## Extending the package

To add a new estimator (e.g. a Cox model via the piecewise exponential
reformulation), write a new file in `R/` that:

1. Implements a single-step `*_update()` function.
2. Implements a `*_fit()` function that loops over observations and calls
   `new_stream_fit()` at the end.

The `stream_fit` S3 class, `print`, `coef`, `plot`, `predict`, `update`,
and `conv_path` are then available automatically.

---

## References

- Fahrmeir, L. (1992). Posterior mode estimation by extended Kalman filtering
  for multivariate dynamic generalised linear models. *JASA*, 87(418), 501-509.
- Toulis, P. and Airoldi, E. (2017). Asymptotic and finite-sample properties
  of estimators based on stochastic gradients. *Annals of Statistics*, 45(4),
  1694-1727.
- Amari, S.-I. (1998). Natural gradient works efficiently in learning.
  *Neural Computation*, 10(2), 251-276.
