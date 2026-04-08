# streamFit

Online sequential parameter estimation and causal inference for streaming data.

`streamFit` provides two complementary families of estimators:

**Streaming GLMs** — fit generalised linear models observation-by-observation,
without ever storing or refitting on the full dataset.

**Online causal estimators** — doubly-robust ATE / ATT estimation and Targeted
Maximum Likelihood Estimation (TMLE) with nuisance models that update as new
observations arrive.

| Module | Class | Cost / step |
|--------|-------|-------------|
| RLS-GLM | `stream_fit` | O(p²) |
| Implicit SGD | `stream_fit` | O(p) |
| Piecewise-exponential Cox | `pe_cox_fit` | O(p²) |
| Batch Super Learner | `super_learner_fit` | O(n · L) |
| Online Super Learner | `online_sl` | O(L) |
| Online ATE / ATT (one-step) | `online_ate` | O(L) |
| Online TMLE | `online_tmle` | O(L) |

---

## Installation

```r
install.packages("devtools")
devtools::install("~/streamFit")
```

---

## Quick start — streaming GLM

```r
library(streamFit)

set.seed(42)
n <- 1000; p <- 4
X     <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta0 <- c(-0.5, 1, -1, 0.5)
y     <- rbinom(n, 1, plogis(X %*% beta0))

## Second-order online fit (RLS-GLM) on first 800 observations
fit <- rls_glm_fit(X[1:800, ], y[1:800], family = binomial())

## Stream in the remaining 200 observations one at a time
for (i in 801:n)
  fit <- update(fit, x = X[i, ], y = y[i])

coef(fit)
confint(fit)
predict(fit, X[1:5, ], type = "response")
```

## Quick start — online TMLE

```r
library(streamFit)

set.seed(1)
n <- 500
W <- cbind(1, matrix(rnorm(n * 2), n, 2))   # intercept + 2 covariates
A <- rbinom(n, 1, plogis(0.4 * W[, 2]))
Y <- 1.0 * A + 0.5 * W[, 2] + rnorm(n, sd = 0.5)   # true ATE = 1

## Define a learner using online RLS-GLM
lib <- list(
  rls = make_learner(
    fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
    predict = function(model, newdata, ...) predict(model, newdata),
    update  = function(model, x, y, family, ...) update(model, x = x, y = y)
  )
)

## Fit TMLE on batch data
tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib)
print(tmle)
confint(tmle)   # 95% CI should cover 1.0

## Stream in new observations
for (i in seq_len(200)) {
  w_i <- c(1, rnorm(2))
  a_i <- rbinom(1, 1, plogis(0.4 * w_i[2]))
  y_i <- 1.0 * a_i + 0.5 * w_i[2] + rnorm(1, sd = 0.5)
  tmle <- update(tmle, w = w_i, a = a_i, y = y_i)
}
confint(tmle)
```

---

## Streaming GLM

### Recursive Least Squares — linear (RLS)

```r
## Low-level single-step update
out <- rls_update(x, y, beta, S, lambda = 1, offset = 0)
# out$beta  -- updated coefficients
# out$S     -- updated inverse Fisher

## Batch initialisation
fit <- rls_fit(X, y, lambda = 1, S0_scale = 100,
               beta_init = NULL, offset = NULL)
```

### Recursive Least Squares — GLM (RLS-GLM)

```r
## Low-level single-step update
out <- rls_glm_update(x, y, beta, S, family,
                      lambda = 1, eta_clip = 10, score_clip = NULL,
                      offset = 0, S_max = NULL)

## Batch initialisation
fit <- rls_glm_fit(X, y, family = gaussian(),
                   lambda     = 1,      # forgetting factor
                   S0_scale   = 100,    # initial inverse Fisher scale
                   eta_clip   = 10,     # linear predictor clipping
                   beta_init  = NULL,
                   score_clip = NULL,   # Huber-type score clipping (Poisson)
                   offset     = NULL,
                   S_max      = NULL)   # diagonal covariance clamping
```

Any R `family` object works: `gaussian()`, `binomial()`, `poisson()`,
`Gamma(link = "log")`, etc.

### Implicit SGD (iSGD)

```r
## Low-level single-step update
beta <- isgd_update(x, y, beta, gamma, family, offset = 0)

## Batch initialisation
fit <- isgd_fit(X, y, family = gaussian(),
                gamma1 = 1,    # initial learning rate
                alpha  = 0.7,  # Robbins–Monro decay exponent in (0.5, 1]
                offset = NULL)
```

### S3 methods on `stream_fit` objects

```r
print(fit)              # method, family, final coefficients, hyperparams
coef(fit)               # final coefficient vector
nobs(fit)               # number of observations processed
vcov(fit)               # variance-covariance matrix (from S)
summary(fit)            # coefficient table with SEs, z/t values, p-values
confint(fit)            # Wald confidence intervals
plot(fit)               # coefficient paths over observations
plot(fit, ref = beta0)  # L2 convergence to a reference vector
predict(fit, Xnew)      # predicted responses for new data
update(fit, x, y)       # single-step streaming update
```

---

## Piecewise-exponential Cox model

A Cox proportional hazards model with piecewise-constant baseline hazard recast
as a Poisson GLM with `offset = log(exposure)`.

```r
## Expand survival data into person-interval records
pe <- surv_to_pe(start, stop, status, X, breaks)

## Batch fit
fit <- pe_cox_fit(start, stop, status, X, breaks,
                  method = "rls_glm", lambda = 1)
fit$beta                  # covariate log-hazard-ratios
fit$baseline_log_hazard   # log baseline hazard per interval

## Stream a single new subject
fit <- pe_cox_update(fit, start_new, stop_new, status_new, x_new)
```

With `lambda < 1`, the online estimator tracks time-varying coefficients.
Use `S_max` to prevent covariance windup on the sparse interval-dummy columns:

```r
fit <- pe_cox_fit(start, stop, status, X, breaks,
                  method = "rls_glm", lambda = 0.999, S_max = 1.0)
```

---

## Super Learner

### Batch Super Learner

Cross-validated ensemble of base learners with deviance-optimal convex weights.

```r
lib <- list(
  rls = make_learner(
    fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
    predict = function(model, newdata, ...) predict(model, newdata),
    update  = function(model, x, y, family, ...) update(model, x = x, y = y)
  )
)

sl <- super_learner_fit(X, y, family = gaussian(), library = lib, k = 5L)
coef(sl)              # ensemble weights
predict(sl, Xnew)     # ensemble predictions
```

### Online Super Learner

Initialises from a batch Super Learner, then updates ensemble weights (and
optionally base learners) one observation at a time via SGD on the deviance loss.

```r
osl <- online_sl_fit(X, y, family = gaussian(), library = lib,
                     k      = 5L,   # CV folds for initial batch SL
                     gamma1 = 1,    # initial SGD learning rate
                     delta  = 0.6)  # Robbins–Monro decay exponent

## Stream in new observations
osl <- update(osl, x = x_new, y = y_new)

coef(osl)             # current ensemble weights
predict(osl, Xnew)    # current ensemble predictions
```

Base learners created with a non-`NULL` `update` argument in `make_learner()`
are updated online alongside the ensemble weights.  Learners without an
`update` function are held fixed at their batch estimates.

---

## Online causal estimators

All causal estimators share the same learner interface (`make_learner`) and the
same predict-then-update protocol: nuisance predictions for observation `i` are
computed before updating the models with observation `i`.

**Intercept convention**: callers must supply the intercept column in `W`.
The outcome model receives `cbind(A, W)` as features; the propensity model
receives `W`.

### Online ATE / ATT (one-step AIPW)

Doubly-robust, semiparametrically efficient one-step estimator.

```r
ate <- online_ate_fit(W, A, Y,
                      Q_library      = lib,
                      g_library      = lib,
                      outcome_family = gaussian(),  # outcome model family
                      target         = "ATE",       # or "ATT"
                      k              = 5L,
                      g_clamp        = 0.01,        # propensity clamping
                      gamma1         = 1.0,
                      delta          = 0.6)

coef(ate)              # point estimate (psi)
confint(ate)           # 95% Wald CI
print(ate)             # psi, SE, CI, n

## Streaming update
ate <- update(ate, w = w_new, a = a_new, y = y_new)
```

The estimator is consistent if either the outcome model `Q` or the propensity
score `g` is correctly specified (double robustness).

### Online TMLE

Extends the one-step estimator with a targeting step that fluctuates `Q` along
a one-dimensional submodel indexed by `ε`, solving the EIF score equation
`mean(H * (Y − Q*(A,W))) ≈ 0`.  This achieves second-order bias reduction
(`O_p(n⁻¹)` remainder vs `O_p(n⁻¹/²)`) and asymptotic efficiency.

```r
tmle <- online_tmle_fit(W, A, Y,
                        Q_library       = lib,
                        g_library       = lib,
                        outcome_family  = gaussian(),
                        target          = "ATE",      # or "ATT"
                        k               = 5L,
                        g_clamp         = 0.01,
                        max_iter        = 10L,        # targeting iterations
                        tol             = 1e-6,       # score-equation threshold
                        sequential_init = FALSE,      # see CARA designs below
                        variance_type   = "iid")      # or "martingale"

coef(tmle)             # point estimate (psi)
confint(tmle)          # 95% Wald CI
print(tmle)            # psi, SE, CI, epsilon, n

## Streaming update (predict-then-update; epsilon updated incrementally)
tmle <- update(tmle, w = w_new, a = a_new, y = y_new)
```

#### Iterative targeting

The `max_iter` and `tol` parameters control the iterative fluctuation loop.
For Gaussian outcomes, one iteration solves the score equation exactly (OLS
normal equations).  For non-Gaussian families (e.g. `binomial()`), multiple
iterations are needed:

```r
tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib,
                        outcome_family = binomial(),
                        max_iter = 20L, tol = 1e-4)
tmle$iter_converged   # TRUE if |mean(H*(Y-Q*a))| < tol
tmle$n_iter           # number of iterations run
tmle$epsilon          # cumulative targeting parameter
```

#### CARA adaptive designs

Under covariate-adaptive randomisation (CARA), observation `i` was collected
under mechanism `G_i` (the allocation rule after observing `1, …, i−1`).  Using
a pooled `G̃` introduces bias in the EIF score equation.  Set
`sequential_init = TRUE` to replay batch observations sequentially so that each
`g_c[i]` is predicted from the model trained on observations `1, …, i−1`:

```r
tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib,
                        sequential_init = TRUE)
```

The streaming phase (`update.online_tmle`) already uses stage-specific `G_i`
by the predict-then-update protocol regardless of `sequential_init`.

#### Martingale CLT variance

Under a fixed treatment mechanism the default `variance_type = "iid"` uses the
sample variance of TMLE EIF contributions.  Under adaptive `G_t`, set
`variance_type = "martingale"` to use the uncentered second moment
`V_t = (1/t) Σ [D*(P̂_i)(O_i)]²` (Chambaz & van der Laan 2011; Chambaz,
Zheng & van der Laan 2017):

```r
tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib,
                        sequential_init = TRUE,
                        variance_type   = "martingale")
```

The running sum `eif_sq_sum = Σ D*(P̂_i)²` is always tracked and available on
the returned object regardless of `variance_type`.

---

## Utilities

```r
## L2 distance from coefficient path to a reference vector
d <- conv_path(fit$beta_path, ref = beta0)
plot(d, type = "l")
```

---

## Tracking time-varying parameters

### RLS-GLM: forgetting factor

```r
## lambda < 1 discounts old observations; effective window ≈ 1 / (1 − lambda)
fit <- rls_glm_fit(X, y, family = binomial(),
                   lambda   = 0.98,   # window ~ 50 observations
                   S0_scale = 0.01)
```

### iSGD: slow decay schedule

```r
## alpha close to 0.5 keeps the learning rate large at late steps
fit <- isgd_fit(X, y, family = binomial(),
                gamma1 = 3,     # calibrate to ~ 1/(w*p), w = IRLS weight
                alpha  = 0.51)
```

### S-diagonal clamping (`S_max`)

Prevents covariance windup on parameters that are not regularly updated
(e.g. interval dummies in piecewise-exponential models):

```r
fit <- rls_glm_fit(X, y, family = poisson(),
                   lambda = 0.99, S_max = 1.0)
```

---

## Running the tests

```r
devtools::test("~/streamFit")
devtools::test("~/streamFit", filter = "online_tmle")   # single module
```

Or from the shell:

```bash
R CMD check --no-manual ~/streamFit
```

---

## References

**Streaming GLM**

- Fahrmeir, L. (1992). Posterior mode estimation by extended Kalman filtering
  for multivariate dynamic generalised linear models. *JASA*, 87(418), 501–509.
- Toulis, P. and Airoldi, E. (2017). Asymptotic and finite-sample properties of
  estimators based on stochastic gradients. *Annals of Statistics*, 45(4),
  1694–1727.
- Amari, S.-I. (1998). Natural gradient works efficiently in learning.
  *Neural Computation*, 10(2), 251–276.

**Survival analysis**

- Friedman, M. (1982). Piecewise exponential models for survival data with
  covariates. *Annals of Statistics*, 10(1), 101–113.
- Martinussen, T. and Scheike, T.H. (2006). *Dynamic Regression Models for
  Survival Data*. Springer.

**Super Learner**

- van der Laan, M.J., Polley, E.C. and Hubbard, A.E. (2007). Super Learner.
  *Statistical Applications in Genetics and Molecular Biology*, 6(1).

**Causal inference / TMLE**

- van der Laan, M.J. and Rose, S. (2011). *Targeted Learning: Causal Inference
  for Observational and Experimental Data*. Springer.
- Chambaz, A. and van der Laan, M.J. (2011). Targeting the optimal design in
  randomized clinical trials with binary outcomes and no covariate.
  *The International Journal of Biostatistics*, 7(1). DOI: 10.2202/1557-4679.1247
- Chambaz, A., Zheng, W. and van der Laan, M.J. (2017). Targeted sequential
  design for targeted learning inference of the optimal treatment rule and its
  mean reward. *Annals of Statistics*, 45(6), 2537–2564.
  DOI: 10.1214/16-AOS1534
