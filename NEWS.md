# streamFit 0.2.0 (development)

## New modules

* **`make_learner()`**: constructs `sl_learner` objects that wrap any
  `fit` / `predict` / `update` triple behind the common Super Learner
  interface.  The `update` slot is optional; learners without it are held
  fixed during online ensemble weight updates.

* **Batch Super Learner** (`super_learner_fit()`): cross-validated ensemble
  with deviance-optimal convex weights.  S3 methods: `coef`, `predict`,
  `print`.

* **Online Super Learner** (`online_sl_fit()`, `update.online_sl()`):
  initialises from a batch Super Learner then updates ensemble weights
  one observation at a time via SGD on the deviance loss (Robbins–Monro
  schedule `γ₁ t^{−δ}`).  Base learners are updated online when their
  `update` function is provided.  S3 methods: `coef`, `predict`, `print`,
  `update`.

* **Online ATE / ATT estimator** (`online_ate_fit()`, `update.online_ate()`):
  doubly-robust one-step AIPW estimator for the average treatment effect
  (ATE) or average treatment effect on the treated (ATT).  Nuisance models
  (`Q` and `g`) are fitted and updated online via the Online Super Learner.
  Propensity scores are clamped to `(g_clamp, 1 − g_clamp)` to prevent
  overflow in the clever covariate `H = A/g − (1−A)/(1−g)`.  Welford
  running variance gives a streaming standard error.  S3 methods: `coef`,
  `confint`, `print`, `update`.

* **Online TMLE** (`online_tmle_fit()`, `update.online_tmle()`): targeted
  maximum likelihood estimator that extends the one-step ATE / ATT estimator
  with a one-dimensional targeting step, fluctuating `Q` along the submodel
  `Q*(A,W) = linkinv(linkfun(Q(A,W)) + ε H(A,W))`.  Achieves second-order
  bias reduction (`O_p(n⁻¹)` remainder) and asymptotic efficiency.  S3
  methods: `coef`, `confint`, `print`, `update`.

## Online TMLE — targeting details

* **Iterative targeting** (`max_iter`, `tol`): the fluctuation step is
  iterated until `|mean(H*(Y − Q*a))| < tol` or `max_iter` is reached.
  `iter_converged` and `n_iter` fields on the returned object report
  convergence status.  For Gaussian outcomes the score equation is satisfied
  exactly in one pass (OLS normal equations); non-Gaussian families
  (e.g. `binomial()`) require genuine iteration.

* **Stage-specific clever covariate** (`sequential_init = FALSE`): under
  CARA adaptive designs, observation `i` was collected under mechanism `G_i`
  (the allocation rule after observing `1, …, i−1`).  Setting
  `sequential_init = TRUE` replays batch observations sequentially so that
  `g_c[i]` is predicted from the propensity model trained on observations
  `1, …, i−1`, then updated.  The streaming phase already uses stage-specific
  `G_i` via the predict-then-update protocol regardless of this flag.

* **Martingale CLT variance** (`variance_type = "iid"` / `"martingale"`):
  under a fixed treatment mechanism the default `"iid"` uses the
  Bessel-corrected sample variance of TMLE EIF contributions.  Under
  adaptive `G_t`, `"martingale"` uses the uncentered second moment
  `V_t = (1/t) Σ [D*(P̂_i)(O_i)]²` (Chambaz & van der Laan 2011;
  Chambaz, Zheng & van der Laan 2017).  The running sum `eif_sq_sum` is
  always tracked and available on the object.

---

# streamFit 0.1.0

* Initial release.
* `rls_fit()` / `rls_update()`: online RLS for linear (Gaussian) models.
* `rls_glm_fit()` / `rls_glm_update()`: online RLS-GLM for any R family object
  via Sherman-Morrison Fisher information updates.
* `isgd_fit()` / `isgd_update()`: online implicit SGD for any R family object.
* `update.stream_fit()`: single-step streaming update on a fitted object.
* `predict.stream_fit()`: predictions on new data, with optional offset.
* `vcov.stream_fit()`: variance-covariance matrix (inverse Fisher for RLS-GLM,
  sandwich estimator for iSGD).
* `summary.stream_fit()` / `confint.stream_fit()`: Wald inference.
* `conv_path()`: L2 convergence path utility.
* **Offset support**: all fitting functions (`rls_fit`, `rls_glm_fit`,
  `isgd_fit`), `update()`, and `predict()` accept an `offset` parameter for
  models with known exposure (e.g. `log(exposure)` in Poisson rate models).
* **Piecewise-exponential Cox model**:
    - `surv_to_pe()`: expand counting-process survival data into person-interval
      records suitable for Poisson GLM fitting.
    - `pe_cox_fit()`: convenience wrapper that expands and fits an online Cox
      model via RLS-GLM or iSGD with `offset = log(exposure)`.
    - `pe_cox_update()`: stream a single new subject through a fitted PE Cox
      model.
* **S-diagonal clamping** (`S_max` parameter): prevents covariance windup when
  using `lambda < 1` with sparse design matrices. Available in
  `rls_glm_update()`, `rls_glm_fit()`, and propagated through
  `update.stream_fit()` and `pe_cox_fit()` / `pe_cox_update()`.
