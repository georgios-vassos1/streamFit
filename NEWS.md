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
