# streamFit 0.1.0

* Initial release.
* `rls_fit()`: online RLS for linear (Gaussian) models.
* `rls_glm_fit()`: online RLS-GLM for any R family object via Sherman-Morrison
  Fisher information updates.
* `isgd_fit()`: online implicit SGD for any R family object.
* `update.stream_fit()`: single-step streaming update on a fitted object.
* `predict.stream_fit()`: predictions on new data from the final coefficient
  estimate.
* `conv_path()`: L2 convergence path utility.
