## Online ATE / ATT Estimator
##
## Semiparametrically efficient, doubly-robust one-step estimator for the
## average treatment effect (ATE) or average treatment effect on the treated
## (ATT).  Nuisance models (outcome regression Q and propensity score g) are
## fitted and updated online via Online Super Learner.
##
## Estimands
## ---------
## ATE: psi = E[Y(1) - Y(0)]
## ATT: psi = E[Y(1) - Y(0) | A = 1]
##
## EIF (ATE):
##   D_ATE(O) = H(A,W)*(Y - Q(A,W)) + Q(1,W) - Q(0,W) - psi
##   where H = A/g(W) - (1-A)/(1-g(W))
##
## One-step (ATE):  psi_hat = mean_i[ Q(1,W_i) - Q(0,W_i) + H_i*(Y_i - Q(A_i,W_i)) ]
##
## Numerator contribution (ATT):
##   phi_i = A_i*(Y_i - Q1_i) - g_i*(1-A_i)/(1-g_i)*(Y_i - Q0_i) + A_i*(Q1_i - Q0_i)
##   psi_ATT = sum(phi_i) / sum(A_i)
##
## Standard errors via Welford's running variance:
##   SE_ATE = sqrt(Var(eif_i) / n)
##   SE_ATT = sqrt(Var(phi_i) / n) / mean(A)   [delta method]


## ---------------------------------------------------------------------------
## Internal helpers
## ---------------------------------------------------------------------------

## Clamp propensity scores to the open interval (eps, 1 - eps).
##
## @param g   Numeric vector of propensity score estimates in [0, 1].
## @param eps Scalar clamping threshold; default 0.01.
## @return    Numeric vector of the same length, values in (eps, 1 - eps).
## @keywords  internal
.g_clamp <- function(g, eps = 0.01) pmin(pmax(g, eps), 1 - eps)


## Compute the ATE efficient influence function for one or more observations.
##
## @param Q1  Predicted outcome under A = 1.
## @param Q0  Predicted outcome under A = 0.
## @param Qa  Predicted outcome under observed A.
## @param g_c Clamped propensity score P(A=1|W).
## @param a   Observed treatment (0 or 1).
## @param y   Observed outcome.
## @return    EIF value(s); length equals length(Q1).
## @keywords  internal
.ate_eif <- function(Q1, Q0, Qa, g_c, a, y) {
  H <- a / g_c - (1 - a) / (1 - g_c)
  H * (y - Qa) + Q1 - Q0
}


## Compute the ATT numerator contribution for one or more observations.
##
## The denominator (sum(A)) is tracked separately; this function returns the
## un-normalised phi_i so that psi_ATT = sum(phi_i) / sum(A_i).
##
## @param Q1  Predicted outcome under A = 1.
## @param Q0  Predicted outcome under A = 0.
## @param Qa  Predicted outcome under observed A (not used; kept for symmetry).
## @param g_c Clamped propensity score P(A=1|W).
## @param a   Observed treatment (0 or 1).
## @param y   Observed outcome.
## @return    Unnormalised contribution(s); length equals length(Q1).
## @keywords  internal
.att_eif <- function(Q1, Q0, Qa, g_c, a, y) {
  a * (y - Q1) - g_c * (1 - a) / (1 - g_c) * (y - Q0) + a * (Q1 - Q0)
}


## Compute the standard error from an online_ate object.
##
## @param object An \code{online_ate} object.
## @return       Scalar standard error.
## @keywords     internal
.ate_se <- function(object) {
  se <- sqrt(object$eif_var / object$n_obs)
  if (object$target == "ATT") se <- se / (object$sum_A / object$n_obs)
  se
}


## ---------------------------------------------------------------------------
## online_ate_fit
## ---------------------------------------------------------------------------

#' Fit an Online ATE / ATT Estimator
#'
#' Initialises a doubly-robust, semiparametrically efficient one-step
#' estimator for the **average treatment effect** (ATE) or **average
#' treatment effect on the treated** (ATT).  Both nuisance models
#' (outcome regression \eqn{Q} and propensity score \eqn{g}) are fitted
#' online via [online_sl_fit()].  A streaming update is available via
#' [update.online_ate()].
#'
#' **ATE efficient influence function (EIF):**
#' \deqn{
#'   D_i = H_i(Y_i - Q(A_i, W_i)) + Q(1, W_i) - Q(0, W_i)
#' }
#' where \eqn{H_i = A_i / g(W_i) - (1 - A_i) / (1 - g(W_i))}.
#'
#' **ATT numerator contribution:**
#' \deqn{
#'   \phi_i = A_i(Y_i - Q_1(W_i))
#'            - g(W_i)\frac{1-A_i}{1-g(W_i)}(Y_i - Q_0(W_i))
#'            + A_i(Q_1(W_i) - Q_0(W_i))
#' }
#' with \eqn{\hat\psi_{\rm ATT} = \sum_i \phi_i / \sum_i A_i}.
#'
#' The estimator is doubly robust: consistent if either \eqn{Q} or \eqn{g}
#' is correctly specified.
#'
#' @param W              Numeric matrix \eqn{n \times p}.  Covariate matrix;
#'   the user should supply any intercept column.
#' @param A              Numeric vector of length \eqn{n} with values in
#'   \eqn{\{0, 1\}}.  Binary treatment indicator.
#' @param Y              Numeric vector of length \eqn{n}.  Outcome.
#' @param Q_library      Named list of `sl_learner` objects (from
#'   [make_learner()]) used for the outcome model
#'   \eqn{Q(A, W) = E[Y | A, W]}.  Features supplied to \eqn{Q} are
#'   \eqn{(A, W)}.
#' @param g_library      Named list of `sl_learner` objects used for the
#'   propensity score \eqn{g(W) = P(A = 1 | W)}.  Features are \eqn{W}.
#' @param outcome_family R `family` object for the outcome model.
#'   Default [stats::gaussian()].
#' @param target         `"ATE"` (default) or `"ATT"`.
#' @param k              Integer.  CV folds for the initial batch super
#'   learner.  Default `5`.
#' @param g_clamp        Scalar in \eqn{(0, 0.5)}.  Propensity scores are
#'   clamped to `(g_clamp, 1 - g_clamp)`.  Default `0.01`.
#' @param gamma1         Initial SGD learning rate for ensemble weight
#'   updates.  Default `1`.
#' @param delta          Robbins-Monro decay exponent in \eqn{(0.5, 1]}.
#'   Default `0.6`.
#'
#' @return An object of class `online_ate` with elements:
#'   \describe{
#'     \item{`Q_sl`}{`online_sl` for \eqn{E[Y|A,W]}; features = cbind(A, W).}
#'     \item{`g_sl`}{`online_sl` for \eqn{P(A=1|W)}; binomial family.}
#'     \item{`target`}{`"ATE"` or `"ATT"`.}
#'     \item{`family`}{Outcome family.}
#'     \item{`g_clamp`}{Propensity clamping threshold.}
#'     \item{`psi`}{Current point estimate.}
#'     \item{`eif_mean`}{Welford running mean of EIF contributions.}
#'     \item{`eif_var`}{Running sample variance of EIF contributions.}
#'     \item{`eif_M2`}{Welford M2 accumulator.}
#'     \item{`sum_A`}{Running \eqn{\sum A_i} (used for ATT normalisation).}
#'     \item{`n_batch`}{Number of batch observations.}
#'     \item{`n_obs`}{Total observations (batch + stream).}
#'     \item{`call`}{Matched call.}
#'   }
#'
#' @seealso [update.online_ate()], [online_sl_fit()], [make_learner()]
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 500; p <- 3
#' W <- cbind(1, matrix(rnorm(n * 2), n, 2))
#' A <- rbinom(n, 1, plogis(0.4 * W[, 2]))
#' Y <- 1.0 * A + 0.5 * W[, 2] + rnorm(n, sd = 0.5)
#'
#' lib <- list(
#'   rls = make_learner(
#'     fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
#'     predict = function(model, newdata, ...) predict(model, newdata),
#'     update  = function(model, x, y, family, ...) update(model, x = x, y = y)
#'   )
#' )
#'
#' ate <- online_ate_fit(W, A, Y, Q_library = lib, g_library = lib)
#' print(ate)
#'
#' ## Stream in new observations
#' for (i in seq_len(100)) {
#'   w_i <- c(1, rnorm(2)); a_i <- rbinom(1, 1, plogis(0.4 * w_i[2]))
#'   y_i <- 1.0 * a_i + 0.5 * w_i[2] + rnorm(1, sd = 0.5)
#'   ate <- update(ate, w = w_i, a = a_i, y = y_i)
#' }
#' confint(ate)
#' }
#'
#' @export
online_ate_fit <- function(W, A, Y,
                           Q_library, g_library,
                           outcome_family = stats::gaussian(),
                           target  = c("ATE", "ATT"),
                           k       = 5L,
                           g_clamp = 0.01,
                           gamma1  = 1.0,
                           delta   = 0.6) {
  cl     <- match.call()
  target <- match.arg(target)

  ## --- Input validation ------------------------------------------------------
  if (!is.matrix(W))
    stop("'W' must be a numeric matrix (rows = observations, cols = features)")
  n <- nrow(W)
  if (!is.numeric(A) || length(A) != n)
    stop("'A' must be a numeric vector of length nrow(W)")
  if (!all(A %in% c(0, 1)))
    stop("'A' must be binary: all values must be 0 or 1")
  if (!is.numeric(Y) || length(Y) != n)
    stop("'Y' must be a numeric vector of length nrow(W)")
  if (!is.numeric(g_clamp) || length(g_clamp) != 1L ||
      g_clamp <= 0 || g_clamp >= 0.5)
    stop("'g_clamp' must be a scalar in (0, 0.5)")

  ## --- Fit nuisance models on batch data -------------------------------------
  Q_feat <- cbind(A, W)   # features for outcome model: (A, W)

  Q_sl <- online_sl_fit(Q_feat, Y,
                        family  = outcome_family,
                        library = Q_library,
                        k       = k,
                        gamma1  = gamma1,
                        delta   = delta)

  g_sl <- online_sl_fit(W, A,
                        family  = stats::binomial(),
                        library = g_library,
                        k       = k,
                        gamma1  = gamma1,
                        delta   = delta)

  ## --- Batch EIF values (predict-only; models already fitted above) ----------
  W1 <- cbind(1, W)   # A = 1 counterfactual features
  W0 <- cbind(0, W)   # A = 0 counterfactual features
  WA <- cbind(A, W)   # observed treatment features

  Q1  <- predict(Q_sl, W1)
  Q0  <- predict(Q_sl, W0)
  Qa  <- predict(Q_sl, WA)
  g_c <- .g_clamp(predict(g_sl, W), g_clamp)

  eif_vec <- if (target == "ATE") {
    .ate_eif(Q1, Q0, Qa, g_c, A, Y)
  } else {
    .att_eif(Q1, Q0, Qa, g_c, A, Y)
  }

  ## --- Initialise Welford accumulators from batch EIF vector -----------------
  eif_mean <- mean(eif_vec)
  eif_M2   <- sum((eif_vec - eif_mean)^2)   # (n-1) * sample_var
  eif_var  <- if (n > 1L) eif_M2 / (n - 1L) else 0
  sum_A    <- sum(A)

  ## --- Point estimate --------------------------------------------------------
  psi <- if (target == "ATE") {
    eif_mean
  } else {
    eif_mean * n / sum_A   # sum(phi_i) / sum(A_i)
  }

  structure(
    list(
      Q_sl     = Q_sl,
      g_sl     = g_sl,
      target   = target,
      family   = outcome_family,
      g_clamp  = g_clamp,
      psi      = psi,
      eif_mean = eif_mean,
      eif_var  = eif_var,
      eif_M2   = eif_M2,
      sum_A    = sum_A,
      n_batch  = n,
      n_obs    = n,
      call     = cl
    ),
    class = "online_ate"
  )
}


## ---------------------------------------------------------------------------
## update.online_ate
## ---------------------------------------------------------------------------

#' Update an `online_ate` object with a new observation
#'
#' Follows the **predict-then-update** protocol:
#' 1. Predict counterfactual outcomes and propensity score with the *current*
#'    nuisance models.
#' 2. Compute the EIF contribution for the new observation.
#' 3. Update the Welford running mean and variance.
#' 4. Update the nuisance models (`Q_sl` and `g_sl`) with the new data point.
#'
#' @param object An `online_ate` object from [online_ate_fit()].
#' @param w      Numeric vector of length \eqn{p}.  Covariate vector for the
#'   new observation (same columns as the \eqn{W} matrix passed to
#'   [online_ate_fit()]).
#' @param a      Scalar \eqn{\in \{0, 1\}}.  Treatment for the new
#'   observation.
#' @param y      Scalar.  Outcome for the new observation.
#' @param ...    Ignored.
#'
#' @return An updated `online_ate` object.
#'
#' @seealso [online_ate_fit()]
#'
#' @export
update.online_ate <- function(object, w, a, y, ...) {
  if (!is.numeric(w))
    stop("'w' must be a numeric vector")
  if (length(a) != 1L || !a %in% c(0, 1))
    stop("'a' must be a scalar equal to 0 or 1")

  ## --- Predict with current nuisance models ----------------------------------
  w1 <- matrix(c(1, w), nrow = 1L)    # A = 1 counterfactual
  w0 <- matrix(c(0, w), nrow = 1L)    # A = 0 counterfactual
  wa <- matrix(c(a, w), nrow = 1L)    # observed treatment
  wm <- matrix(w, nrow = 1L)

  Q1  <- predict(object$Q_sl, w1)
  Q0  <- predict(object$Q_sl, w0)
  Qa  <- predict(object$Q_sl, wa)
  g_c <- .g_clamp(predict(object$g_sl, wm), object$g_clamp)

  ## --- EIF contribution for this observation ---------------------------------
  eif <- if (object$target == "ATE") {
    .ate_eif(Q1, Q0, Qa, g_c, a, y)
  } else {
    .att_eif(Q1, Q0, Qa, g_c, a, y)
  }

  ## --- Welford running update ------------------------------------------------
  n_new <- object$n_obs + 1L
  delta           <- eif - object$eif_mean
  object$eif_mean <- object$eif_mean + delta / n_new
  delta2          <- eif - object$eif_mean
  object$eif_M2   <- object$eif_M2 + delta * delta2
  object$eif_var  <- if (n_new > 1L) object$eif_M2 / (n_new - 1L) else 0

  ## --- Update sum_A and point estimate ---------------------------------------
  object$sum_A <- object$sum_A + a

  object$psi <- if (object$target == "ATE") {
    object$eif_mean
  } else {
    object$eif_mean * n_new / object$sum_A   # sum(phi_i) / sum(A_i)
  }

  ## --- Update nuisance models (online, after prediction) --------------------
  object$Q_sl <- update(object$Q_sl, x = c(a, w), y = y)
  object$g_sl <- update(object$g_sl, x = w, y = a)

  object$n_obs <- n_new
  object
}


## ---------------------------------------------------------------------------
## S3 methods
## ---------------------------------------------------------------------------

#' Print an `online_ate` object
#'
#' @param x      An `online_ate` object.
#' @param digits Number of significant digits. Default `4`.
#' @param ...    Ignored.
#' @export
print.online_ate <- function(x, digits = 4L, ...) {
  se <- .ate_se(x)
  ci <- x$psi + c(-1, 1) * stats::qnorm(0.975) * se

  cat("Online", x$target, "Estimator\n")
  cat("Target      :", x$target, "\n")
  cat("Family      :", x$family$family, "/", x$family$link, "\n")
  cat("Observations:", x$n_obs,
      sprintf("(%d batch + %d stream)\n",
              x$n_batch, x$n_obs - x$n_batch))
  cat(sprintf("Estimate    : %.*f\n", digits, x$psi))
  cat(sprintf("Std. Error  : %.*f\n", digits, se))
  cat(sprintf("95%% CI      : [%.*f, %.*f]\n",
              digits, ci[1L], digits, ci[2L]))
  invisible(x)
}


#' Extract the current ATE/ATT estimate from an `online_ate` object
#'
#' @param object An `online_ate` object.
#' @param ...    Ignored.
#' @return A named scalar: the current point estimate.
#' @export
coef.online_ate <- function(object, ...) {
  setNames(object$psi, object$target)
}


#' Confidence interval for an `online_ate` object
#'
#' Computes a Wald-type confidence interval \eqn{\hat\psi \pm z_{\alpha/2} \cdot \widehat{SE}}.
#'
#' @param object An `online_ate` object.
#' @param parm   Ignored (present for S3 compatibility).
#' @param level  Confidence level. Default `0.95`.
#' @param ...    Ignored.
#' @return A \eqn{1 \times 2} matrix with the lower and upper bounds.
#' @export
confint.online_ate <- function(object, parm, level = 0.95, ...) {
  se    <- .ate_se(object)
  z     <- stats::qnorm(1 - (1 - level) / 2)
  probs <- c((1 - level) / 2, 1 - (1 - level) / 2)
  ci    <- object$psi + c(-z, z) * se

  matrix(ci, nrow = 1L,
         dimnames = list(
           object$target,
           paste0(format(100 * probs, trim = TRUE), "%")
         ))
}
