## Online TMLE Estimator
##
## Targeted Maximum Likelihood (TMLE) version of the online ATE / ATT
## estimator.  Extends online_ate_fit() with a one-dimensional targeting
## step that fluctuates the initial outcome regression Q along a submodel
## indexed by epsilon, solving the EIF score equation
## mean(H * (Y - Q*(A,W))) ≈ 0.
##
## The targeting step achieves second-order bias reduction (remainder O_p(n^-1)
## vs O_p(n^{-1/2})) and asymptotic efficiency under both correctly and
## misspecified nuisance models.
##
## Estimands and targeting submodel
## ----------------------------------
## ATE clever covariate:
##   H(A,W)  = A/g(W) − (1−A)/(1−g(W))
##   H1(W)   = 1/g(W)
##   H0(W)   = −1/(1−g(W))
##
## ATT clever covariate (normalised by running mean(A)):
##   H_att(A,W) = (A − g(W)*(1−A)/(1−g(W))) / mean(A)
##   H1_att(W)  = 1 / mean(A)
##   H0_att(W)  = −g(W) / (mean(A)*(1−g(W)))
##
## Fluctuation submodel:
##   Q*(A,W) = linkinv( linkfun(Q(A,W)) + ε * H(A,W) )
##
## TMLE EIF reuses .ate_eif / .att_eif with starred Q.
##
## Key helpers reused from online_ate.R: .g_clamp, .ate_eif, .att_eif.


## ---------------------------------------------------------------------------
## Internal helpers
## ---------------------------------------------------------------------------

## Compute the clever covariate H for one or more observations.
##
## @param a      Observed treatment scalar or vector (0/1).
## @param g_c    Clamped propensity score P(A=1|W).
## @param target "ATE" or "ATT".
## @param mean_A Running or batch mean of A (ATT normalisation).
## @return       Numeric scalar or vector of H values.
## @keywords     internal
.tmle_h <- function(a, g_c, target, mean_A) {
  if (target == "ATE") {
    a / g_c - (1 - a) / (1 - g_c)
  } else {
    (a - g_c * (1 - a) / (1 - g_c)) / mean_A
  }
}


## Compute counterfactual H values at A=1 (H1) and A=0 (H0).
##
## @param g_c    Clamped propensity score.
## @param target "ATE" or "ATT".
## @param mean_A Running or batch mean of A (ATT normalisation).
## @return       Named list with elements \code{H1} and \code{H0}.
## @keywords     internal
.tmle_h_cf <- function(g_c, target, mean_A) {
  if (target == "ATE") {
    list(H1 =  1 / g_c,
         H0 = -1 / (1 - g_c))
  } else {
    list(H1 =  1 / mean_A,
         H0 = -g_c / (mean_A * (1 - g_c)))
  }
}


## Apply the TMLE fluctuation submodel: Q*(A,W) = linkinv(linkfun(Q) + eps*H).
##
## @param Q   Numeric vector of outcome predictions (response scale).
## @param H   Numeric vector of clever covariate values (same length as Q).
## @param eps Scalar epsilon (targeting parameter).
## @param fam R family object.
## @return    Numeric vector of targeted predictions (response scale).
## @keywords  internal
.tmle_q_star <- function(Q, H, eps, fam) {
  fam$linkinv(fam$linkfun(Q) + eps * H)
}


## Compute the standard error from an online_tmle object.
##
## @param object An \code{online_tmle} object.
## @return       Scalar standard error.
## @keywords     internal
.tmle_se <- function(object) {
  se <- sqrt(object$eif_var / object$n_obs)
  if (object$target == "ATT") se <- se / (object$sum_A / object$n_obs)
  se
}


## ---------------------------------------------------------------------------
## online_tmle_fit
## ---------------------------------------------------------------------------

#' Fit an Online TMLE ATE / ATT Estimator
#'
#' Initialises a Targeted Maximum Likelihood (TMLE) estimator for the
#' **average treatment effect** (ATE) or **average treatment effect on the
#' treated** (ATT).  Both nuisance models (outcome regression \eqn{Q} and
#' propensity score \eqn{g}) are fitted online via [online_sl_fit()].  An
#' additional one-dimensional targeting step fluctuates \eqn{Q} along the
#' submodel
#' \deqn{Q^*(A,W) = g^{-1}\!\bigl(g(Q(A,W)) + \varepsilon H(A,W)\bigr)}
#' by fitting \eqn{\varepsilon} via [rls_glm_fit()].  A streaming update
#' is available via [update.online_tmle()].
#'
#' **Targeting step** solves the EIF score equation
#' \eqn{\sum_i H_i(Y_i - Q^*(A_i,W_i)) \approx 0}, which achieves
#' second-order bias reduction and asymptotic efficiency.
#'
#' **ATE clever covariate:**
#' \deqn{H(A,W) = A/g(W) - (1-A)/(1-g(W))}
#'
#' **ATT clever covariate** (normalised by \eqn{\bar A}):
#' \deqn{H_{\rm ATT}(A,W) = \bigl(A - g(W)(1-A)/(1-g(W))\bigr) / \bar A}
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
#' @param max_iter       Maximum number of targeting iterations.  Each
#'   iteration refits \eqn{\varepsilon} on the current \eqn{Q^*} and updates
#'   \eqn{Q^*}.  Default `10`.
#' @param tol            Convergence threshold for the score equation
#'   \eqn{|\bar H(Y - Q^*_a)| < \texttt{tol}}.  Default `1e-6`.
#'
#' @return An object of class `online_tmle` with elements:
#'   \describe{
#'     \item{`Q_sl`}{`online_sl` for \eqn{E[Y|A,W]}; features = cbind(A, W).}
#'     \item{`g_sl`}{`online_sl` for \eqn{P(A=1|W)}; binomial family.}
#'     \item{`epsilon_fit`}{1-D `stream_fit` (RLS-GLM) seeded with the
#'       final batch \eqn{\varepsilon}; used for incremental streaming updates.}
#'     \item{`epsilon`}{Cumulative \eqn{\varepsilon} across all targeting
#'       iterations: `sum(eps_k)`.  Equals `epsilon_fit$beta[1]` for Gaussian.}
#'     \item{`n_iter`}{Number of targeting iterations actually run.}
#'     \item{`iter_converged`}{Logical: did the score equation reach `tol`?}
#'     \item{`target`}{`"ATE"` or `"ATT"`.}
#'     \item{`family`}{Outcome family.}
#'     \item{`g_clamp`}{Propensity clamping threshold.}
#'     \item{`psi`}{Current TMLE point estimate.}
#'     \item{`eif_mean`}{Welford running mean of TMLE EIF contributions.}
#'     \item{`eif_var`}{Running sample variance of TMLE EIF contributions.}
#'     \item{`eif_M2`}{Welford M2 accumulator.}
#'     \item{`sum_A`}{Running \eqn{\sum A_i} (used for ATT normalisation).}
#'     \item{`n_batch`}{Number of batch observations.}
#'     \item{`n_obs`}{Total observations (batch + stream).}
#'     \item{`call`}{Matched call.}
#'   }
#'
#' @seealso [update.online_tmle()], [online_ate_fit()], [online_sl_fit()],
#'   [make_learner()]
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
#' tmle <- online_tmle_fit(W, A, Y, Q_library = lib, g_library = lib)
#' print(tmle)
#'
#' ## Stream in new observations
#' for (i in seq_len(100)) {
#'   w_i <- c(1, rnorm(2)); a_i <- rbinom(1, 1, plogis(0.4 * w_i[2]))
#'   y_i <- 1.0 * a_i + 0.5 * w_i[2] + rnorm(1, sd = 0.5)
#'   tmle <- update(tmle, w = w_i, a = a_i, y = y_i)
#' }
#' confint(tmle)
#' }
#'
#' @export
online_tmle_fit <- function(W, A, Y,
                            Q_library, g_library,
                            outcome_family = stats::gaussian(),
                            target   = c("ATE", "ATT"),
                            k        = 5L,
                            g_clamp  = 0.01,
                            gamma1   = 1.0,
                            delta    = 0.6,
                            max_iter = 10L,
                            tol      = 1e-6) {
  cl     <- match.call()
  target <- match.arg(target)

  ## --- Input validation (mirrors online_ate_fit) ----------------------------
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

  ## --- Fit nuisance models on batch data ------------------------------------
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

  ## --- Predict nuisance quantities on batch data ----------------------------
  W1 <- cbind(1, W)   # A = 1 counterfactual features
  W0 <- cbind(0, W)   # A = 0 counterfactual features
  WA <- cbind(A, W)   # observed treatment features

  Q1  <- predict(Q_sl, W1)
  Q0  <- predict(Q_sl, W0)
  Qa  <- predict(Q_sl, WA)
  g_c <- .g_clamp(predict(g_sl, W), g_clamp)

  ## --- Iterative targeting: fluctuate Q along 1-D submodel until EIF ≈ 0 ---
  ##
  ## For any link function: linkfun(Q*_k) = linkfun(Q*_{k-1}) + eps_k * H
  ## so after K iterations:  linkfun(Q*_K) = linkfun(Q) + epsilon_total * H
  ## where epsilon_total = sum_k eps_k.  This means Q* depends only on the
  ## unadapted Q and epsilon_total, which keeps the streaming update simple.
  mean_A  <- mean(A)
  H_batch <- .tmle_h(A, g_c, target, mean_A)
  Hcf     <- .tmle_h_cf(g_c, target, mean_A)

  ## Running Q* starts at the unadapted Q; linkfun(Q*) accumulates eps_k * H.
  Qa_star <- Qa
  Q1_star <- Q1
  Q0_star <- Q0
  epsilon_total  <- 0
  iter_converged <- FALSE

  for (iter in seq_len(max_iter)) {
    eps_fit_iter <- rls_glm_fit(
      X      = matrix(H_batch, ncol = 1L),
      y      = Y,
      family = outcome_family,
      offset = outcome_family$linkfun(Qa_star)   # offset = linkfun of current Q*
    )
    eps_k         <- eps_fit_iter$beta[1L]
    epsilon_total <- epsilon_total + eps_k

    Qa_star <- .tmle_q_star(Qa_star, H_batch, eps_k, outcome_family)
    Q1_star <- .tmle_q_star(Q1_star, Hcf$H1,  eps_k, outcome_family)
    Q0_star <- .tmle_q_star(Q0_star, Hcf$H0,  eps_k, outcome_family)

    if (abs(mean(H_batch * (Y - Qa_star))) < tol) {
      iter_converged <- TRUE
      break
    }
  }
  n_iter <- iter   # number of iterations actually run

  ## Seed epsilon_fit for streaming updates.
  ## Use the last iteration's inverse-Fisher S (informative about curvature at
  ## the converged point) but override beta to epsilon_total so that
  ## epsilon_fit$beta[1] == epsilon_total exactly.  This avoids a spurious
  ## second RLS-GLM pass that would corrupt epsilon_total for logistic families.
  epsilon_fit        <- eps_fit_iter
  epsilon_fit$beta   <- c(epsilon_total)
  epsilon            <- epsilon_total

  ## --- TMLE EIF vector (uses starred Q) -------------------------------------
  eif_vec <- if (target == "ATE") {
    .ate_eif(Q1_star, Q0_star, Qa_star, g_c, A, Y)
  } else {
    .att_eif(Q1_star, Q0_star, Qa_star, g_c, A, Y)
  }

  ## --- Initialise Welford accumulators from batch TMLE EIF ------------------
  eif_mean <- mean(eif_vec)
  eif_M2   <- sum((eif_vec - eif_mean)^2)   # (n-1) * sample_var
  eif_var  <- if (n > 1L) eif_M2 / (n - 1L) else 0
  sum_A    <- sum(A)

  ## --- Point estimate -------------------------------------------------------
  psi <- if (target == "ATE") {
    eif_mean
  } else {
    eif_mean * n / sum_A   # sum(phi_i) / sum(A_i)
  }

  structure(
    list(
      Q_sl           = Q_sl,
      g_sl           = g_sl,
      epsilon_fit    = epsilon_fit,
      epsilon        = epsilon,
      n_iter         = n_iter,
      iter_converged = iter_converged,
      target         = target,
      family         = outcome_family,
      g_clamp        = g_clamp,
      psi            = psi,
      eif_mean       = eif_mean,
      eif_var        = eif_var,
      eif_M2         = eif_M2,
      sum_A          = sum_A,
      n_batch        = n,
      n_obs          = n,
      call           = cl
    ),
    class = "online_tmle"
  )
}


## ---------------------------------------------------------------------------
## update.online_tmle
## ---------------------------------------------------------------------------

#' Update an `online_tmle` object with a new observation
#'
#' Follows the **predict-then-update** protocol:
#' 1. Compute the current `mean_A` (before updating `sum_A`).
#' 2. Predict counterfactual outcomes and propensity score with the *current*
#'    nuisance models.
#' 3. Compute clever covariates `H`, `H1`, `H0` using the current `mean_A`.
#' 4. Compute targeted predictions `Q*a`, `Q1*`, `Q0*` using the current
#'    `epsilon`.
#' 5. Compute the TMLE EIF contribution; update the Welford running mean and
#'    variance; update `psi`.
#' 6. Update the epsilon model with the new observation.
#' 7. Update `Q_sl` and `g_sl` with the new data point.
#' 8. Increment `sum_A` and `n_obs`.
#'
#' @param object An `online_tmle` object from [online_tmle_fit()].
#' @param w      Numeric vector of length \eqn{p}.  Covariate vector for the
#'   new observation (same columns as the \eqn{W} matrix passed to
#'   [online_tmle_fit()]).
#' @param a      Scalar \eqn{\in \{0, 1\}}.  Treatment for the new
#'   observation.
#' @param y      Scalar.  Outcome for the new observation.
#' @param ...    Ignored.
#'
#' @return An updated `online_tmle` object.
#'
#' @seealso [online_tmle_fit()]
#'
#' @export
update.online_tmle <- function(object, w, a, y, ...) {
  if (!is.numeric(w))
    stop("'w' must be a numeric vector")
  if (length(a) != 1L || !a %in% c(0, 1))
    stop("'a' must be a scalar equal to 0 or 1")

  ## --- Current epsilon and mean_A (before this observation) -----------------
  epsilon <- object$epsilon
  mean_A  <- object$sum_A / object$n_obs   # n_obs >= 1 after batch fit

  ## --- Predict with current nuisance models ---------------------------------
  w1 <- matrix(c(1, w), nrow = 1L)    # A = 1 counterfactual
  w0 <- matrix(c(0, w), nrow = 1L)    # A = 0 counterfactual
  wa <- matrix(c(a, w), nrow = 1L)    # observed treatment
  wm <- matrix(w, nrow = 1L)

  Q1  <- predict(object$Q_sl, w1)
  Q0  <- predict(object$Q_sl, w0)
  Qa  <- predict(object$Q_sl, wa)
  g_c <- .g_clamp(predict(object$g_sl, wm), object$g_clamp)

  ## --- Clever covariates using current mean_A --------------------------------
  H   <- .tmle_h(a, g_c, object$target, mean_A)
  Hcf <- .tmle_h_cf(g_c, object$target, mean_A)

  ## --- Targeted predictions using current epsilon ---------------------------
  Qa_star <- .tmle_q_star(Qa, H,       epsilon, object$family)
  Q1_star <- .tmle_q_star(Q1, Hcf$H1, epsilon, object$family)
  Q0_star <- .tmle_q_star(Q0, Hcf$H0, epsilon, object$family)

  ## --- TMLE EIF for this observation ----------------------------------------
  eif <- if (object$target == "ATE") {
    .ate_eif(Q1_star, Q0_star, Qa_star, g_c, a, y)
  } else {
    .att_eif(Q1_star, Q0_star, Qa_star, g_c, a, y)
  }

  ## --- Welford running update -----------------------------------------------
  n_new           <- object$n_obs + 1L
  delta_eif       <- eif - object$eif_mean
  object$eif_mean <- object$eif_mean + delta_eif / n_new
  delta2_eif      <- eif - object$eif_mean
  object$eif_M2   <- object$eif_M2 + delta_eif * delta2_eif
  object$eif_var  <- if (n_new > 1L) object$eif_M2 / (n_new - 1L) else 0

  ## --- Update epsilon model, then epsilon -----------------------------------
  object$epsilon_fit <- update(object$epsilon_fit,
                               x      = H,
                               y      = y,
                               offset = object$family$linkfun(Qa))
  object$epsilon <- object$epsilon_fit$beta[1L]

  ## --- Update sum_A and point estimate --------------------------------------
  object$sum_A <- object$sum_A + a

  object$psi <- if (object$target == "ATE") {
    object$eif_mean
  } else {
    object$eif_mean * n_new / object$sum_A   # sum(phi_i) / sum(A_i)
  }

  ## --- Update nuisance models (online, after prediction) -------------------
  object$Q_sl <- update(object$Q_sl, x = c(a, w), y = y)
  object$g_sl <- update(object$g_sl, x = w, y = a)

  object$n_obs <- n_new
  object
}


## ---------------------------------------------------------------------------
## S3 methods
## ---------------------------------------------------------------------------

#' Print an `online_tmle` object
#'
#' @param x      An `online_tmle` object.
#' @param digits Number of significant digits. Default `4`.
#' @param ...    Ignored.
#' @export
print.online_tmle <- function(x, digits = 4L, ...) {
  se <- .tmle_se(x)
  ci <- x$psi + c(-1, 1) * stats::qnorm(0.975) * se

  cat("Online TMLE", x$target, "Estimator\n")
  cat("Target      :", x$target, "\n")
  cat("Family      :", x$family$family, "/", x$family$link, "\n")
  cat("Observations:", x$n_obs,
      sprintf("(%d batch + %d stream)\n",
              x$n_batch, x$n_obs - x$n_batch))
  cat(sprintf("epsilon     : %.*f  (%d iter%s)\n",
              digits, x$epsilon, x$n_iter,
              if (isTRUE(x$iter_converged)) "" else ", not converged"))
  cat(sprintf("Estimate    : %.*f\n", digits, x$psi))
  cat(sprintf("Std. Error  : %.*f\n", digits, se))
  cat(sprintf("95%% CI      : [%.*f, %.*f]\n",
              digits, ci[1L], digits, ci[2L]))
  invisible(x)
}


#' Extract the current ATE/ATT TMLE estimate from an `online_tmle` object
#'
#' @param object An `online_tmle` object.
#' @param ...    Ignored.
#' @return A named scalar: the current point estimate.
#' @export
coef.online_tmle <- function(object, ...) {
  stats::setNames(object$psi, object$target)
}


#' Confidence interval for an `online_tmle` object
#'
#' Computes a Wald-type confidence interval
#' \eqn{\hat\psi \pm z_{\alpha/2} \cdot \widehat{SE}}.
#'
#' @param object An `online_tmle` object.
#' @param parm   Ignored (present for S3 compatibility).
#' @param level  Confidence level. Default `0.95`.
#' @param ...    Ignored.
#' @return A \eqn{1 \times 2} matrix with the lower and upper bounds.
#' @export
confint.online_tmle <- function(object, parm, level = 0.95, ...) {
  se    <- .tmle_se(object)
  z     <- stats::qnorm(1 - (1 - level) / 2)
  probs <- c((1 - level) / 2, 1 - (1 - level) / 2)
  ci    <- object$psi + c(-z, z) * se

  matrix(ci, nrow = 1L,
         dimnames = list(
           object$target,
           paste0(format(100 * probs, trim = TRUE), "%")
         ))
}
