## Numerically stable softmax
##
## @param theta Numeric vector.
## @return Numeric vector of the same length, summing to 1.
## @keywords internal
.softmax <- function(theta) {
  theta <- theta - max(theta)
  e     <- exp(theta)
  e / sum(e)
}


#' Fit an Online Super Learner
#'
#' Initialises an online ensemble by running a batch [super_learner_fit()] on
#' the supplied data, then returns an `online_sl` object whose ensemble weights
#' can be updated one observation at a time via [update.online_sl()].
#'
#' **Weight update rule.**  At each step \eqn{t} the ensemble weights are
#' maintained as \eqn{\alpha_t = \text{softmax}(\theta_t)}.  Given a new
#' observation \eqn{(x_t, y_t)} and base-learner predictions
#' \eqn{z_t = (z_{t1},\dots,z_{tL})^\top}, the gradient of the deviance loss
#' with respect to \eqn{\theta} is
#' \deqn{
#'   \nabla_\theta L_t = -\frac{2(y_t - \mu_t)}{V(\mu_t)}\;
#'   \alpha_t \odot (z_t - \mu_{\text{raw},t})
#' }
#' where \eqn{\mu_{\text{raw},t} = z_t^\top \alpha_t} and
#' \eqn{\mu_t = \text{clamp}(\mu_{\text{raw},t})}.  The SGD update is
#' \deqn{\theta_{t+1} = \theta_t - \gamma_t\,\nabla_\theta L_t}
#' with the Robbins--Monro schedule \eqn{\gamma_t = \gamma_1\,t^{-\delta}}.
#'
#' **Base learner updates.** If a learner was created with a non-`NULL`
#' `update` argument in [make_learner()], that function is called after each
#' weight update to keep the base learner current.  Learners without an
#' `update` function are held fixed at their batch estimates.
#'
#' @param X Numeric matrix `n x p`. Initial training data design matrix.
#' @param y Numeric vector of length `n`. Initial training response.
#' @param family An R `family` object. Default [stats::gaussian()].
#' @param library A named list of `sl_learner` objects created with
#'   [make_learner()].
#' @param k Integer. Number of CV folds for the initial batch SL. Default `5`.
#' @param gamma1 Positive scalar. Initial learning rate for the weight SGD
#'   schedule \eqn{\gamma_t = \gamma_1 t^{-\delta}}. Default `1`.
#' @param delta Decay exponent in `(0.5, 1]`. Controls how quickly the
#'   learning rate decays. Default `0.6`.
#'
#' @return An object of class `online_sl` with elements:
#'   \describe{
#'     \item{`theta`}{Current softmax parameter vector (length `L`).}
#'     \item{`weights`}{Current ensemble weights `softmax(theta)`.}
#'     \item{`learners`}{List of base-learner models (updated online if
#'       their `update` function is provided).}
#'     \item{`library`}{The learner specifications.}
#'     \item{`family`}{The family object.}
#'     \item{`init_sl`}{The initial batch `super_learner_fit` object.}
#'     \item{`hyperparams`}{List of `gamma1` and `delta`.}
#'     \item{`n_obs`}{Total number of observations processed (batch + online).}
#'     \item{`call`}{The matched call.}
#'   }
#'
#' @seealso [update.online_sl()], [make_learner()], [super_learner_fit()]
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 400; p <- 4
#' X     <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
#' beta0 <- c(-0.5, 1, -1, 0.5)
#' y     <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta0)))
#'
#' lib <- list(
#'   rls_glm = make_learner(
#'     fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
#'     predict = function(model, newdata, ...) predict(model, newdata),
#'     update  = function(model, x, y, family, ...) update(model, x = x, y = y)
#'   ),
#'   isgd = make_learner(
#'     fit     = function(X, y, family, ...) isgd_fit(X, y, family = family),
#'     predict = function(model, newdata, ...) predict(model, newdata),
#'     update  = function(model, x, y, family, ...) update(model, x = x, y = y)
#'   )
#' )
#'
#' ## Fit on first 300 observations
#' osl <- online_sl_fit(X[1:300, ], y[1:300], family = binomial(), library = lib)
#'
#' ## Stream in the remaining 100 observations
#' for (i in 301:400) {
#'   osl <- update(osl, x = X[i, ], y = y[i])
#' }
#' print(osl)
#' }
#'
#' @export
online_sl_fit <- function(X, y, family = stats::gaussian(),
                           library, k = 5L,
                           gamma1 = 1.0, delta = 0.6) {
  cl <- match.call()

  if (!is.matrix(X))
    stop("'X' must be a numeric matrix")
  if (!is.numeric(y) || length(y) != nrow(X))
    stop("'y' must be a numeric vector of length nrow(X)")
  if (!is.list(library) || length(library) == 0L)
    stop("'library' must be a non-empty list of sl_learner objects")
  for (i in seq_along(library)) {
    if (!inherits(library[[i]], "sl_learner"))
      stop("every element of 'library' must be created with make_learner()")
  }
  if (!is.numeric(gamma1) || length(gamma1) != 1L || gamma1 <= 0)
    stop("'gamma1' must be a positive scalar")
  if (!is.numeric(delta) || length(delta) != 1L || delta <= 0.5 || delta > 1)
    stop("'delta' must be in (0.5, 1]")

  ## --- Batch initialisation via super_learner_fit ----------------------------
  init_sl <- super_learner_fit(X, y, family = family, library = library, k = k)

  ## Initialise theta from batch weights: softmax(log(w)) = w
  w0    <- pmax(init_sl$weights, .Machine$double.eps)
  theta <- log(w0)

  structure(
    list(
      theta       = theta,
      weights     = .softmax(theta),
      learners    = init_sl$learners,
      library     = init_sl$library,
      family      = family,
      init_sl     = init_sl,
      hyperparams = list(gamma1 = gamma1, delta = delta),
      n_obs       = nrow(X),
      call        = cl
    ),
    class = "online_sl"
  )
}


#' Update an `online_sl` object with a new observation
#'
#' Performs one online step: computes base-learner predictions, updates the
#' ensemble weights via SGD, and (if supported) updates the base learners.
#'
#' @param object An `online_sl` object produced by [online_sl_fit()].
#' @param x Numeric vector of length `p`. Feature vector for the new
#'   observation.
#' @param y Scalar response.
#' @param ... Ignored.
#'
#' @return An updated `online_sl` object.
#'
#' @seealso [online_sl_fit()]
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 400; p <- 4
#' X     <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
#' beta0 <- c(-0.5, 1, -1, 0.5)
#' y     <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta0)))
#'
#' lib <- list(
#'   rls_glm = make_learner(
#'     fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
#'     predict = function(model, newdata, ...) predict(model, newdata),
#'     update  = function(model, x, y, family, ...) update(model, x = x, y = y)
#'   )
#' )
#' osl <- online_sl_fit(X[1:300, ], y[1:300], family = binomial(), library = lib)
#' osl <- update(osl, x = X[301, ], y = y[301])
#' }
#'
#' @export
update.online_sl <- function(object, x, y, ...) {
  if (!is.numeric(x))
    stop("'x' must be a numeric vector")

  x1   <- matrix(x, nrow = 1L)
  fam  <- object$family
  hp   <- object$hyperparams
  L    <- length(object$learners)

  ## --- Base-learner predictions at x ----------------------------------------
  z <- vapply(seq_len(L), function(j)
    object$library[[j]]$predict(object$learners[[j]], x1),
    numeric(1L))

  ## --- Current ensemble prediction -------------------------------------------
  alpha   <- object$weights
  mu_raw  <- as.numeric(crossprod(z, alpha))
  mu      <- .sl_clamp(mu_raw, fam)

  ## --- Gradient of deviance loss w.r.t. theta --------------------------------
  ## ∂L/∂θ_j = -2(y - μ)/V(μ) · α_j · (z_j - μ_raw)
  V_mu  <- fam$variance(mu)
  grad  <- -2 * (y - mu) / V_mu * alpha * (z - mu_raw)

  ## --- SGD weight update -----------------------------------------------------
  t            <- object$n_obs + 1L
  gamma        <- hp$gamma1 * t^(-hp$delta)
  object$theta   <- object$theta - gamma * grad
  object$weights <- .softmax(object$theta)

  ## --- Optional base-learner updates -----------------------------------------
  for (j in seq_len(L)) {
    upd <- object$library[[j]]$update
    if (!is.null(upd))
      object$learners[[j]] <- upd(object$learners[[j]], x, y, fam)
  }

  object$n_obs <- t
  object
}

## Utility: NULL-coalescing operator (avoids importing rlang)
`%||%` <- function(a, b) if (!is.null(a)) a else b


#' Predict method for an `online_sl` object
#'
#' Computes ensemble predictions using the current weights and base-learner
#' models.
#'
#' @param object An `online_sl` object.
#' @param newdata Numeric matrix of new covariates (`n_new x p`).
#' @param type `"response"` (default) or `"link"`.
#' @param ... Ignored.
#' @return Numeric vector of length `nrow(newdata)`.
#' @export
predict.online_sl <- function(object, newdata,
                               type = c("response", "link"), ...) {
  type <- match.arg(type)
  if (!is.matrix(newdata)) newdata <- as.matrix(newdata)

  L     <- length(object$learners)
  preds <- matrix(NA_real_, nrow = nrow(newdata), ncol = L)
  for (j in seq_len(L))
    preds[, j] <- object$library[[j]]$predict(object$learners[[j]], newdata)

  mu_hat <- as.vector(preds %*% object$weights)
  if (type == "link") object$family$linkfun(mu_hat) else mu_hat
}


#' Print an `online_sl` object
#'
#' @param x An `online_sl` object.
#' @param digits Number of significant digits. Default `4`.
#' @param ... Ignored.
#' @export
print.online_sl <- function(x, digits = 4L, ...) {
  cat("Online Super Learner\n")
  cat("Family      :", x$family$family, "/", x$family$link, "\n")
  cat("Observations:", x$n_obs,
      sprintf("(%d batch + %d online)\n",
              x$init_sl$n_obs, x$n_obs - x$init_sl$n_obs))
  cat("CV folds    :", x$init_sl$k, "\n\n")
  cat("Current ensemble weights:\n")
  print(round(x$weights, digits))
  cat("\nInitial batch weights:\n")
  print(round(x$init_sl$weights, digits))
  invisible(x)
}


#' Extract current ensemble weights from an `online_sl` object
#'
#' @param object An `online_sl` object.
#' @param ... Ignored.
#' @return Named numeric vector of current ensemble weights.
#' @export
coef.online_sl <- function(object, ...) object$weights
