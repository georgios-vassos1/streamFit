#' Construct a `stream_fit` object
#'
#' Internal constructor used by [rls_fit()], [rls_glm_fit()], and [isgd_fit()].
#'
#' @param beta Final coefficient vector (length `p`).
#' @param beta_path Coefficient path matrix (`n x p`).
#' @param S Final inverse Fisher information matrix (`p x p`). `NULL` for iSGD.
#' @param family R family object used in fitting. `NULL` for non-GLM methods.
#' @param method Character string: one of `"RLS"`, `"RLS-GLM"`, `"iSGD"`.
#' @param call The matched call captured in the fitting function.
#' @param hyperparams Named list of hyperparameters.
#'
#' @return An object of class `stream_fit`.
#' @keywords internal
new_stream_fit <- function(beta, beta_path, S = NULL, family = NULL,
                           method, call, hyperparams) {
  structure(
    list(
      beta        = beta,
      beta_path   = beta_path,
      S           = S,
      family      = family,
      method      = method,
      call        = call,
      hyperparams = hyperparams
    ),
    class = "stream_fit"
  )
}


#' Print a `stream_fit` object
#'
#' @param x A `stream_fit` object.
#' @param digits Number of significant digits. Default `4`.
#' @param ... Ignored.
#' @export
print.stream_fit <- function(x, digits = 4L, ...) {
  cat("Stream fit\n")
  cat("Method     :", x$method, "\n")
  if (!is.null(x$family))
    cat("Family     :", x$family$family, "/", x$family$link, "\n")
  cat("Observations:", nrow(x$beta_path), "\n")
  cat("Coefficients:\n")
  nms <- names(x$beta)
  if (is.null(nms)) nms <- paste0("beta[", seq_along(x$beta), "]")
  print(stats::setNames(round(x$beta, digits), nms))
  hp <- Filter(Negate(is.null), x$hyperparams)
  hp_str <- if (length(hp) == 0) "none" else
    paste(names(hp), round(unlist(hp), 4), sep = " = ", collapse = ", ")
  cat("Hyperparams:", hp_str, "\n")
  invisible(x)
}


#' Extract coefficients from a `stream_fit` object
#'
#' @param object A `stream_fit` object.
#' @param ... Ignored.
#' @return Named numeric vector of final coefficients.
#' @export
coef.stream_fit <- function(object, ...) {
  nms <- names(object$beta)
  if (is.null(nms)) nms <- paste0("beta[", seq_along(object$beta), "]")
  stats::setNames(object$beta, nms)
}


#' Summary of a `stream_fit` object
#'
#' @param object A `stream_fit` object.
#' @param ... Ignored.
#' @export
summary.stream_fit <- function(object, ...) {
  print(object)
  invisible(object)
}


#' Plot a `stream_fit` object
#'
#' Plots all coefficient trajectories over observations. If `ref` is supplied,
#' the L2 distance to the reference vector is plotted instead.
#'
#' @param x A `stream_fit` object.
#' @param ref Optional numeric vector of length `p`. If supplied, plots the L2
#'   distance to `ref` (e.g. batch MLE or true parameter) rather than the raw
#'   coefficient paths.
#' @param coef_index Integer vector of coefficient indices to plot when plotting
#'   raw paths. Default: all coefficients.
#' @param ... Additional arguments passed to [graphics::plot()] or
#'   [graphics::matplot()].
#' @export
plot.stream_fit <- function(x, ref = NULL, coef_index = NULL, ...) {
  if (!is.null(ref)) {
    d <- conv_path(x$beta_path, ref)
    graphics::plot(d, type = "l",
                   xlab = "Observation",
                   ylab = "L2 distance to reference",
                   main = paste(x$method, ": convergence path"),
                   ...)
  } else {
    idx <- if (is.null(coef_index)) seq_len(ncol(x$beta_path)) else coef_index
    graphics::matplot(x$beta_path[, idx, drop = FALSE],
                      type = "l", lty = 1L,
                      xlab = "Observation",
                      ylab = "Coefficient value",
                      main = paste(x$method, ": coefficient paths"),
                      ...)
  }
  invisible(x)
}


#' Predict method for a \code{stream_fit} object
#'
#' Computes predicted values (on the response scale by default) for new data
#' using the final coefficient estimate.
#'
#' @param object A \code{stream_fit} object.
#' @param newdata Numeric matrix of new covariates (n_new x p). Must have the
#'   same number of columns as the training design matrix.
#' @param type Character string: \code{"response"} (default) returns
#'   \code{g^{-1}(X \%*\% beta)}; \code{"link"} returns \code{X \%*\% beta}.
#' @param ... Ignored.
#' @return Numeric vector of length \code{nrow(newdata)}.
#' @export
predict.stream_fit <- function(object, newdata, type = c("response", "link"), ...) {
  type <- match.arg(type)
  if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
  if (ncol(newdata) != length(object$beta))
    stop("ncol(newdata) must equal the number of fitted coefficients")
  eta <- as.vector(newdata %*% object$beta)
  if (type == "link") return(eta)
  if (!is.null(object$family)) object$family$linkinv(eta) else eta
}


#' Update a \code{stream_fit} object with a new observation
#'
#' Processes a single new observation and returns an updated \code{stream_fit}
#' object. This is the primary online/streaming interface: fit an initial model
#' on a batch, then call \code{update()} for each new observation as it arrives.
#'
#' @param object A \code{stream_fit} object produced by \code{rls_fit()},
#'   \code{rls_glm_fit()}, or \code{isgd_fit()}.
#' @param x Numeric vector of length \code{p}. Feature vector for the new
#'   observation.
#' @param y Scalar response.
#' @param ... Ignored.
#' @return An updated \code{stream_fit} object. The \code{beta_path} is
#'   extended by one row.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 200; p <- 3
#' X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
#' y <- rbinom(n, 1, 1 / (1 + exp(-X %*% c(-0.5, 1, -0.8))))
#'
#' ## Fit on first 100 observations
#' fit <- rls_glm_fit(X[1:100, ], y[1:100], family = binomial())
#'
#' ## Stream in the remaining observations one at a time
#' for (i in 101:n) {
#'   fit <- update(fit, x = X[i, ], y = y[i])
#' }
#' round(coef(fit), 3)
#' }
#' @export
update.stream_fit <- function(object, x, y, ...) {
  if (length(x) != length(object$beta))
    stop("length of 'x' must equal the number of fitted coefficients")
  hp <- object$hyperparams
  if (object$method == "RLS") {
    out <- rls_update(x, y, object$beta, object$S,
                      lambda = hp$lambda)
    new_beta <- out$beta
    new_S    <- out$S
  } else if (object$method == "RLS-GLM") {
    out <- rls_glm_update(x, y, object$beta, object$S, object$family,
                          lambda     = hp$lambda,
                          eta_clip   = hp$eta_clip,
                          score_clip = hp$score_clip)
    new_beta <- out$beta
    new_S    <- out$S
  } else if (object$method == "iSGD") {
    t      <- nrow(object$beta_path) + 1L
    gamma  <- hp$gamma1 * t^(-hp$alpha)
    new_beta <- isgd_update(x, y, object$beta, gamma, object$family)
    new_S    <- NULL
  } else {
    stop("update.stream_fit: unknown method '", object$method, "'")
  }
  object$beta      <- new_beta
  object$beta_path <- rbind(object$beta_path, new_beta)
  object$S         <- new_S
  object
}
