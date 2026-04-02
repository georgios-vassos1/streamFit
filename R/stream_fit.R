#' Check whether a family requires dispersion estimation
#'
#' Returns `TRUE` for families where the dispersion parameter phi is unknown
#' and must be estimated (Gamma, inverse.gaussian, quasi families).
#' Returns `FALSE` for families with known dispersion (binomial, Poisson,
#' negative binomial).
#'
#' @param family An R `family` object.
#' @return Logical scalar.
#' @keywords internal
.needs_dispersion <- function(family) {
  grepl("^(Gamma|inverse\\.gaussian|quasi)", family$family)
}


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
#' @param A_hat Average Fisher information matrix (`p x p`). iSGD only.
#' @param B_hat Average score outer-product matrix (`p x p`). iSGD only.
#' @param rss Residual sum of squares (scalar). RLS only.
#' @param n_obs Number of observations processed (integer).
#' @param pearson_ss Cumulative Pearson sum of squares (scalar). RLS-GLM only,
#'   for families that require dispersion estimation (Gamma, inverse.gaussian).
#'
#' @return An object of class `stream_fit`.
#' @keywords internal
new_stream_fit <- function(beta, beta_path, S = NULL, family = NULL,
                           method, call, hyperparams,
                           A_hat = NULL, B_hat = NULL,
                           rss = NULL, n_obs = NULL,
                           pearson_ss = NULL) {
  structure(
    list(
      beta        = beta,
      beta_path   = beta_path,
      S           = S,
      family      = family,
      method      = method,
      call        = call,
      hyperparams = hyperparams,
      A_hat       = A_hat,
      B_hat       = B_hat,
      rss         = rss,
      n_obs       = n_obs,
      pearson_ss  = pearson_ss
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
  n_obs <- if (!is.null(x$n_obs)) x$n_obs else nrow(x$beta_path)
  cat("Observations:", n_obs, "\n")
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


#' Number of observations in a `stream_fit` object
#'
#' @param object A `stream_fit` object.
#' @param ... Ignored.
#' @return Integer number of observations processed.
#' @export
nobs.stream_fit <- function(object, ...) {
  if (!is.null(object$n_obs)) object$n_obs else nrow(object$beta_path)
}


#' Variance-covariance matrix for a `stream_fit` object
#'
#' Extracts the estimated variance-covariance matrix of the coefficient
#' estimates. For RLS (Gaussian), this is `sigma2_hat * S`. For RLS-GLM, this
#' is the final inverse Fisher information `S`. For iSGD, this is the Toulis &
#' Airoldi (2017) asymptotic sandwich estimator `(1/n) * A_inv %*% B %*% A_inv`.
#'
#' @param object A `stream_fit` object.
#' @param ... Ignored.
#' @return A `p x p` variance-covariance matrix.
#' @export
vcov.stream_fit <- function(object, ...) {
  switch(object$method,
    RLS = {
      if (is.null(object$rss))
        stop("vcov requires RSS; refit with a current version of rls_fit()")
      p <- length(object$beta)
      sigma2 <- object$rss / (object$n_obs - p)
      sigma2 * object$S
    },
    `RLS-GLM` = {
      if (!is.null(object$family) && .needs_dispersion(object$family)) {
        p <- length(object$beta)
        phi_hat <- object$pearson_ss / (object$n_obs - p)
        phi_hat * object$S
      } else {
        object$S
      }
    },
    iSGD = {
      if (is.null(object$A_hat) || is.null(object$B_hat))
        stop("vcov requires sandwich matrices; refit with compute_vcov = TRUE")
      A_inv <- chol2inv(chol(object$A_hat))
      (1.0 / object$n_obs) * A_inv %*% object$B_hat %*% A_inv
    },
    stop("vcov: unknown method '", object$method, "'")
  )
}


#' Summary of a `stream_fit` object
#'
#' Produces a summary with coefficient table including standard errors,
#' test statistics, and p-values.
#'
#' @param object A `stream_fit` object.
#' @param ... Ignored.
#' @return A `summary.stream_fit` object (invisibly).
#' @export
summary.stream_fit <- function(object, ...) {
  V   <- tryCatch(vcov(object), error = function(e) NULL)
  beta <- object$beta
  nms  <- names(beta)
  if (is.null(nms)) nms <- paste0("beta[", seq_along(beta), "]")
  p <- length(beta)
  n_obs <- object$n_obs
  if (is.null(n_obs)) n_obs <- nrow(object$beta_path)

  if (!is.null(V)) {
    se  <- sqrt(pmax(diag(V), 0))
    use_t <- object$method == "RLS" ||
      (object$method == "RLS-GLM" && !is.null(object$family) &&
       .needs_dispersion(object$family))
    if (use_t) {
      stat <- beta / se
      df   <- n_obs - p
      pval <- 2 * stats::pt(-abs(stat), df = df)
      coef_table <- cbind(Estimate = beta, `Std. Error` = se,
                          `t value` = stat, `Pr(>|t|)` = pval)
    } else {
      stat <- beta / se
      pval <- 2 * stats::pnorm(-abs(stat))
      coef_table <- cbind(Estimate = beta, `Std. Error` = se,
                          `z value` = stat, `Pr(>|z|)` = pval)
    }
    rownames(coef_table) <- nms
  } else {
    coef_table <- cbind(Estimate = stats::setNames(beta, nms))
  }

  out <- list(
    coefficients = coef_table,
    vcov         = V,
    method       = object$method,
    family       = object$family,
    n_obs        = n_obs,
    call         = object$call
  )
  class(out) <- "summary.stream_fit"
  out
}


#' Print a `summary.stream_fit` object
#'
#' @param x A `summary.stream_fit` object.
#' @param digits Number of significant digits. Default `4`.
#' @param signif.stars Logical. Show significance stars? Default `TRUE`.
#' @param ... Ignored.
#' @export
print.summary.stream_fit <- function(x, digits = 4L, signif.stars = TRUE, ...) {
  cat("Stream fit\n")
  cat("Method:", x$method, "\n")
  if (!is.null(x$family))
    cat("Family:", x$family$family, "/", x$family$link, "\n")
  cat("Observations:", x$n_obs, "\n\n")
  cat("Coefficients:\n")
  stats::printCoefmat(x$coefficients, digits = digits,
                      signif.stars = signif.stars, na.print = "NA",
                      has.Pvalue = ncol(x$coefficients) >= 4L)
  invisible(x)
}


#' Confidence intervals for `stream_fit` coefficients
#'
#' Computes Wald confidence intervals based on the asymptotic
#' variance-covariance matrix.
#'
#' @param object A `stream_fit` object.
#' @param parm Integer or character vector specifying which parameters to
#'   include. Default: all.
#' @param level Confidence level. Default `0.95`.
#' @param ... Ignored.
#' @return A matrix with columns for the lower and upper bounds.
#' @export
confint.stream_fit <- function(object, parm, level = 0.95, ...) {
  V  <- vcov(object)
  se <- sqrt(pmax(diag(V), 0))
  p  <- length(object$beta)
  n_obs <- object$n_obs
  if (is.null(n_obs)) n_obs <- nrow(object$beta_path)

  use_t <- object$method == "RLS" ||
    (object$method == "RLS-GLM" && !is.null(object$family) &&
     .needs_dispersion(object$family))
  if (use_t) {
    q <- stats::qt((1 + level) / 2, df = n_obs - p)
  } else {
    q <- stats::qnorm((1 + level) / 2)
  }
  ci <- cbind(object$beta - q * se, object$beta + q * se)
  pct <- format(100 * c((1 - level) / 2, (1 + level) / 2), digits = 3)
  colnames(ci) <- paste0(pct, " %")
  nms <- names(object$beta)
  if (is.null(nms)) nms <- paste0("beta[", seq_along(object$beta), "]")
  rownames(ci) <- nms
  if (!missing(parm)) ci <- ci[parm, , drop = FALSE]
  ci
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
#' @return An updated \code{stream_fit} object. Note: \code{beta_path} is
#'   \strong{not} extended (to avoid O(n^2) copy cost); only \code{beta} and
#'   the sufficient statistics (\code{S}, \code{rss}, \code{A_hat}, etc.) are
#'   updated.
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
  switch(object$method,
    RLS = {
      out <- rls_update(x, y, object$beta, object$S, lambda = hp$lambda)
      object$beta <- out$beta
      object$S    <- out$S
      if (!is.null(object$rss)) {
        object$rss   <- object$rss + (y - as.numeric(crossprod(x, out$beta)))^2
        object$n_obs <- object$n_obs + 1L
      }
    },
    `RLS-GLM` = {
      out <- rls_glm_update(x, y, object$beta, object$S, object$family,
                            lambda     = hp$lambda,
                            eta_clip   = hp$eta_clip,
                            score_clip = hp$score_clip)
      object$beta <- out$beta
      object$S    <- out$S
      if (!is.null(object$n_obs)) object$n_obs <- object$n_obs + 1L
      if (!is.null(object$pearson_ss)) {
        eta_new <- pmin(pmax(as.numeric(crossprod(x, out$beta)),
                             -hp$eta_clip), hp$eta_clip)
        mu_new  <- object$family$linkinv(eta_new)
        V_new   <- object$family$variance(mu_new)
        object$pearson_ss <- object$pearson_ss + (y - mu_new)^2 / V_new
      }
    },
    iSGD = {
      ## Step index must be computed BEFORE n_obs is incremented
      t     <- object$n_obs + 1L
      gamma <- hp$gamma1 * t^(-hp$alpha)
      ## Accumulate sandwich matrices before updating beta
      if (!is.null(object$A_hat)) {
        eta_pre <- as.numeric(crossprod(x, object$beta))
        mu      <- object$family$linkinv(eta_pre)
        dmu     <- object$family$mu.eta(eta_pre)
        V_mu    <- object$family$variance(mu)
        w       <- dmu^2 / V_mu
        score_i <- (y - mu) * dmu / V_mu
        n_old   <- object$n_obs
        n_new   <- n_old + 1L
        xx      <- tcrossprod(x)
        object$A_hat <- (n_old * object$A_hat + w * xx) / n_new
        object$B_hat <- (n_old * object$B_hat + score_i^2 * xx) / n_new
        object$n_obs <- n_new
      } else {
        object$n_obs <- object$n_obs + 1L
      }
      object$beta <- isgd_update(x, y, object$beta, gamma, object$family)
    },
    stop("update.stream_fit: unknown method '", object$method, "'")
  )
  object
}
