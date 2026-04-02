#' Fast 1-D root finding via Brent's method
#'
#' A thin wrapper around [stats::uniroot()] using Brent's method for 1-D
#' root finding. The function and bracket semantics are identical to
#' `uniroot`.
#'
#' @param f Function whose root is sought. Must be continuous on `interval`.
#' @param interval Numeric vector of length 2: a bracket `[a, b]` such that
#'   `f(a)` and `f(b)` have opposite signs.
#' @param tol Convergence tolerance (absolute). Default `1e-8`.
#' @param maxiter Maximum number of iterations. Default `1000L`.
#' @param ... Additional arguments passed to `f`.
#'
#' @return Scalar root estimate.
#' @keywords internal
fast_uniroot <- function(f, interval, tol = 1e-8, maxiter = 1000L, ...) {
  stats::uniroot(f, interval = interval, tol = tol, maxiter = maxiter, ...)$root
}


#' Coefficient convergence path
#'
#' Computes the Euclidean (L2) distance between each row of a coefficient path
#' matrix and a reference vector. Useful for assessing convergence to the batch
#' MLE or a known true parameter vector.
#'
#' @param beta_path Numeric matrix of size \code{n x p}. Row \code{t} is the
#'   coefficient estimate after processing observation \code{t}.
#' @param ref Numeric vector of length \code{p}. Reference coefficient vector
#'   (e.g. batch MLE or true parameter).
#'
#' @return Numeric vector of length \code{n} giving the L2 distance at each
#'   step.
#'
#' @examples
#' set.seed(1)
#' n <- 200; p <- 3
#' X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
#' beta0 <- c(0.5, -1, 0.8)
#' y <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta0)))
#' fit <- rls_glm_fit(X, y, family = binomial())
#' d <- conv_path(fit$beta_path, beta0)
#' plot(d, type = "l", xlab = "Observation", ylab = "L2 distance to truth")
#'
#' @export
conv_path <- function(beta_path, ref) {
  sqrt(rowSums(sweep(beta_path, 2L, ref, FUN = "-")^2))
}
