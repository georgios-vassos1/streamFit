#' Single-step RLS update (linear / Gaussian model)
#'
#' Performs one step of the Recursive Least Squares algorithm for a linear
#' model. The inverse covariance matrix `S` is updated via the
#' Sherman-Morrison rank-1 formula, avoiding a full matrix inversion.
#'
#' Update equations:
#' \deqn{S_{t+1} = \frac{1}{\lambda}\!\left(S_t -
#'   \frac{S_t x_t x_t^\top S_t}{1 + x_t^\top S_t x_t}\right)}
#' \deqn{\beta_{t+1} = \beta_t + S_{t+1} x_t \,(y_t - x_t^\top \beta_t)}
#'
#' This is the special case of [rls_glm_update()] for `family = gaussian()`.
#'
#' @param x Numeric vector of length `p`. Feature vector for the current
#'   observation.
#' @param y Scalar response.
#' @param beta Numeric vector of length `p`. Current coefficient estimate.
#' @param S Numeric matrix `p x p`. Current inverse covariance matrix.
#' @param lambda Forgetting factor in `(0, 1]`. Default `1` (no forgetting).
#'   Effective memory window: `1 / (1 - lambda)`.
#'
#' @return A list with:
#'   \describe{
#'     \item{`beta`}{Updated coefficient vector (length `p`).}
#'     \item{`S`}{Updated inverse covariance matrix (`p x p`).}
#'   }
#'
#' @seealso [rls_fit()] for fitting over a full dataset;
#'   [rls_glm_update()] for the GLM generalisation.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' p <- 3; beta_true <- c(1, -0.5, 0.8)
#' S <- 100 * diag(p); beta <- numeric(p)
#' for (i in seq_len(500)) {
#'   x    <- c(1, rnorm(p - 1))
#'   y    <- sum(x * beta_true) + rnorm(1)
#'   out  <- rls_update(x, y, beta, S)
#'   beta <- out$beta; S <- out$S
#' }
#' round(beta, 3)
#' }
#'
#' @export
rls_update <- function(x, y, beta, S, lambda = 1.0) {
  if (!is.numeric(lambda) || length(lambda) != 1 || lambda <= 0 || lambda > 1)
    stop("'lambda' must be a scalar in (0, 1]")
  if (length(x) != length(beta))
    stop("length of 'x' must equal length of 'beta'")
  if (!identical(dim(S), c(length(x), length(x))))
    stop("'S' must be a square matrix with dimension equal to length(x)")
  Sx    <- as.vector(S %*% x)
  denom <- as.numeric(1.0 + crossprod(x, Sx))
  S_new <- (S - tcrossprod(Sx) / denom) / lambda
  eps   <- as.numeric(y - crossprod(x, beta))
  list(beta = beta + as.vector(S_new %*% x) * eps, S = S_new)
}


#' Fit RLS over a dataset (linear / Gaussian model)
#'
#' Fits a linear model sequentially using Recursive Least Squares, processing
#' each observation in `O(p^2)` time. Equivalent to weighted least squares
#' with exponential forgetting when `lambda < 1`.
#'
#' @param X Numeric matrix `n x p`. Design matrix (include a column of ones
#'   for an intercept).
#' @param y Numeric vector of length `n`. Response vector.
#' @param lambda Forgetting factor in `(0, 1]`. Default `1`.
#' @param S0_scale Positive scalar. Initial inverse covariance is
#'   `S0_scale * diag(p)`. Larger values give a more diffuse prior and faster
#'   early convergence. Default `100`.
#' @param beta_init Optional numeric vector of length `p`. Starting coefficient
#'   vector. Defaults to zero.
#'
#' @return A `stream_fit` object with elements:
#'   \describe{
#'     \item{`beta`}{Final coefficient vector.}
#'     \item{`beta_path`}{Matrix `n x p`: coefficient estimate after each step.}
#'     \item{`S`}{Final inverse covariance matrix.}
#'   }
#'
#' @seealso [rls_update()], [rls_glm_fit()]
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' n <- 500; p <- 4
#' X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
#' beta_true <- c(0.5, 1, -1, 0.5)
#' y <- X %*% beta_true + rnorm(n)
#' fit <- rls_fit(X, y)
#' round(coef(fit), 3)
#' plot(fit)
#' }
#'
#' @export
rls_fit <- function(X, y, lambda = 1.0, S0_scale = 100.0, beta_init = NULL) {
  if (nrow(X) != length(y))
    stop("nrow(X) must equal length(y)")
  if (!is.numeric(lambda) || length(lambda) != 1 || lambda <= 0 || lambda > 1)
    stop("'lambda' must be a scalar in (0, 1]")
  if (!is.numeric(S0_scale) || S0_scale <= 0)
    stop("'S0_scale' must be a positive scalar")
  if (!is.null(beta_init) && length(beta_init) != ncol(X))
    stop("length of 'beta_init' must equal ncol(X)")
  cl   <- match.call()
  n    <- nrow(X); p <- ncol(X)
  beta <- if (is.null(beta_init)) numeric(p) else beta_init
  S    <- S0_scale * diag(p)
  path <- matrix(NA_real_, nrow = n, ncol = p)
  for (i in seq_len(n)) {
    out       <- rls_update(X[i, ], y[i], beta, S, lambda)
    beta      <- out$beta
    S         <- out$S
    path[i, ] <- beta
  }
  new_stream_fit(
    beta        = beta,
    beta_path   = path,
    S           = S,
    family      = stats::gaussian(),
    method      = "RLS",
    call        = cl,
    hyperparams = list(lambda = lambda, S0_scale = S0_scale)
  )
}
