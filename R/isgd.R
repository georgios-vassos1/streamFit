#' Implicit equation for iSGD (internal)
#'
#' @keywords internal
.implicit_fn <- function(xi, y, eta, gamma, norm_x2, linkinv) {
  xi - gamma * (y - linkinv(eta + xi * norm_x2))
}


#' Single-step implicit SGD update
#'
#' Performs one step of Implicit (proximal) Stochastic Gradient Descent for a
#' GLM. Solves the scalar implicit equation
#' \deqn{\xi_t = \gamma_t\bigl(y_t - g^{-1}(\eta_t + \xi_t\|x_t\|^2)\bigr)}
#' then updates `beta_{t+1} = beta_t + xi_t * x_t`.
#'
#' For Gaussian the equation has a closed form. For all other families a 1-D
#' root is found via Brent's method. The implicit formulation is self-limiting:
#' extreme observations (e.g. large Poisson counts) produce bounded steps
#' without requiring explicit score clipping.
#'
#' @param x Numeric vector of length `p`. Feature vector.
#' @param y Scalar response.
#' @param beta Numeric vector of length `p`. Current coefficient estimate.
#' @param gamma Positive scalar. Learning rate for this step.
#' @param family An R `family` object.
#'
#' @return Updated coefficient vector (length `p`).
#'
#' @references
#' Toulis, P. and Airoldi, E. (2017). Asymptotic and finite-sample properties
#' of estimators based on stochastic gradients. *Annals of Statistics*, 45(4),
#' 1694--1727.
#'
#' @seealso [isgd_fit()], [rls_glm_update()]
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' p <- 3; beta <- numeric(p); fam <- binomial()
#' beta_true <- c(-0.5, 1, -0.8)
#' for (i in seq_len(1000)) {
#'   x    <- c(1, rnorm(p - 1))
#'   y    <- rbinom(1, 1, fam$linkinv(sum(x * beta_true)))
#'   beta <- isgd_update(x, y, beta, gamma = 1.0 * i^(-0.7), fam)
#' }
#' round(beta, 3)
#' }
#'
#' @export
isgd_update <- function(x, y, beta, gamma, family) {
  if (!is.numeric(gamma) || length(gamma) != 1 || gamma <= 0)
    stop("'gamma' must be a positive scalar")
  if (length(x) != length(beta))
    stop("length of 'x' must equal length of 'beta'")
  eta     <- as.numeric(crossprod(x, beta))
  mu      <- family$linkinv(eta)
  r       <- gamma * (y - mu)
  if (abs(r) < .Machine$double.eps) return(beta)
  norm_x2 <- as.numeric(crossprod(x))
  if (family$family == "gaussian") {
    xi <- r / (1.0 + gamma * norm_x2)
  } else {
    interval <- if (r > 0) c(0, r) else c(r, 0)
    xi <- fast_uniroot(
      f        = .implicit_fn,
      interval = interval,
      y = y, eta = eta, gamma = gamma,
      norm_x2 = norm_x2, linkinv = family$linkinv
    )
  }
  beta + xi * x
}


#' Fit iSGD over a dataset
#'
#' Fits a generalised linear model sequentially using Implicit Stochastic
#' Gradient Descent with the learning rate schedule
#' `gamma_t = gamma1 * t^(-alpha)`. Processes each observation in `O(p)` time.
#'
#' **Hyperparameter guidance.**
#' `alpha` must be in `(0.5, 1]` (Robbins-Monro conditions). The default
#' `alpha = 0.7` works well for stationary targets. For tracking time-varying
#' parameters use `alpha` close to `0.5` (slowest valid decay). The natural
#' scale for `gamma1` is the inverse Fisher information per observation:
#' `gamma1 ~ 1 / (w * E[||x||^2])` where `w` is the expected IRLS weight
#' (`~0.25` for balanced logistic, `~mean(y)` for Poisson).
#'
#' @param X Numeric matrix `n x p`. Design matrix.
#' @param y Numeric vector of length `n`. Response vector.
#' @param family An R `family` object. Default [stats::gaussian()].
#' @param gamma1 Positive scalar. Initial learning rate. Default `1`.
#' @param alpha Decay exponent in `(0.5, 1]`. Default `0.7`.
#'
#' @return A `stream_fit` object (without `S`, which iSGD does not maintain).
#'
#' @references
#' Toulis, P. and Airoldi, E. (2017). Asymptotic and finite-sample properties
#' of estimators based on stochastic gradients. *Annals of Statistics*, 45(4),
#' 1694--1727.
#'
#' @seealso [isgd_update()], [rls_glm_fit()]
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' n <- 1000; p <- 4
#' X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
#' beta0 <- c(-0.5, 1, -1, 0.5)
#' y <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta0)))
#' fit <- isgd_fit(X, y, family = binomial(), gamma1 = 1, alpha = 0.7)
#' round(coef(fit), 3)
#' }
#'
#' @export
isgd_fit <- function(X, y, family = stats::gaussian(), gamma1 = 1.0, alpha = 0.7) {
  if (nrow(X) != length(y))
    stop("nrow(X) must equal length(y)")
  if (!is.numeric(gamma1) || length(gamma1) != 1 || gamma1 <= 0)
    stop("'gamma1' must be a positive scalar")
  if (!is.numeric(alpha) || alpha <= 0.5 || alpha > 1)
    stop("'alpha' must be in (0.5, 1]")
  cl     <- match.call()
  n      <- nrow(X); p <- ncol(X)
  beta   <- numeric(p)
  gammas <- gamma1 * seq_len(n)^(-alpha)
  path   <- matrix(NA_real_, nrow = n, ncol = p)
  for (i in seq_len(n)) {
    beta      <- isgd_update(X[i, ], y[i], beta, gammas[i], family)
    path[i, ] <- beta
  }
  new_stream_fit(
    beta        = beta,
    beta_path   = path,
    S           = NULL,
    family      = family,
    method      = "iSGD",
    call        = cl,
    hyperparams = list(gamma1 = gamma1, alpha = alpha)
  )
}
