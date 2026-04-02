#' Single-step RLS-GLM update
#'
#' Performs one step of the online natural-gradient / IRLS update for a
#' generalised linear model. The inverse Fisher information `S` is updated via
#' the Sherman-Morrison rank-1 formula; the coefficient is updated via a
#' natural gradient step.
#'
#' Update equations (w is the IRLS weight, s the scalar score):
#' \deqn{w_t = \frac{(\partial\mu/\partial\eta)^2}{V(\mu_t)}}
#' \deqn{S_{t+1} = \frac{1}{\lambda}\!\left(S_t -
#'   \frac{w_t S_t x_t x_t^\top S_t}{1 + w_t x_t^\top S_t x_t}\right)}
#' \deqn{s_t = (y_t - \mu_t)\,\frac{\partial\mu/\partial\eta}{V(\mu_t)}, \quad
#'   \beta_{t+1} = \beta_t + S_{t+1} x_t s_t}
#'
#' Reduces to [rls_update()] for `family = gaussian()`.
#'
#' @param x Numeric vector of length `p`. Feature vector.
#' @param y Scalar response.
#' @param beta Numeric vector of length `p`. Current coefficient estimate.
#' @param S Numeric matrix `p x p`. Current inverse Fisher information.
#' @param family An R `family` object, e.g. [stats::binomial()],
#'   [stats::poisson()], [stats::gaussian()].
#' @param lambda Forgetting factor in `(0, 1]`. Default `1`.
#' @param eta_clip Positive scalar. Linear predictor is clamped to
#'   `[-eta_clip, eta_clip]` to prevent Fisher weight collapse. Default `10`.
#' @param score_clip Optional positive scalar. Clamps the score contribution
#'   `s_t` to `[-score_clip, score_clip]`. Recommended for unbounded response
#'   families (Poisson, Gamma). Natural scale: `k * sqrt(V(mu))`, `k = 5`.
#' @param offset Numeric scalar offset added to the linear predictor.
#'   Typical use: `log(exposure)` in Poisson rate models. Default `0`.
#' @param S_max Optional positive scalar. Upper bound for the diagonal
#'   elements of `S`. After each update, any diagonal entry exceeding
#'   `S_max` is rescaled (along with its row and column) back to `S_max`.
#'   This prevents covariance windup when `lambda < 1` and the design
#'   matrix has sparse indicator columns (e.g. interval dummies in
#'   piecewise-exponential models). Default `NULL` (no clamping).
#'
#' @return A list with:
#'   \describe{
#'     \item{`beta`}{Updated coefficient vector.}
#'     \item{`S`}{Updated inverse Fisher information matrix.}
#'   }
#'
#' @seealso [rls_glm_fit()], [isgd_update()]
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' p <- 3; beta_true <- c(-0.5, 1, -0.8)
#' S <- 10 * diag(p); beta <- numeric(p); fam <- binomial()
#' for (i in seq_len(1000)) {
#'   x    <- c(1, rnorm(p - 1))
#'   y    <- rbinom(1, 1, fam$linkinv(sum(x * beta_true)))
#'   out  <- rls_glm_update(x, y, beta, S, fam)
#'   beta <- out$beta; S <- out$S
#' }
#' round(beta, 3)
#' }
#'
#' @export
rls_glm_update <- function(x, y, beta, S, family,
                            lambda = 1.0, eta_clip = 10.0,
                            score_clip = NULL, offset = 0,
                            S_max = NULL) {
  if (!is.numeric(lambda) || length(lambda) != 1 || lambda <= 0 || lambda > 1)
    stop("'lambda' must be a scalar in (0, 1]")
  if (length(x) != length(beta))
    stop("length of 'x' must equal length of 'beta'")
  if (!identical(dim(S), c(length(x), length(x))))
    stop("'S' must be a square matrix with dimension equal to length(x)")
  if (!is.numeric(eta_clip) || length(eta_clip) != 1 || eta_clip <= 0)
    stop("'eta_clip' must be a positive scalar")
  if (!is.null(score_clip) &&
      (!is.numeric(score_clip) || length(score_clip) != 1 || score_clip <= 0))
    stop("'score_clip' must be a positive scalar")
  if (!is.null(S_max) &&
      (!is.numeric(S_max) || length(S_max) != 1 || S_max <= 0))
    stop("'S_max' must be a positive scalar")
  eta  <- pmin(pmax(as.numeric(crossprod(x, beta)) + offset, -eta_clip), eta_clip)
  mu   <- family$linkinv(eta)
  dmu  <- family$mu.eta(eta)
  vmu  <- family$variance(mu)
  w    <- dmu^2 / vmu

  Sx    <- as.vector(S %*% x)
  denom <- as.numeric(1.0 + w * crossprod(x, Sx))
  S_new <- (S - w * tcrossprod(Sx) / denom) / lambda
  gain  <- Sx / (lambda * denom)           # Kalman gain: S_new %*% x in O(p)

  ## S-diagonal clamping: prevent covariance windup for sparse designs
  if (!is.null(S_max)) {
    d_S  <- diag(S_new)
    over <- d_S > S_max
    if (any(over)) {
      sf    <- ifelse(over, sqrt(S_max / d_S), 1.0)
      S_new <- S_new * outer(sf, sf)
    }
  }

  score <- (y - mu) * dmu / vmu
  if (!is.null(score_clip))
    score <- pmin(pmax(score, -score_clip), score_clip)

  list(beta = beta + gain * score, S = S_new)
}


#' Fit RLS-GLM over a dataset
#'
#' Fits a generalised linear model sequentially using the online natural-gradient
#' / IRLS algorithm with Sherman-Morrison Fisher information updates. Processes
#' each observation in `O(p^2)` time; requires `O(p^2)` storage.
#'
#' @param X Numeric matrix `n x p`. Design matrix.
#' @param y Numeric vector of length `n`. Response vector.
#' @param family An R `family` object. Default [stats::gaussian()].
#' @param lambda Forgetting factor in `(0, 1]`. Default `1`. With `lambda < 1`
#'   the algorithm tracks time-varying parameters; effective memory window is
#'   `1 / (1 - lambda)`.
#' @param S0_scale Positive scalar. Initial inverse Fisher is
#'   `S0_scale * diag(p)`. Default `100`.
#' @param eta_clip Positive scalar. Linear predictor clipping bound. Default `10`.
#' @param beta_init Optional numeric vector of length `p`. Starting coefficients.
#'   For Poisson, `c(log(mean(y)), rep(0, p - 1))` avoids large early residuals.
#' @param score_clip Optional positive scalar. Score clipping bound.
#'   For Poisson a natural scale is `5 * sqrt(mean(y))`.
#' @param offset Optional numeric vector of length `n` added to the linear
#'   predictor, or `NULL` (no offset). Typical use: `log(exposure)` in
#'   Poisson rate models.
#' @param S_max Optional positive scalar. Upper bound for diagonal elements
#'   of the inverse Fisher information `S`. After each update, any diagonal
#'   entry exceeding `S_max` is rescaled (along with its row and column) back
#'   to `S_max`. This prevents covariance windup when `lambda < 1` and the
#'   design matrix has sparse indicator columns (e.g. interval dummies in
#'   piecewise-exponential survival models). Default `NULL` (no clamping).
#'
#' @return A `stream_fit` object.
#'
#' @seealso [rls_glm_update()], [isgd_fit()]
#'
#' @examples
#' \donttest{
#' # Logistic regression
#' set.seed(42)
#' n <- 1000; p <- 4
#' X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
#' beta0 <- c(-0.5, 1, -1, 0.5)
#' y <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta0)))
#' fit <- rls_glm_fit(X, y, family = binomial())
#' round(coef(fit), 3)
#' }
#'
#' @export
rls_glm_fit <- function(X, y, family = stats::gaussian(),
                         lambda = 1.0, S0_scale = 100.0,
                         eta_clip = 10.0, beta_init = NULL,
                         score_clip = NULL, offset = NULL,
                         S_max = NULL) {
  if (nrow(X) != length(y))
    stop("nrow(X) must equal length(y)")
  if (!is.numeric(lambda) || length(lambda) != 1 || lambda <= 0 || lambda > 1)
    stop("'lambda' must be a scalar in (0, 1]")
  if (!is.numeric(S0_scale) || S0_scale <= 0)
    stop("'S0_scale' must be a positive scalar")
  if (!is.null(beta_init) && length(beta_init) != ncol(X))
    stop("length of 'beta_init' must equal ncol(X)")
  if (!is.numeric(eta_clip) || length(eta_clip) != 1 || eta_clip <= 0)
    stop("'eta_clip' must be a positive scalar")
  if (!is.null(score_clip) &&
      (!is.numeric(score_clip) || length(score_clip) != 1 || score_clip <= 0))
    stop("'score_clip' must be a positive scalar")
  if (!is.null(offset)) {
    if (!is.numeric(offset) || length(offset) != nrow(X))
      stop("'offset' must be a numeric vector of length n, or NULL")
  }
  if (!is.null(S_max) &&
      (!is.numeric(S_max) || length(S_max) != 1 || S_max <= 0))
    stop("'S_max' must be a positive scalar")
  cl   <- match.call()
  n    <- nrow(X); p <- ncol(X)
  beta <- if (is.null(beta_init)) numeric(p) else beta_init
  S    <- S0_scale * diag(p)
  path    <- matrix(NA_real_, nrow = n, ncol = p)
  linkinv <- family$linkinv
  mu.eta  <- family$mu.eta
  variance <- family$variance
  has_clip   <- !is.null(score_clip)
  has_offset <- !is.null(offset)
  has_S_max  <- !is.null(S_max)
  needs_disp <- .needs_dispersion(family)
  pearson_ss <- if (needs_disp) 0.0 else NULL
  for (i in seq_len(n)) {
    x_i      <- X[i, ]
    offset_i <- if (has_offset) offset[i] else 0
    eta   <- pmin(pmax(as.numeric(crossprod(x_i, beta)) + offset_i, -eta_clip), eta_clip)
    mu    <- linkinv(eta)
    dmu   <- mu.eta(eta)
    vmu   <- variance(mu)
    w     <- dmu^2 / vmu
    Sx    <- as.vector(S %*% x_i)
    denom <- as.numeric(1.0 + w * crossprod(x_i, Sx))
    S     <- (S - w * tcrossprod(Sx) / denom) / lambda
    gain  <- Sx / (lambda * denom)
    ## S-diagonal clamping
    if (has_S_max) {
      d_S  <- diag(S)
      over <- d_S > S_max
      if (any(over)) {
        sf <- ifelse(over, sqrt(S_max / d_S), 1.0)
        S  <- S * outer(sf, sf)
      }
    }
    score <- (y[i] - mu) * dmu / vmu
    if (has_clip) score <- pmin(pmax(score, -score_clip), score_clip)
    beta  <- beta + gain * score
    path[i, ] <- beta
    if (needs_disp) {
      eta_post <- pmin(pmax(as.numeric(crossprod(x_i, beta)) + offset_i, -eta_clip),
                       eta_clip)
      mu_post  <- linkinv(eta_post)
      pearson_ss <- pearson_ss + (y[i] - mu_post)^2 / variance(mu_post)
    }
  }
  new_stream_fit(
    beta        = beta,
    beta_path   = path,
    S           = S,
    family      = family,
    method      = "RLS-GLM",
    call        = cl,
    hyperparams = list(lambda     = lambda,
                       S0_scale   = S0_scale,
                       eta_clip   = eta_clip,
                       score_clip = score_clip,
                       S_max      = S_max),
    n_obs       = n,
    pearson_ss  = pearson_ss
  )
}
