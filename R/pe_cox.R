#' Expand survival data into piecewise-exponential person-interval records
#'
#' Converts counting-process survival data \code{(start, stop, status)} into
#' person-interval records suitable for Poisson GLM fitting.
#' Each subject is split across the intervals defined by \code{breaks}, producing
#' one record per (subject, interval) pair where the subject is at risk.
#' The design matrix includes interval-dummy columns (one per interval, no
#' intercept) so the baseline hazard is fully parameterized.
#'
#' @param start Numeric vector. Left endpoint of each subject's risk interval
#'   (use \code{0} for standard right-censored data).
#' @param stop Numeric vector. Right endpoint (event or censoring time).
#' @param status Integer 0/1 vector. Event indicator.
#' @param X Numeric matrix (\code{n x q}). Time-fixed covariates.
#' @param breaks Numeric vector of \code{K + 1} cut points defining \code{K}
#'   intervals. Must be sorted in increasing order.
#' @param id Optional vector of subject identifiers (for recurrent events).
#'   Defaults to \code{seq_along(start)}.
#'
#' @return A list with components:
#'   \describe{
#'     \item{\code{d}}{Integer vector of event indicators per person-interval.}
#'     \item{\code{e}}{Numeric vector of exposure (time at risk) per person-interval.}
#'     \item{\code{X_pe}}{Design matrix: \code{cbind(interval_dummies, X_expanded)}.
#'       The first \code{K} columns are indicator variables for each interval
#'       (baseline hazard parameterization); the remaining \code{q} columns are
#'       the original covariates replicated for each record.}
#'     \item{\code{id_pe}}{Expanded subject IDs.}
#'     \item{\code{interval}}{Integer vector of interval indices per record.}
#'     \item{\code{breaks}}{The break points used (stored for downstream use).}
#'   }
#'
#' @examples
#' # One subject, event at time 2.5, breaks at 0, 1, 2, 3
#' pe <- surv_to_pe(start = 0, stop = 2.5, status = 1,
#'                  X = matrix(0.5, nrow = 1, ncol = 1),
#'                  breaks = c(0, 1, 2, 3))
#' pe$e   # exposures: 1, 1, 0.5
#' pe$d   # events:    0, 0, 1
#'
#' @seealso \code{\link{pe_cox_fit}}, \code{\link{pe_cox_update}}
#' @export
surv_to_pe <- function(start, stop, status, X, breaks, id = NULL) {
  ## --- input validation ------------------------------------------------
  n <- length(start)
  if (length(stop) != n || length(status) != n)
    stop("'start', 'stop', and 'status' must have the same length")
  if (!is.matrix(X)) X <- as.matrix(X)
  if (nrow(X) != n)
    stop("nrow(X) must equal length(start)")
  if (is.unsorted(breaks) || length(breaks) < 2L)
    stop("'breaks' must be a sorted numeric vector with at least 2 elements")
  if (is.null(id)) id <- seq_len(n)

  K <- length(breaks) - 1L
  q <- ncol(X)

  ## --- pre-allocate (upper bound: n * K records) -----------------------
  max_rec <- n * K
  d_out   <- integer(max_rec)
  e_out   <- numeric(max_rec)
  id_out  <- vector(typeof(id), max_rec)
  int_out <- integer(max_rec)
  X_cov   <- matrix(0, nrow = max_rec, ncol = q)
  X_int   <- matrix(0, nrow = max_rec, ncol = K)

  idx <- 0L
  for (i in seq_len(n)) {
    for (k in seq_len(K)) {
      t_lo <- max(start[i], breaks[k])
      t_hi <- min(stop[i], breaks[k + 1L])
      if (t_hi > t_lo) {
        idx <- idx + 1L
        e_out[idx]    <- t_hi - t_lo
        d_out[idx]    <- as.integer(status[i] * (stop[i] <= breaks[k + 1L]))
        id_out[idx]   <- id[i]
        int_out[idx]  <- k
        X_int[idx, k] <- 1
        X_cov[idx, ]  <- X[i, ]
      }
    }
  }

  ## --- trim to actual number of records --------------------------------
  keep <- seq_len(idx)
  X_pe <- cbind(X_int[keep, , drop = FALSE], X_cov[keep, , drop = FALSE])

  ## column names
  int_names <- paste0("interval_", seq_len(K))
  cov_names <- colnames(X)
  if (is.null(cov_names)) cov_names <- paste0("x", seq_len(q))
  colnames(X_pe) <- c(int_names, cov_names)

  list(
    d        = d_out[keep],
    e        = e_out[keep],
    X_pe     = X_pe,
    id_pe    = id_out[keep],
    interval = int_out[keep],
    breaks   = breaks
  )
}


#' Fit an online piecewise-exponential Cox model
#'
#' Convenience wrapper that expands survival data via \code{\link{surv_to_pe}},
#' then fits a Poisson GLM with \code{offset = log(exposure)} using
#' \code{\link{rls_glm_fit}} or \code{\link{isgd_fit}}. The first \code{K}
#' coefficients parameterize the log baseline hazard in each interval; the
#' remaining coefficients are the covariate (log hazard-ratio) effects.
#'
#' With forgetting factor \code{lambda < 1}, the online estimator tracks
#' time-varying regression coefficients \eqn{\beta(t)}, analogous to the
#' dynamic Cox regression estimated by \code{timereg::timecox()} in batch mode.
#'
#' @param start Numeric vector. Left endpoint of each subject's risk interval.
#' @param stop Numeric vector. Right endpoint (event or censoring time).
#' @param status Integer 0/1 vector. Event indicator.
#' @param X Numeric matrix (\code{n x q}). Time-fixed covariates.
#' @param breaks Numeric vector of \code{K + 1} cut points defining \code{K}
#'   intervals.
#' @param method Character: \code{"rls_glm"} (default) or \code{"isgd"}.
#' @param lambda Forgetting factor in \code{(0, 1]}. Default \code{1}.
#' @param ... Additional arguments passed to the underlying fitting function
#'   (e.g. \code{S0_scale}, \code{score_clip}, \code{eta_clip}, \code{S_max}).
#'   When using \code{lambda < 1}, setting \code{S_max} (e.g. \code{S_max = 1})
#'   is recommended to prevent covariance windup on the sparse interval-dummy
#'   columns.
#'
#' @return A list of class \code{"pe_cox_fit"} containing:
#'   \describe{
#'     \item{\code{stream_fit}}{The underlying \code{stream_fit} object.}
#'     \item{\code{breaks}}{Interval cut points.}
#'     \item{\code{n_intervals}}{Number of intervals \code{K}.}
#'     \item{\code{covariate_names}}{Names of the \code{q} covariate columns.}
#'     \item{\code{baseline_log_hazard}}{First \code{K} coefficients
#'       (log baseline hazard per interval).}
#'     \item{\code{beta}}{Last \code{q} coefficients (covariate effects).}
#'   }
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' n <- 500
#' x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.5)
#' X <- cbind(x1, x2)
#' lam0 <- 0.5  # constant baseline hazard
#' T_event <- rexp(n, rate = lam0 * exp(0.5 * x1 - 0.3 * x2))
#' C <- runif(n, 2, 5)
#' time <- pmin(T_event, C); status <- as.integer(T_event <= C)
#' breaks <- seq(0, max(time) + 0.01, length.out = 6)
#' fit <- pe_cox_fit(rep(0, n), time, status, X, breaks,
#'                   method = "rls_glm", lambda = 1)
#' fit$beta  # should be close to c(0.5, -0.3)
#' }
#'
#' @seealso \code{\link{surv_to_pe}}, \code{\link{pe_cox_update}},
#'   \code{\link{rls_glm_fit}}, \code{\link{isgd_fit}}
#' @export
pe_cox_fit <- function(start, stop, status, X, breaks,
                       method = c("rls_glm", "isgd"),
                       lambda = 1.0, ...) {
  method <- match.arg(method)
  if (!is.matrix(X)) X <- as.matrix(X)

  ## expand to person-interval records
  pe <- surv_to_pe(start, stop, status, X, breaks)

  K <- length(pe$breaks) - 1L
  q <- ncol(X)
  p <- K + q

  ## Records are kept in the natural (subject-by-subject, interval-by-interval)
  ## order from surv_to_pe. This gives balanced updates across intervals.

  ## offset = log(exposure)
  log_e <- log(pe$e)

  ## warm start: flat hazard + zero covariate effects
  avg_rate <- sum(pe$d) / sum(pe$e)
  if (avg_rate <= 0) avg_rate <- 0.01
  beta_init <- c(rep(log(avg_rate), K), rep(0, q))

  dots <- list(...)

  ## PE data has binary response (d in {0,1}) with small expected rates.
  ## S0_scale = 1 (not 100) prevents overshoot on early event records.
  if (method == "rls_glm") {
    sf <- rls_glm_fit(
      X          = pe$X_pe,
      y          = pe$d,
      family     = stats::poisson(),
      lambda     = lambda,
      S0_scale   = if (!is.null(dots$S0_scale)) dots$S0_scale else 1.0,
      eta_clip   = if (!is.null(dots$eta_clip)) dots$eta_clip else 10.0,
      beta_init  = beta_init,
      score_clip = dots$score_clip,
      offset     = log_e,
      S_max      = dots$S_max
    )
  } else {
    sf <- isgd_fit(
      X      = pe$X_pe,
      y      = pe$d,
      family = stats::poisson(),
      offset = log_e
    )
  }

  ## extract named coefficients
  cov_names <- colnames(X)
  if (is.null(cov_names)) cov_names <- paste0("x", seq_len(q))

  all_beta <- coef(sf)
  baseline_log_haz <- all_beta[seq_len(K)]
  names(baseline_log_haz) <- paste0("interval_", seq_len(K))
  beta_cov <- all_beta[K + seq_len(q)]
  names(beta_cov) <- cov_names

  structure(
    list(
      stream_fit         = sf,
      breaks             = breaks,
      n_intervals        = K,
      covariate_names    = cov_names,
      baseline_log_hazard = baseline_log_haz,
      beta               = beta_cov
    ),
    class = "pe_cox_fit"
  )
}


#' Update a piecewise-exponential Cox model with a new observation
#'
#' Takes a single new observation (a subject's risk interval from \code{start}
#' to \code{stop}), expands it against the existing interval breaks, and calls
#' \code{\link[=update.stream_fit]{update()}} for each resulting
#' person-interval record. This is the streaming interface for online Cox
#' regression.
#'
#' @param object A \code{pe_cox_fit} object produced by \code{\link{pe_cox_fit}}.
#' @param start Scalar. Left endpoint of the subject's risk interval.
#' @param stop Scalar. Right endpoint (event or censoring time).
#' @param status Integer 0/1. Event indicator.
#' @param x_new Numeric vector of length \code{q}. Covariate values.
#'
#' @return An updated \code{pe_cox_fit} object.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 200
#' x1 <- rnorm(n)
#' X <- matrix(x1, ncol = 1)
#' T_event <- rexp(n, rate = 0.5 * exp(0.3 * x1))
#' C <- runif(n, 2, 5)
#' time <- pmin(T_event, C); status <- as.integer(T_event <= C)
#' breaks <- c(0, 1, 2, 3, 4, 5)
#'
#' ## Fit on first 100
#' fit <- pe_cox_fit(rep(0, 100), time[1:100], status[1:100],
#'                   X[1:100, , drop = FALSE], breaks)
#'
#' ## Stream remaining observations
#' for (i in 101:n) {
#'   fit <- pe_cox_update(fit, 0, time[i], status[i], X[i, ])
#' }
#' fit$beta
#' }
#'
#' @seealso \code{\link{pe_cox_fit}}, \code{\link{surv_to_pe}}
#' @export
pe_cox_update <- function(object, start, stop, status, x_new) {
  if (!inherits(object, "pe_cox_fit"))
    stop("'object' must be a pe_cox_fit object")

  breaks <- object$breaks
  K <- object$n_intervals

  ## expand single observation into person-interval records
  pe <- surv_to_pe(
    start  = start,
    stop   = stop,
    status = status,
    X      = matrix(x_new, nrow = 1L),
    breaks = breaks
  )

  ## update for each person-interval record
  sf <- object$stream_fit
  for (j in seq_along(pe$d)) {
    sf <- update(sf,
                 x      = pe$X_pe[j, ],
                 y      = pe$d[j],
                 offset = log(pe$e[j]))
  }
  object$stream_fit <- sf

  ## refresh coefficient summaries
  all_beta <- coef(sf)
  object$baseline_log_hazard <- all_beta[seq_len(K)]
  names(object$baseline_log_hazard) <- paste0("interval_", seq_len(K))
  q <- length(all_beta) - K
  object$beta <- all_beta[K + seq_len(q)]
  names(object$beta) <- object$covariate_names

  object
}
