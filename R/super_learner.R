#' Construct a learner specification
#'
#' Creates a learner object for use in [super_learner_fit()] and
#' [online_sl_fit()]. A learner wraps a `fit` function, a `predict` function,
#' and an optional `update` function behind a common interface.
#'
#' @param fit A function with signature `function(X, y, family, ...)` that
#'   returns a fitted model object.
#' @param predict A function with signature `function(model, newdata, ...)` that
#'   returns a numeric vector of **response-scale** predictions.
#' @param update Optional function with signature
#'   `function(model, x, y, family, ...)` that ingests a single new observation
#'   and returns an updated model. Used by [online_sl_fit()] to keep base
#'   learners current as new data arrives. If `NULL` (default), the base
#'   learner is held fixed during online weight updates.
#' @param name Optional character string used as a label. If `NULL`, the
#'   learner is labelled by its position in the library.
#'
#' @return A list of class `sl_learner` with elements `fit`, `predict`,
#'   `update`, and `name`.
#'
#' @seealso [super_learner_fit()], [online_sl_fit()]
#'
#' @examples
#' \donttest{
#' ## Learner with online update support via stream_fit
#' lrn <- make_learner(
#'   fit     = function(X, y, family, ...) rls_glm_fit(X, y, family = family),
#'   predict = function(model, newdata, ...) predict(model, newdata),
#'   update  = function(model, x, y, family, ...) update(model, x = x, y = y)
#' )
#' }
#'
#' @export
make_learner <- function(fit, predict, update = NULL, name = NULL) {
  if (!is.function(fit))
    stop("'fit' must be a function")
  if (!is.function(predict))
    stop("'predict' must be a function")
  if (!is.null(update) && !is.function(update))
    stop("'update' must be a function or NULL")
  structure(list(fit = fit, predict = predict, update = update, name = name),
            class = "sl_learner")
}


#' Family-appropriate clamping for predicted means
#'
#' Prevents degenerate deviance values without distorting predictions for
#' families where the mean is unbounded (e.g. Gaussian).
#'
#' * Binomial / quasi-binomial: clamp to `(eps, 1 - eps)`.
#' * Poisson / quasi-Poisson / Gamma / inverse.gaussian: clamp to `(eps, Inf)`.
#' * Gaussian and others: no clamping.
#'
#' @param mu Numeric vector of predicted means.
#' @param family An R `family` object.
#' @return Clamped numeric vector.
#' @keywords internal
.sl_clamp <- function(mu, family) {
  fam <- family$family
  if (grepl("^(binomial|quasibinomial)", fam))
    return(pmax(pmin(mu, 1 - .Machine$double.eps), .Machine$double.eps))
  if (grepl("^(poisson|quasipoisson|Gamma|inverse\\.gaussian|negbin)", fam))
    return(pmax(mu, .Machine$double.eps))
  mu
}


#' Build cross-validation folds
#'
#' @param n Integer. Number of observations.
#' @param k Integer. Number of folds.
#' @return A list of length `k`, each element an integer index vector of
#'   held-out observations.
#' @keywords internal
.sl_make_folds <- function(n, k) {
  idx <- sample.int(n)
  split(idx, cut(seq_len(n), breaks = k, labels = FALSE))
}


#' Build prequential (one-step-ahead) out-of-sample predictions
#'
#' For each time \eqn{t > n_{\min}}, trains each base learner on observations
#' \eqn{1,\dots,t-1} and records its prediction at \eqn{t} **before** seeing
#' \eqn{y_t}.  Rows \eqn{1,\dots,n_{\min}} are left as `NA` (burn-in).
#'
#' Learners that expose an `update` function are advanced one step at a time
#' (O(n) total work).  Learners without `update` are re-fitted from scratch
#' on the growing prefix at each step (O(n^2) fits); for large `n`, supplying
#' an online learner is strongly preferred.
#'
#' @param X Numeric matrix `n x p`.
#' @param y Numeric vector of length `n`.
#' @param family An R `family` object.
#' @param library Named list of `sl_learner` objects.
#' @param n_min Integer. Burn-in size; first model fit uses rows `1:n_min`.
#' @param lnames Character vector of learner names (length `L`).
#' @return Numeric matrix `n x L` with `NA` in rows `1:n_min`.
#' @keywords internal
.sl_seq_predictions <- function(X, y, family, library, n_min, lnames) {
  n <- nrow(X)
  L <- length(library)
  Z <- matrix(NA_real_, nrow = n, ncol = L, dimnames = list(NULL, lnames))

  ## Initialise each learner on the burn-in block
  models <- vector("list", L)
  for (j in seq_len(L))
    models[[j]] <- library[[j]]$fit(
      X[seq_len(n_min), , drop = FALSE], y[seq_len(n_min)], family)

  ## Prequential loop: predict at t, then update with (x_t, y_t)
  for (t in seq.int(n_min + 1L, n)) {
    x_t <- X[t, , drop = FALSE]

    for (j in seq_len(L)) {
      ## Record prediction before seeing y_t
      Z[t, j] <- library[[j]]$predict(models[[j]], x_t)

      ## Advance the model
      upd <- library[[j]]$update
      if (!is.null(upd)) {
        models[[j]] <- upd(models[[j]], X[t, ], y[t], family)
      } else {
        ## No online update: re-fit on all data up to and including t
        models[[j]] <- library[[j]]$fit(
          X[seq_len(t), , drop = FALSE], y[seq_len(t)], family)
      }
    }
  }

  Z
}


#' Fit meta-learner weights on stacked out-of-fold predictions
#'
#' Finds non-negative weights summing to one that minimise the total deviance
#' of the convex combination of response-scale predictions `Z %*% alpha`
#' against `y`. Uses a softmax reparametrisation so that `optim` operates over
#' an unconstrained vector.
#'
#' @param Z Numeric matrix `n x L` of out-of-fold predictions (response scale).
#' @param y Numeric vector of length `n`. Response.
#' @param family An R `family` object.
#' @return Named numeric vector of length `L` with non-negative weights summing
#'   to one.
#' @keywords internal
.sl_meta_weights <- function(Z, y, family) {
  L <- ncol(Z)
  if (L == 1L) return(stats::setNames(1, colnames(Z)))

  dev_resids <- family$dev.resids

  objective <- function(theta) {
    ## Numerically stable softmax
    theta  <- theta - max(theta)
    alpha  <- exp(theta)
    alpha  <- alpha / sum(alpha)
    mu_hat <- .sl_clamp(as.vector(Z %*% alpha), family)
    sum(dev_resids(y, mu_hat, rep(1.0, length(y))))
  }

  opt <- stats::optim(
    par    = rep(0.0, L),
    fn     = objective,
    method = "BFGS",
    control = list(maxit = 500L, reltol = 1e-10)
  )

  theta <- opt$par - max(opt$par)
  alpha <- exp(theta) / sum(exp(theta))
  stats::setNames(alpha, colnames(Z))
}


#' Fit a Super Learner
#'
#' Combines a library of base learners via cross-validation. Two CV strategies
#' are available:
#'
#' * **`"kfold"`** (default) — standard K-fold CV. Observations are randomly
#'   partitioned into `k` folds; each fold is held out in turn for prediction.
#'   Assumes i.i.d. observations.
#'
#' * **`"sequential"`** — prequential (one-step-ahead) CV for dependent or
#'   streaming data. At each time \eqn{t > n_{\min}}, the base learners are
#'   trained on observations \eqn{1,\ldots,t-1} and their predictions at
#'   \eqn{t} are recorded before \eqn{y_t} is revealed.  This respects the
#'   temporal ordering of the data and is valid under weak mixing conditions
#'   without assuming i.i.d. errors (Chambaz & van der Laan, 2014).
#'
#' After CV, all base learners are refitted on the full data and the
#' meta-learner finds the optimal convex combination of their predictions by
#' minimising the total deviance over the CV predictions.
#'
#' Each learner in `library` must be created with [make_learner()]:
#' * `fit(X, y, family, ...)` — trains the model and returns a fitted object.
#' * `predict(model, newdata, ...)` — returns a numeric vector of
#'   **response-scale** predictions.
#' * `update(model, x, y, family, ...)` (optional) — ingests one observation
#'   and returns an updated model. Required for O(n) sequential CV; without it
#'   the learner is re-fitted from scratch at every step (O(n^2)).
#'
#' @param X Numeric matrix `n x p`. Design matrix.
#' @param y Numeric vector of length `n`. Response vector.
#' @param family An R `family` object. Default [stats::gaussian()].
#' @param library A named list of `sl_learner` objects created with
#'   [make_learner()]. Must have at least one element.
#' @param k Integer. Number of cross-validation folds (only used when
#'   `cv_type = "kfold"`). Default `5`.
#' @param cv_type Character. Cross-validation strategy: `"kfold"` (default)
#'   or `"sequential"` (prequential).
#' @param n_min Integer. Burn-in size for sequential CV: the number of
#'   observations used to initialise the base learners before the first
#'   prequential prediction is recorded.  Defaults to
#'   `max(2 * ncol(X), 20L)`. Ignored when `cv_type = "kfold"`.
#'
#' @return An object of class `super_learner_fit` with elements:
#'   \describe{
#'     \item{`weights`}{Named numeric vector of meta-learner weights.}
#'     \item{`learners`}{List of base-learner models fitted on the full data.}
#'     \item{`library`}{The learner specifications supplied by the user.}
#'     \item{`family`}{The family object.}
#'     \item{`cv_risk`}{Named numeric vector of per-learner cross-validated
#'       deviance.}
#'     \item{`sl_cv_risk`}{Scalar. Cross-validated deviance of the ensemble
#'       itself (i.e. the deviance of `Z \%*\% weights` against `y`). By
#'       construction this is \eqn{\leq} `min(cv_risk)` (oracle inequality).}
#'     \item{`Z`}{Numeric matrix `n x L` of out-of-sample predictions from
#'       each base learner (response scale). Rows `1:n_min` are `NA` under
#'       sequential CV.}
#'     \item{`cv_type`}{Character. The CV strategy used (`"kfold"` or
#'       `"sequential"`).}
#'     \item{`k`}{Number of CV folds (k-fold only; `NA` otherwise).}
#'     \item{`n_min`}{Burn-in size (sequential only; `NA` otherwise).}
#'     \item{`n_obs`}{Number of observations.}
#'     \item{`call`}{The matched call.}
#'   }
#'
#' @references
#' Chambaz, A. & van der Laan, M. J. (2014). Targeting the optimal design in
#' randomized trials with binary outcomes and no covariate. *Int. J. Biostat.*,
#' 10(1), 97–131. (Sequential CV framework.)
#'
#' @seealso [make_learner()], [predict.super_learner_fit()], [online_sl_fit()]
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' n <- 500; p <- 4
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
#'     predict = function(model, newdata, ...) predict(model, newdata)
#'   )
#' )
#'
#' ## Standard k-fold
#' sl_kf <- super_learner_fit(X, y, family = binomial(), library = lib, k = 5)
#' print(sl_kf)
#'
#' ## Prequential CV (respects temporal ordering)
#' sl_sq <- super_learner_fit(X, y, family = binomial(), library = lib,
#'                             cv_type = "sequential", n_min = 50L)
#' print(sl_sq)
#' }
#'
#' @export
super_learner_fit <- function(X, y, family = stats::gaussian(),
                               library, k = 5L,
                               cv_type = c("kfold", "sequential"),
                               n_min = NULL) {
  cl      <- match.call()
  cv_type <- match.arg(cv_type)

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

  n <- nrow(X)
  L <- length(library)

  if (cv_type == "kfold") {
    if (!is.numeric(k) || length(k) != 1L || k < 2L || k > n)
      stop("'k' must be an integer in [2, nrow(X)]")
    k     <- as.integer(k)
    n_min <- NA_integer_
  } else {
    if (is.null(n_min))
      n_min <- max(2L * ncol(X), 20L)
    n_min <- as.integer(n_min)
    if (n_min < 1L || n_min >= n)
      stop("'n_min' must be a positive integer less than nrow(X)")
    if (n - n_min < L + 1L)
      stop("too few observations after burn-in to fit the meta-learner")
    k <- NA_integer_
  }

  ## Assign names to learners
  lnames <- names(library)
  if (is.null(lnames)) lnames <- character(L)
  for (i in seq_len(L)) {
    if (nchar(lnames[i]) == 0L) {
      lnames[i] <- if (!is.null(library[[i]]$name)) library[[i]]$name
                   else paste0("L", i)
    }
  }
  names(library) <- lnames

  ## --- Build stacked prediction matrix Z (n x L) ----------------------------
  if (cv_type == "kfold") {
    Z     <- matrix(NA_real_, nrow = n, ncol = L, dimnames = list(NULL, lnames))
    folds <- .sl_make_folds(n, k)

    for (fold in folds) {
      train_idx <- setdiff(seq_len(n), fold)
      X_tr <- X[train_idx, , drop = FALSE]
      y_tr <- y[train_idx]
      X_vl <- X[fold,      , drop = FALSE]

      for (j in seq_len(L)) {
        m_j        <- library[[j]]$fit(X_tr, y_tr, family)
        Z[fold, j] <- library[[j]]$predict(m_j, X_vl)
      }
    }
  } else {
    Z <- .sl_seq_predictions(X, y, family, library, n_min, lnames)
  }

  ## --- Restrict to rows with complete predictions ----------------------------
  ## For k-fold: all rows are complete. For sequential: rows 1:n_min are NA.
  ok <- complete.cases(Z)

  ## --- Per-learner CV risk (mean deviance over complete rows) ----------------
  cv_risk <- vapply(seq_len(L), function(j) {
    mu_j <- .sl_clamp(Z[ok, j], family)
    mean(family$dev.resids(y[ok], mu_j, rep(1.0, sum(ok))))
  }, numeric(1L))
  names(cv_risk) <- lnames

  ## --- Meta-learner weights --------------------------------------------------
  weights <- .sl_meta_weights(Z[ok, , drop = FALSE], y[ok], family)

  ## --- Ensemble CV risk (oracle inequality: sl_cv_risk <= min(cv_risk)) ------
  mu_sl      <- .sl_clamp(as.vector(Z[ok, , drop = FALSE] %*% weights), family)
  sl_cv_risk <- mean(family$dev.resids(y[ok], mu_sl, rep(1.0, sum(ok))))

  ## --- Refit all learners on full data ---------------------------------------
  learners <- vector("list", L)
  names(learners) <- lnames
  for (j in seq_len(L))
    learners[[j]] <- library[[j]]$fit(X, y, family)

  structure(
    list(
      weights    = weights,
      learners   = learners,
      library    = library,
      family     = family,
      cv_risk    = cv_risk,
      sl_cv_risk = sl_cv_risk,
      Z          = Z,
      cv_type    = cv_type,
      k          = k,
      n_min      = n_min,
      n_obs      = n,
      call       = cl
    ),
    class = "super_learner_fit"
  )
}


#' Print a `super_learner_fit` object
#'
#' @param x A `super_learner_fit` object.
#' @param digits Number of significant digits. Default `4`.
#' @param ... Ignored.
#' @export
print.super_learner_fit <- function(x, digits = 4L, ...) {
  cat("Super Learner\n")
  cat("Family      :", x$family$family, "/", x$family$link, "\n")
  cat("Observations:", x$n_obs, "\n")
  if (x$cv_type == "kfold") {
    cat("CV strategy : k-fold (k =", x$k, ")\n\n")
  } else {
    cat("CV strategy : sequential (n_min =", x$n_min, ")\n\n")
  }
  cat("Learner weights and CV risk:\n")
  tbl <- rbind(
    weight  = round(x$weights,  digits),
    cv_risk = round(x$cv_risk,  digits)
  )
  print(tbl)
  cat(sprintf("SL CV risk  : %.*f  (oracle inequality: <= min cv_risk)\n",
              digits, x$sl_cv_risk))
  invisible(x)
}


#' Extract meta-learner weights from a `super_learner_fit` object
#'
#' @param object A `super_learner_fit` object.
#' @param ... Ignored.
#' @return Named numeric vector of meta-learner weights.
#' @export
coef.super_learner_fit <- function(object, ...) object$weights


#' Predict method for a `super_learner_fit` object
#'
#' Computes ensemble predictions for new data by taking the weighted convex
#' combination of each base learner's response-scale prediction.
#'
#' @param object A `super_learner_fit` object.
#' @param newdata Numeric matrix of new covariates (`n_new x p`). Must have
#'   the same number of columns as the training design matrix.
#' @param type Character string: `"response"` (default) returns the
#'   ensemble prediction on the response scale; `"link"` applies the link
#'   function to that prediction.
#' @param ... Ignored.
#' @return Numeric vector of length `nrow(newdata)`.
#' @export
predict.super_learner_fit <- function(object, newdata,
                                       type = c("response", "link"), ...) {
  type <- match.arg(type)
  if (!is.matrix(newdata)) newdata <- as.matrix(newdata)

  L      <- length(object$learners)
  preds  <- matrix(NA_real_, nrow = nrow(newdata), ncol = L)
  for (j in seq_len(L)) {
    preds[, j] <- object$library[[j]]$predict(object$learners[[j]], newdata)
  }

  mu_hat <- as.vector(preds %*% object$weights)

  if (type == "link") object$family$linkfun(mu_hat) else mu_hat
}
