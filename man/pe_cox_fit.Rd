\name{pe_cox_fit}
\alias{pe_cox_fit}
\title{Fit an online piecewise-exponential Cox model}
\usage{
pe_cox_fit(start, stop, status, X, breaks,
           method = c("rls_glm", "isgd"), lambda = 1, ...)
}
\arguments{
\item{start}{Numeric vector. Left endpoint of each subject's risk interval.}
\item{stop}{Numeric vector. Right endpoint (event or censoring time).}
\item{status}{Integer 0/1 vector. Event indicator.}
\item{X}{Numeric matrix (\code{n x q}). Time-fixed covariates.}
\item{breaks}{Numeric vector of \code{K + 1} cut points defining \code{K}
  intervals.}
\item{method}{Character: \code{"rls_glm"} (default) or \code{"isgd"}.}
\item{lambda}{Forgetting factor in \code{(0, 1]}. Default \code{1}.}
\item{...}{Additional arguments passed to the underlying fitting function
  (e.g. \code{S0_scale}, \code{score_clip}, \code{eta_clip}, \code{S_max}).
  When using \code{lambda < 1}, setting \code{S_max} (e.g. \code{S_max = 1})
  is recommended to prevent covariance windup on the sparse interval-dummy
  columns.}
}
\value{
A list of class \code{"pe_cox_fit"} containing:
\describe{
  \item{\code{stream_fit}}{The underlying \code{stream_fit} object.}
  \item{\code{breaks}}{Interval cut points.}
  \item{\code{n_intervals}}{Number of intervals \code{K}.}
  \item{\code{covariate_names}}{Names of the \code{q} covariate columns.}
  \item{\code{baseline_log_hazard}}{First \code{K} coefficients
    (log baseline hazard per interval).}
  \item{\code{beta}}{Last \code{q} coefficients (covariate effects).}
}
}
\description{
Convenience wrapper that expands survival data via \code{\link{surv_to_pe}},
then fits a Poisson GLM with \code{offset = log(exposure)} using
\code{\link{rls_glm_fit}} or \code{\link{isgd_fit}}. The first \code{K}
coefficients parameterize the log baseline hazard in each interval; the
remaining coefficients are the covariate (log hazard-ratio) effects.

With forgetting factor \code{lambda < 1}, the online estimator tracks
time-varying regression coefficients, analogous to the dynamic Cox regression
estimated by \code{timereg::timecox()} in batch mode.
}
\examples{
\donttest{
set.seed(42)
n <- 500
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.5)
X <- cbind(x1, x2)
lam0 <- 0.5
T_event <- rexp(n, rate = lam0 * exp(0.5 * x1 - 0.3 * x2))
C <- runif(n, 2, 5)
time <- pmin(T_event, C); status <- as.integer(T_event <= C)
breaks <- seq(0, max(time) + 0.01, length.out = 6)
fit <- pe_cox_fit(rep(0, n), time, status, X, breaks,
                  method = "rls_glm", lambda = 1)
fit$beta
}
}
\seealso{
\code{\link{surv_to_pe}}, \code{\link{pe_cox_update}},
\code{\link{rls_glm_fit}}, \code{\link{isgd_fit}}
}
