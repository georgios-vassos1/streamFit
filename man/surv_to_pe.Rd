\name{surv_to_pe}
\alias{surv_to_pe}
\title{Expand survival data into piecewise-exponential person-interval records}
\usage{
surv_to_pe(start, stop, status, X, breaks, id = NULL)
}
\arguments{
\item{start}{Numeric vector. Left endpoint of each subject's risk interval
  (use \code{0} for standard right-censored data).}
\item{stop}{Numeric vector. Right endpoint (event or censoring time).}
\item{status}{Integer 0/1 vector. Event indicator.}
\item{X}{Numeric matrix (\code{n x q}). Time-fixed covariates.}
\item{breaks}{Numeric vector of \code{K + 1} cut points defining \code{K}
  intervals. Must be sorted in increasing order.}
\item{id}{Optional vector of subject identifiers (for recurrent events).
  Defaults to \code{seq_along(start)}.}
}
\value{
A list with components:
\describe{
  \item{\code{d}}{Integer vector of event indicators per person-interval.}
  \item{\code{e}}{Numeric vector of exposure (time at risk) per person-interval.}
  \item{\code{X_pe}}{Design matrix: \code{cbind(interval_dummies, X_expanded)}.
    The first \code{K} columns are indicator variables for each interval;
    the remaining \code{q} columns are the original covariates.}
  \item{\code{id_pe}}{Expanded subject IDs.}
  \item{\code{interval}}{Integer vector of interval indices per record.}
  \item{\code{breaks}}{The break points used.}
}
}
\description{
Converts counting-process survival data \code{(start, stop, status)} into
person-interval records suitable for Poisson GLM fitting. Each subject is
split across the intervals defined by \code{breaks}, producing one record per
(subject, interval) pair where the subject is at risk. The design matrix
includes interval-dummy columns (one per interval, no intercept) so the
baseline hazard is fully parameterized.
}
\examples{
# One subject, event at time 2.5, breaks at 0, 1, 2, 3
pe <- surv_to_pe(start = 0, stop = 2.5, status = 1,
                 X = matrix(0.5, nrow = 1, ncol = 1),
                 breaks = c(0, 1, 2, 3))
pe$e   # exposures: 1, 1, 0.5
pe$d   # events:    0, 0, 1
}
\seealso{
\code{\link{pe_cox_fit}}, \code{\link{pe_cox_update}}
}
