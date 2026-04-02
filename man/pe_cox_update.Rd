\name{pe_cox_update}
\alias{pe_cox_update}
\title{Update a piecewise-exponential Cox model with a new observation}
\usage{
pe_cox_update(object, start, stop, status, x_new)
}
\arguments{
\item{object}{A \code{pe_cox_fit} object produced by \code{\link{pe_cox_fit}}.}
\item{start}{Scalar. Left endpoint of the subject's risk interval.}
\item{stop}{Scalar. Right endpoint (event or censoring time).}
\item{status}{Integer 0/1. Event indicator.}
\item{x_new}{Numeric vector of length \code{q}. Covariate values.}
}
\value{
An updated \code{pe_cox_fit} object.
}
\description{
Takes a single new observation (a subject's risk interval from \code{start}
to \code{stop}), expands it against the existing interval breaks, and calls
\code{\link[=update.stream_fit]{update()}} for each resulting
person-interval record. This is the streaming interface for online Cox
regression.
}
\examples{
\donttest{
set.seed(1)
n <- 200
x1 <- rnorm(n)
X <- matrix(x1, ncol = 1)
T_event <- rexp(n, rate = 0.5 * exp(0.3 * x1))
C <- runif(n, 2, 5)
time <- pmin(T_event, C); status <- as.integer(T_event <= C)
breaks <- c(0, 1, 2, 3, 4, 5)

## Fit on first 100
fit <- pe_cox_fit(rep(0, 100), time[1:100], status[1:100],
                  X[1:100, , drop = FALSE], breaks)

## Stream remaining observations
for (i in 101:n) {
  fit <- pe_cox_update(fit, 0, time[i], status[i], X[i, ])
}
fit$beta
}
}
\seealso{
\code{\link{pe_cox_fit}}, \code{\link{surv_to_pe}}
}
