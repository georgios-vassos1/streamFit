\name{rls_update}
\alias{rls_update}
\title{Single-step RLS update (linear / Gaussian model)}
\usage{
rls_update(x, y, beta, S, lambda = 1, offset = 0)
}
\arguments{
\item{x}{Numeric vector of length \code{p}. Feature vector for the current
observation.}
\item{y}{Scalar response.}
\item{beta}{Numeric vector of length \code{p}. Current coefficient estimate.}
\item{S}{Numeric matrix \code{p x p}. Current inverse covariance matrix.}
\item{lambda}{Forgetting factor in \code{(0, 1]}. Default \code{1} (no forgetting).
Effective memory window: \code{1 / (1 - lambda)}.}
\item{offset}{Numeric scalar offset added to the linear predictor.
Typical use: \code{log(exposure)} in Poisson rate models. Default \code{0}.}
}
\value{
A list with:
\describe{
  \item{\code{beta}}{Updated coefficient vector (length \code{p}).}
  \item{\code{S}}{Updated inverse covariance matrix (\code{p x p}).}
}
}
\description{
Performs one step of the Recursive Least Squares algorithm for a linear
model. The inverse covariance matrix \code{S} is updated via the
Sherman-Morrison rank-1 formula, avoiding a full matrix inversion.
}
\examples{
\donttest{
set.seed(1)
p <- 3; beta_true <- c(1, -0.5, 0.8)
S <- 100 * diag(p); beta <- numeric(p)
for (i in seq_len(500)) {
  x    <- c(1, rnorm(p - 1))
  y    <- sum(x * beta_true) + rnorm(1)
  out  <- rls_update(x, y, beta, S)
  beta <- out$beta; S <- out$S
}
round(beta, 3)
}
}
\seealso{
\code{\link{rls_fit}} for fitting over a full dataset;
\code{\link{rls_glm_update}} for the GLM generalisation.
}
