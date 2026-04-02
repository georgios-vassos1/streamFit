\name{rls_fit}
\alias{rls_fit}
\title{Fit RLS over a dataset (linear / Gaussian model)}
\usage{
rls_fit(X, y, lambda = 1, S0_scale = 100, beta_init = NULL)
}
\arguments{
\item{X}{Numeric matrix \code{n x p}. Design matrix (include a column of ones
for an intercept).}
\item{y}{Numeric vector of length \code{n}. Response vector.}
\item{lambda}{Forgetting factor in \code{(0, 1]}. Default \code{1}.}
\item{S0_scale}{Positive scalar. Initial inverse covariance is
\code{S0_scale * diag(p)}. Default \code{100}.}
\item{beta_init}{Optional numeric vector of length \code{p}. Starting coefficient
vector. Defaults to zero.}
}
\value{
A \code{stream_fit} object with elements:
\describe{
  \item{\code{beta}}{Final coefficient vector.}
  \item{\code{beta_path}}{Matrix \code{n x p}: coefficient estimate after each step.}
  \item{\code{S}}{Final inverse covariance matrix.}
}
}
\description{
Fits a linear model sequentially using Recursive Least Squares, processing
each observation in O(p^2) time. Equivalent to weighted least squares
with exponential forgetting when \code{lambda < 1}.
}
\examples{
\donttest{
set.seed(42)
n <- 500; p <- 4
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta_true <- c(0.5, 1, -1, 0.5)
y <- X \%*\% beta_true + rnorm(n)
fit <- rls_fit(X, y)
round(coef(fit), 3)
plot(fit)
}
}
\seealso{
\code{\link{rls_update}}, \code{\link{rls_glm_fit}}
}
