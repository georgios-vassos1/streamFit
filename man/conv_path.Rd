\name{conv_path}
\alias{conv_path}
\title{Coefficient convergence path}
\usage{
conv_path(beta_path, ref)
}
\arguments{
\item{beta_path}{Numeric matrix of size \code{n x p}. Row \code{t} is the
coefficient estimate after processing observation \code{t}.}
\item{ref}{Numeric vector of length \code{p}. Reference coefficient vector
(e.g. batch MLE or true parameter).}
}
\value{
Numeric vector of length \code{n} giving the L2 distance at each step.
}
\description{
Computes the Euclidean (L2) distance between each row of a coefficient path
matrix and a reference vector. Useful for assessing convergence to the batch
MLE or a known true parameter vector.
}
\examples{
set.seed(1)
n <- 200; p <- 3
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta0 <- c(0.5, -1, 0.8)
y <- rbinom(n, 1, 1 / (1 + exp(-X \%*\% beta0)))
fit <- rls_glm_fit(X, y, family = binomial())
d <- conv_path(fit$beta_path, beta0)
plot(d, type = "l", xlab = "Observation", ylab = "L2 distance to truth")
}
