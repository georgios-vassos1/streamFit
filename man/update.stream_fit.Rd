\name{update.stream_fit}
\alias{update.stream_fit}
\title{Update a \code{stream_fit} object with a new observation}
\usage{
\method{update}{stream_fit}(object, x, y, ...)
}
\arguments{
\item{object}{A \code{stream_fit} object produced by \code{rls_fit()},
\code{rls_glm_fit()}, or \code{isgd_fit()}.}
\item{x}{Numeric vector of length \code{p}. Feature vector for the new
observation.}
\item{y}{Scalar response.}
\item{...}{Ignored.}
}
\value{
An updated \code{stream_fit} object. Note: \code{beta_path} is
\strong{not} extended (to avoid O(n^2) copy cost); only \code{beta} and
the sufficient statistics (\code{S}, \code{rss}, \code{A_hat}, etc.) are
updated.
}
\description{
Processes a single new observation and returns an updated \code{stream_fit}
object. This is the primary online/streaming interface: fit an initial model
on a batch, then call \code{update()} for each new observation as it arrives.
}
\examples{
\donttest{
set.seed(1)
n <- 200; p <- 3
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
y <- rbinom(n, 1, 1 / (1 + exp(-X \%*\% c(-0.5, 1, -0.8))))

## Fit on first 100 observations
fit <- rls_glm_fit(X[1:100, ], y[1:100], family = binomial())

## Stream in the remaining observations one at a time
for (i in 101:n) {
  fit <- update(fit, x = X[i, ], y = y[i])
}
round(coef(fit), 3)
}
}
