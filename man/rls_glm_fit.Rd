\name{rls_glm_fit}
\alias{rls_glm_fit}
\title{Fit RLS-GLM over a dataset}
\usage{
rls_glm_fit(X, y, family = stats::gaussian(), lambda = 1, S0_scale = 100,
  eta_clip = 10, beta_init = NULL, score_clip = NULL, offset = NULL)
}
\arguments{
\item{X}{Numeric matrix \code{n x p}. Design matrix.}
\item{y}{Numeric vector of length \code{n}. Response vector.}
\item{family}{An R \code{family} object. Default \code{\link[stats]{gaussian}}.}
\item{lambda}{Forgetting factor in \code{(0, 1]}. Default \code{1}.}
\item{S0_scale}{Positive scalar. Initial inverse Fisher is
\code{S0_scale * diag(p)}. Default \code{100}.}
\item{eta_clip}{Positive scalar. Linear predictor clipping bound. Default \code{10}.}
\item{beta_init}{Optional numeric vector of length \code{p}. Starting coefficients.}
\item{score_clip}{Optional positive scalar. Score clipping bound.}
\item{offset}{Optional numeric vector of length \code{n} added to the linear
predictor, or \code{NULL} (no offset). Typical use: \code{log(exposure)} in
Poisson rate models.}
}
\value{
A \code{stream_fit} object.
}
\description{
Fits a generalised linear model sequentially using the online natural-gradient
/ IRLS algorithm with Sherman-Morrison Fisher information updates. Processes
each observation in O(p^2) time.
}
\examples{
\donttest{
# Logistic regression
set.seed(42)
n <- 1000; p <- 4
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta0 <- c(-0.5, 1, -1, 0.5)
y <- rbinom(n, 1, 1 / (1 + exp(-X \%*\% beta0)))
fit <- rls_glm_fit(X, y, family = binomial())
round(coef(fit), 3)
}
}
\seealso{
\code{\link{rls_glm_update}}, \code{\link{isgd_fit}}
}
