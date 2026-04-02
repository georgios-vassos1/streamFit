\name{isgd_update}
\alias{isgd_update}
\title{Single-step implicit SGD update}
\usage{
isgd_update(x, y, beta, gamma, family)
}
\arguments{
\item{x}{Numeric vector of length \code{p}. Feature vector.}
\item{y}{Scalar response.}
\item{beta}{Numeric vector of length \code{p}. Current coefficient estimate.}
\item{gamma}{Positive scalar. Learning rate for this step.}
\item{family}{An R \code{family} object.}
}
\value{
Updated coefficient vector (length \code{p}).
}
\description{
Performs one step of Implicit (proximal) Stochastic Gradient Descent for a
GLM. Solves the scalar implicit equation
\deqn{\xi_t = \gamma_t(y_t - g^{-1}(\eta_t + \xi_t \|x_t\|^2))}
then updates \code{beta_{t+1} = beta_t + xi_t * x_t}.
For Gaussian the equation has a closed form. For all other families a 1-D
root is found via Brent's method.
}
\examples{
\donttest{
set.seed(1)
p <- 3; beta <- numeric(p); fam <- binomial()
beta_true <- c(-0.5, 1, -0.8)
for (i in seq_len(1000)) {
  x    <- c(1, rnorm(p - 1))
  y    <- rbinom(1, 1, fam$linkinv(sum(x * beta_true)))
  beta <- isgd_update(x, y, beta, gamma = 1.0 * i^(-0.7), fam)
}
round(beta, 3)
}
}
\references{
Toulis, P. and Airoldi, E. (2017). Asymptotic and finite-sample properties
of estimators based on stochastic gradients. \emph{Annals of Statistics}, 45(4),
1694--1727.
}
\seealso{
\code{\link{isgd_fit}}, \code{\link{rls_glm_update}}
}
