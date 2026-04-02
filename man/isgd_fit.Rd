\name{isgd_fit}
\alias{isgd_fit}
\title{Fit iSGD over a dataset}
\usage{
isgd_fit(X, y, family = stats::gaussian(), gamma1 = 1, alpha = 0.7,
         compute_vcov = TRUE)
}
\arguments{
\item{X}{Numeric matrix \code{n x p}. Design matrix.}
\item{y}{Numeric vector of length \code{n}. Response vector.}
\item{family}{An R \code{family} object. Default \code{\link[stats]{gaussian}}.}
\item{gamma1}{Positive scalar. Initial learning rate. Default \code{1}.}
\item{alpha}{Decay exponent in \code{(0.5, 1]}. Default \code{0.7}.}
\item{compute_vcov}{Logical. If \code{TRUE}, accumulate the sandwich covariance
  matrices \code{A_hat} and \code{B_hat} needed for asymptotic inference
  (Toulis & Airoldi, 2017). Adds \code{O(p^2)} cost per step.
  Default \code{TRUE}.}
}
\value{
A \code{stream_fit} object (without \code{S}, which iSGD does not maintain).
}
\description{
Fits a generalised linear model sequentially using Implicit Stochastic
Gradient Descent with the learning rate schedule
\code{gamma_t = gamma1 * t^(-alpha)}. Processes each observation in O(p) time.
}
\examples{
\donttest{
set.seed(42)
n <- 1000; p <- 4
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta0 <- c(-0.5, 1, -1, 0.5)
y <- rbinom(n, 1, 1 / (1 + exp(-X \%*\% beta0)))
fit <- isgd_fit(X, y, family = binomial(), gamma1 = 1, alpha = 0.7)
round(coef(fit), 3)
}
}
\references{
Toulis, P. and Airoldi, E. (2017). Asymptotic and finite-sample properties
of estimators based on stochastic gradients. \emph{Annals of Statistics}, 45(4),
1694--1727.
}
\seealso{
\code{\link{isgd_update}}, \code{\link{rls_glm_fit}}
}
