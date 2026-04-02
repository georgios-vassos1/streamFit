\name{rls_glm_update}
\alias{rls_glm_update}
\title{Single-step RLS-GLM update}
\usage{
rls_glm_update(x, y, beta, S, family, lambda = 1, eta_clip = 10,
  score_clip = NULL)
}
\arguments{
\item{x}{Numeric vector of length \code{p}. Feature vector.}
\item{y}{Scalar response.}
\item{beta}{Numeric vector of length \code{p}. Current coefficient estimate.}
\item{S}{Numeric matrix \code{p x p}. Current inverse Fisher information.}
\item{family}{An R \code{family} object, e.g. \code{\link[stats]{binomial}},
\code{\link[stats]{poisson}}, \code{\link[stats]{gaussian}}.}
\item{lambda}{Forgetting factor in \code{(0, 1]}. Default \code{1}.}
\item{eta_clip}{Positive scalar. Linear predictor is clamped to
\code{[-eta_clip, eta_clip]}. Default \code{10}.}
\item{score_clip}{Optional positive scalar. Clamps the score contribution.}
}
\value{
A list with:
\describe{
  \item{\code{beta}}{Updated coefficient vector.}
  \item{\code{S}}{Updated inverse Fisher information matrix.}
}
}
\description{
Performs one step of the online natural-gradient / IRLS update for a
generalised linear model. The inverse Fisher information \code{S} is updated via
the Sherman-Morrison rank-1 formula. Reduces to \code{\link{rls_update}} for
\code{family = gaussian()}.
}
\examples{
\donttest{
set.seed(1)
p <- 3; beta_true <- c(-0.5, 1, -0.8)
S <- 10 * diag(p); beta <- numeric(p); fam <- binomial()
for (i in seq_len(1000)) {
  x    <- c(1, rnorm(p - 1))
  y    <- rbinom(1, 1, fam$linkinv(sum(x * beta_true)))
  out  <- rls_glm_update(x, y, beta, S, fam)
  beta <- out$beta; S <- out$S
}
round(beta, 3)
}
}
\seealso{
\code{\link{rls_glm_fit}}, \code{\link{isgd_update}}
}
