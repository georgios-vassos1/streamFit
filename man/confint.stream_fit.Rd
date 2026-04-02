\name{confint.stream_fit}
\alias{confint.stream_fit}
\title{Confidence intervals for stream_fit coefficients}
\usage{
\method{confint}{stream_fit}(object, parm, level = 0.95, ...)
}
\arguments{
\item{object}{A \code{stream_fit} object.}
\item{parm}{Integer or character vector specifying which parameters to
  include. Default: all.}
\item{level}{Confidence level. Default \code{0.95}.}
\item{...}{Ignored.}
}
\value{
A matrix with columns for the lower and upper confidence bounds.
}
\description{
Computes Wald confidence intervals based on the asymptotic
variance-covariance matrix.
}
