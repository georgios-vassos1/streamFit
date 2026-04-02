\name{summary.stream_fit}
\alias{summary.stream_fit}
\title{Summary of a stream_fit object}
\usage{
\method{summary}{stream_fit}(object, ...)
}
\arguments{
\item{object}{A \code{stream_fit} object.}
\item{...}{Ignored.}
}
\value{
A \code{summary.stream_fit} object (invisibly) containing a coefficient
table with standard errors, test statistics, and p-values.
}
\description{
Produces a summary with coefficient table including standard errors,
test statistics, and p-values. Uses t-values for RLS (Gaussian) and
z-values for GLM methods.
}
