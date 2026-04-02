\name{vcov.stream_fit}
\alias{vcov.stream_fit}
\title{Variance-covariance matrix for a stream_fit object}
\usage{
\method{vcov}{stream_fit}(object, ...)
}
\arguments{
\item{object}{A \code{stream_fit} object.}
\item{...}{Ignored.}
}
\value{
A \code{p x p} variance-covariance matrix.
}
\description{
Extracts the estimated variance-covariance matrix of the coefficient
estimates. For RLS (Gaussian), this is \code{sigma2_hat * S}. For RLS-GLM,
this is the final inverse Fisher information \code{S}. For iSGD, this is the
Toulis & Airoldi (2017) asymptotic sandwich estimator
\code{(1/n) * A_inv \%*\% B \%*\% A_inv}.
}
