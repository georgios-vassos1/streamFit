\name{nobs.stream_fit}
\alias{nobs.stream_fit}
\title{Number of observations in a stream_fit object}
\usage{
\method{nobs}{stream_fit}(object, ...)
}
\arguments{
\item{object}{A \code{stream_fit} object.}
\item{...}{Ignored.}
}
\value{
Integer number of observations processed.
}
\description{
Returns the number of observations used to fit the model, including any
observations added via \code{\link{update.stream_fit}}.
}
