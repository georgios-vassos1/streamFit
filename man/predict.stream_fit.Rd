\name{predict.stream_fit}
\alias{predict.stream_fit}
\title{Predict method for a \code{stream_fit} object}
\usage{
\method{predict}{stream_fit}(object, newdata, type = c("response", "link"), ...)
}
\arguments{
\item{object}{A \code{stream_fit} object.}
\item{newdata}{Numeric matrix of new covariates (n_new x p). Must have the
same number of columns as the training design matrix.}
\item{type}{Character string: \code{"response"} (default) returns
\code{g^{-1}(X \%*\% beta)}; \code{"link"} returns \code{X \%*\% beta}.}
\item{...}{Ignored.}
}
\value{
Numeric vector of length \code{nrow(newdata)}.
}
\description{
Computes predicted values (on the response scale by default) for new data
using the final coefficient estimate.
}
