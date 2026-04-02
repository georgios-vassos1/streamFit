\name{plot.stream_fit}
\alias{plot.stream_fit}
\title{Plot a stream_fit object}
\usage{
\method{plot}{stream_fit}(x, ref = NULL, coef_index = NULL, ...)
}
\arguments{
\item{x}{A \code{stream_fit} object.}
\item{ref}{Optional numeric vector of length \code{p}. If supplied, plots the L2
distance to \code{ref} (e.g. batch MLE or true parameter) rather than the raw
coefficient paths.}
\item{coef_index}{Integer vector of coefficient indices to plot when plotting
raw paths. Default: all coefficients.}
\item{...}{Additional arguments passed to \code{\link[graphics]{plot}} or
\code{\link[graphics]{matplot}}.}
}
\value{
Invisibly returns \code{x}.
}
\description{
Plots all coefficient trajectories over observations. If \code{ref} is supplied,
the L2 distance to the reference vector is plotted instead.
}
