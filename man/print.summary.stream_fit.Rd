\name{print.summary.stream_fit}
\alias{print.summary.stream_fit}
\title{Print a summary.stream_fit object}
\usage{
\method{print}{summary.stream_fit}(x, digits = 4L, signif.stars = TRUE, ...)
}
\arguments{
\item{x}{A \code{summary.stream_fit} object.}
\item{digits}{Number of significant digits. Default \code{4}.}
\item{signif.stars}{Logical. Show significance stars? Default \code{TRUE}.}
\item{...}{Ignored.}
}
\description{
Print method for summary objects produced by
\code{\link{summary.stream_fit}}. Mimics the output of
\code{print.summary.glm}.
}
