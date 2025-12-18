\name{dmgh}
\alias{dmgh}
\title{
Density of multivariate Generalized Hyperbolic distribution
}
\description{
Computes the density of an observation for a multivariate Generalized Hyperbolic distribution.
}
\usage{
dmgh(x, mu, alpha, Sig, omega, lambda, LOG = FALSE)
}
\arguments{
  \item{x}{
A numeric vector of dimension (1 x p).
}
  \item{mu}{
A (1 x p) numeric vector of location values. 
}
  \item{alpha}{
A (1 x p) numeric vector of skewness values. 
}
  \item{Sig}{
A (p x p) numeric covariance matrix.
}
  \item{omega}{
A numeric value for the first gamma parameter.
}
  \item{lambda}{
A numeric value for the second gamma parameter.
}
  \item{LOG}{
A logical value indicating if the logarithm of the density is returned (default: LOG = FALSE).
}
}
\value{
A numeric value of the density of the observation \code{x} for the multivariate Generalized Hyperbolic distribution with parameters \code{mean}, \code{alpha}, \code{Sig},  \code{omega} and \code{lambda}.
}
\examples{
x = c(1.2, 0.4, 0.8)
mu = c(1, 0, 2)
alpha = c(0.2, -0.1, 0.3)
Sig = matrix(c(1.0, 0.5, 0.5,
                0.5, 1.0, 0.5,
                0.5, 0.5, 1.0), nrow = 3, ncol = 3)
omega = 1
lambda = 2
dens = dmgh(x, mu, alpha, Sig, omega, lambda, LOG = FALSE)
dens
}