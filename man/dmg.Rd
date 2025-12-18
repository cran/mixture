\name{dmg}
\alias{dmg}
\title{
Density of multivariate Gaussian distribution
}
\description{
Computes the density of an observation for a multivariate Gaussian distribution.
}
\usage{
dmg(x, mu, Sig, LOG = FALSE)
}
\arguments{
  \item{x}{
A numeric vector of dimension (1 x p).
}
  \item{mu}{
A (1 x p) numeric vector of location values. 
}
  \item{Sig}{
A (p x p) numeric covariance matrix.
}
  \item{LOG}{
A logical value indicating if the logarithm of the density is returned (default: LOG = FALSE).
}
}
\value{
A numeric value of the density of the observation \code{x} for the multivariate Gaussian distribution with parameters \code{mean}, and \code{Sig}.
}
\examples{
x = c(1.2, 0.4, 0.8)
mu = c(1, 0, 2)
Sig = matrix(c(1.0, 0.5, 0.5,
                0.5, 1.0, 0.5,
                0.5, 0.5, 1.0), nrow = 3, ncol = 3)
dens = dmg(x, mu, Sig, LOG = FALSE)
dens
}