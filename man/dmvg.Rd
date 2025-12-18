\name{dmvg}
\alias{dmvg}
\title{
Density of multivariate Variance Gamma distribution
}
\description{
Computes the density of an observation for a multivariate Variance Gamma distribution.
}
\usage{
dmvg(x, mu, alpha, Sig, gamma, LOG = FALSE)
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
  \item{gamma}{
A numeric value for the gamma parameter.
}
  \item{LOG}{
A logical value indicating if the logarithm of the density is returned (default: LOG = FALSE).
}
}
\value{
A numeric value of the density of the observation \code{x} for the multivariate Variance Gamma distribution with parameters \code{mean}, \code{alpha}, \code{Sig}, and \code{gamma}.
}
%\references{
%% ~put references to the literature/web site here ~
%}
\examples{
x = c(1.2, 0.4, 0.8)
mu = c(1, 0, 2)
alpha = c(0.2, -0.1, 0.3)
Sig = matrix(c(1.0, 0.5, 0.5,
                0.5, 1.0, 0.5,
                0.5, 0.5, 1.0), nrow = 3, ncol = 3)
gamma = 4
dens = dmvg(x, mu, alpha, Sig, gamma, LOG = FALSE) 
dens
}