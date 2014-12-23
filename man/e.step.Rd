\name{e.step}
\alias{e.step}
\title{E-Step}
\description{
Carries out the E-step for EM algorithm}
\usage{
e.step(data=NULL, gpar=NULL, labels=NULL, v=1)
}
\arguments{
  \item{data}{
A matrix or data frame such that rows correspond to observations and columns correspond to variables. Note that this function currently only works with multivariate data p > 1. 
}
  \item{gpar}{
	A list of the model parameters.
}
  \item{labels}{
	A vector of groups labels. If \code{NULL} none are known.
}
  \item{v}{
	The value for deterministic annealing. If v=1 the standard estimate is used.
}
}
\details{
Carries out the E-step for EM algorithm
}
\value{
A nxG matrix of weights. 
}
\examples{
data("x2")
u0 = runif(nrow(x2))
m0 = m.step(data=x2, covtype="VVV", w=cbind(u0,1-u0), D=NULL, mtol=1e-8, mmax=10)
w0 = e.step(data=x2, gpar=m0, labels=NULL, v=1)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
