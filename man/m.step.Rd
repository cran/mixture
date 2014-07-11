\name{m.step}
\alias{m.step}
\title{M-Step}
\description{
Carries out the M-step for EM algorithm}
\usage{
m.step(data=NULL, covtype=NULL, w=NULL, v=1, D=NULL, mtol=NULL, mmax=NULL)
}
\arguments{
  \item{data}{
A matrix or data frame such that rows correspond to observations and columns correspond to variables. Note that this function currently only works with multivariate data p > 1. 
}
  \item{covtype}{
	A three letter sequence indicating the covariance structure.
}
  \item{w}{
	A nxG matrix of weights. 
}
  \item{v}{
	The value for deterministic annealing. If v=1 the standard estimate is used.
}
  \item{D}{
	An initial value for D. If \code{NULL} then the identity matrix is used.
}
  \item{mtol}{
	The convergence criteria for the m.step if an iterative procedure is necessary. 
}
  \item{mmax}{
	The maximum number of iterations for an iterative procedure. 
}
}
\details{
Carries out the M-step for EM algorithm
}
\value{
A list of the model parameters with the \code{mu}, \code{sigma},  \code{invsigma} and  \code{logdet} for each group.
}
 \examples{
data("x2")
u0 = runif(nrow(x2))
m.step(data=x2, covtype="VVV", w=cbind(u0,1-u0), v=1, D=NULL, mtol=1e-8, mmax=10)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
