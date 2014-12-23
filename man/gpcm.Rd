\name{gpcm}
\alias{gpcm}
\title{Gaussian Parsimonious Clustering Models}
\description{
Carries out model-based clustering or classification using some or all of the 14 parsimonious Gaussian clustering models (GPCM).}
\usage{
gpcm(data=NULL,  G=1:3, mnames=NULL, start=0, label=NULL, veo=FALSE, 
nmax=1000, atol=1e-8, mtol=1e-8, mmax=10, pprogress=FALSE, pwarning=TRUE) 
}
\arguments{
  \item{data}{
A matrix or data frame such that rows correspond to observations and columns correspond to variables. Note that this function currently only works with multivariate data p > 1. 
}
  \item{G}{
	A sequence of integers giving the number of components to be used.
}
  \item{mnames}{
	The models (i.e., covariance structures) to be used. If \code{NULL} then all 14 are fitted. 
}
  \item{start}{
	If \code{0} then the kmeans function is used for initialization.
	If a positive value is inputted then best out of ceiling(k) random initializations are used. 
	If \code{is.vector} then deterministic annealing is used with the given sequence of values in [0,1]; cf. Zhou and Lange (2010). 
	If \code{is.matrix} then matrix is used as an initialization matrix as along as it has non-negative elements. Note: only models with the same number of columns of this matrix will be fit.
	If \code{is.function} then this function is used for building an initialization matrix. See Examples. 
}
  \item{label}{
	If \code{NULL} then the data has no known groups.
	If \code{is.integer} then some of the observations have known groups. If \code{label[i]=k} then observation belongs to group  \code{k}. If \code{label[i]=0} then observation has no known group. See Examples. 
}
  \item{veo}{
	If \code{TRUE} then if the number variables in the model exceeds the number of observations the model is still fitted.
}
  \item{nmax}{
	The maximum number of iterations each EM algorithm is allowed to use. 
}
  \item{atol}{
	A number specifying the epsilon value for the convergence criteria used in the EM algorithms. For each algorithm, the criterion is based on the difference between the log-likelihood at an iteration and an asymptotic estimate of the log-likelihood at that iteration. This asymptotic estimate is based on the Aitken acceleration and details are given in the References. 
}
  \item{mtol}{
	A number specifying the epsilon value for the convergence criteria used in the M-step in the GEM algorithms.
}
  \item{mmax}{
	The maximum number of iterations each M-step is allowed in the GEM algorithms.
}
  \item{pprogress}{
	If \code{TRUE} print the progress of the function.
}
  \item{pwarning}{
	If \code{TRUE} print the warnings.
}
}
\details{
The data \code{x} are either clustered or classified using Gaussian mixture models with some or all of the 14 parsimonious covariance structures described in Celeux & Govaert (1995). The algorithms given by Celeux & Govaert (1995) is used for 12 of the 14 models; the "EVE" and "VVE" models use the algorithms given in Browne & McNicholas (2012, 2013). Starting values are very important to the successful operation of these algorithms and so care must be taken in the interpretation of results. 
}
\value{
An object of class \code{gpcm} is a list with components:
\item{map}{A vector of integers indicating the maximum \emph{a posteriori} classifications for the best model.}
\item{gpar}{A list of the model parameters.}
\item{bicModel}{A list containing; the number of groups for the best model, the covariance structure, and Bayesian Information Criterion (BIC) value.}
\item{loglik}{The log-likelihood values from fitting the best model.}
\item{z}{A matrix giving the raw values upon which \code{map} is based.}
\item{BIC}{An array containing the log-likelihood (loglik), number of model parameters (npar) and BIC indexed by the covariance structure and number of components.}
\item{start}{The value inputted into \code{start}.}
\item{startobject}{The type of object inputted into \code{start}.}
}
\note{
Dedicated \code{print}, \code{plot} and \code{summary} functions are available for objects of class \code{gpcm}.
}
\author{
Ryan P. Browne, Aisha ElSherbiny and Paul D. McNicholas.

Maintainer: Ryan Browne <rbrowne@uoguelph.ca>
}
\references{
Browne, R.P. and McNicholas, P.D. (2014). Estimating common principal components in high dimensions. Advances in Data Analysis and Classification \bold{8}(2), 217-226.

Celeux, G., Govaert, G. (1995). Gaussian parsimonious clustering models. \emph{Pattern Recognition} \bold{28}(5), 781-793.
}
\examples{
data("x2")

# use k-means starts
ax0 = gpcm(x2, G=1:5, mnames=c("VVV", "EVE"),start=0, pprogress=TRUE, atol=1e-2)
ax0 = gpcm(x2, G=1:5, mnames=( "EVE"),start=0, pprogress=TRUE, atol=1e-2)
summary(ax0)
ax0

# use 6 random values for starting values
#ax6 = gpcm(x2, G=1:5, mnames=c("VVV", "EVE"),start= 6)
#summary(ax6)
#ax6

# use deterministic annealing for starting values
#axNULL = gpcm(x2, G=1:5, mnames=c("VVV", "EVE"), start=NULL)
#summary(axNULL)
#axNULL

# use your deterministic annealing for starting values
#vseq0 = rep(seq(.05, 1, length.out=5),each=2)
#axv = gpcm(x2, G=1:5, mnames=c("VVV", "EVE"), start=vseq0)
#summary(axv)
#axv

# Initialization using your own function 
#igparhc <-  function(data=NULL, g=NULL,covtype=NULL) {
#	lw = cutree(hclust(dist(data), "complete"),k=g)
#	w = matrix(0, nrow=nrow(data), ncol=g)
#	for (j in 1:ncol(w)) w[,j] = as.numeric( lw ==j )
#	return(w)
#}
#axhclust = gpcm(x2, G=1:5, mnames=c("VVV", "EVE"),start= igparhc)
#summary(axhclust)
#axhclust

# Estimate all 14 covariance structures from k-means starts 
#ax = gpcm(x2, G=1:5, start=0)
#summary(ax)
#ax

# model based classification
#x2.label = numeric(nrow(x2))
#x2.label[c(10,50, 110, 150, 210, 250)] = c(1,1,2,2,3,3)
#plot(x2, col=x2.label)
#axl = gpcm(x2, G=1:5, mnames=c("VVV", "EVE"), label=x2.label)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
