\name{pcm}
\alias{pcm}
\title{Parsimonious Clustering Models}
\description{
Carries out model-based clustering or classification using some or all of the 14 parsimonious settings with any one of the GPCM, STPCM, VGPCM, or GHPCM families.}
\usage{
pcm(data=NULL, G=1:3, pcmfamily=c(gpcm,vgpcm,tpcm),
		mnames=NULL, start=2, label=NULL, 
		veo=FALSE, da=c(1.0),
		nmax=1000, atol=1e-8, mtol=1e-8, mmax=10, burn=5,
		pprogress=FALSE, pwarning=TRUE, seed=123) 
}
\arguments{
  \item{data}{
A matrix or data frame such that rows correspond to observations and columns correspond to variables. Note that this function currently only works with multivariate data p > 1. 
}
  \item{G}{
	A sequence of integers giving the number of components to be used.
}
\item{pcmfamily}{
	The family of models to be used. If \code{NULL} then all are fitted. 
}
  \item{mnames}{
	The models (i.e., covariance structures) to be used. If \code{NULL} then all 14 are fitted. 
}
  \item{start}{
	If \code{0} then the random soft function is used for initialization.
	If \code{1} then the random hard function is used for initialization.
	If \code{2} then the kmeans function is used for initialization. 
	If \code{>2} then multiple random soft starts are used for initialization. 
	If \code{is.matrix} then matrix is used as an initialization matrix as along as it has non-negative elements. Note: only models with the same number of columns of this matrix will be fit.
}
\item{label}{
	If \code{NULL} then the data has no known groups.
	If \code{is.integer} then some of the observations have known groups. If \code{label[i]=k} then observation belongs to group  \code{k}. If \code{label[i]=0} then observation has no known group. See Examples. 
}
  \item{veo}{
	Stands for "Variables exceed observations". If \code{TRUE} then if the number variables in the model exceeds the number of observations the model is still fitted.
}
  \item{da}{
  Stands for Determinstic Annealing. A vector of doubles. 
}
  \item{nmax}{
	The maximum number of iterations each EM algorithm is allowed to use. 
}
  \item{atol}{
	A number specifying the epsilon value for the convergence criteria used in the EM algorithms. For each algorithm, the criterion is based on the difference between the log-likelihood at an iteration and an asymptotic estimate of the log-likelihood at that iteration. This asymptotic estimate is based on the Aitken acceleration and details are given in the References. 
}
  \item{mtol}{
	A number specifying the epsilon value for the convergence criteria used in the M-step in the EM algorithms.
}
\item{mmax}{
	The maximum number of iterations each M-step is allowed in the GEM algorithms.
}
\item{burn}{
	The burn in period for imputing data. (Missing observations are removed and a model is estimated seperately before placing an imputation step within the EM.)
}
  \item{pprogress}{
	If \code{TRUE} print the progress of the function.
}
  \item{pwarning}{
	If \code{TRUE} print the warnings.
}
	\item{seed}{
	The seed for the run, default is 123
}
}
\details{
The data \code{x} are either clustered or classified using Skew-t mixture models with some or all of the 14 parsimonious covariance structures described in Celeux & Govaert (1995). The algorithms given by Celeux & Govaert (1995) is used for 12 of the 14 models; the "EVE" and "VVE" models use the algorithms given in Browne & McNicholas (2014). Starting values are very important to the successful operation of these algorithms and so care must be taken in the interpretation of results. 
}
\value{
An object of class \code{pcm} is a list with components:
\item{gpcm}{If applicable, the output of running the Gaussian Parsimonious Family.}
\item{vgpcm}{If applicable, the output of running the Variance-Gamma Parsimonious Family.}
\item{stpcm}{If applicable, the output of running the Skew-T Parsimonious Family.}
\item{ghpcm}{If applicable, the output of running the Generalized Hyperbolic Parsimonious Family.}
\item{best_model}{An object of corresponding to the output of the best performing family.}
}
\note{
Dedicated \code{print}, and \code{summary} functions are available for objects of class \code{pcm}, \code{gpcm}, \code{ghpcm}, \code{stpcm}, or \code{vgpcm}.

}
\author{
Nik Pocuca, Ryan P. Browne and Paul D. McNicholas.

Maintainer: Paul D. McNicholas <mcnicholas@math.mcmaster.ca>
}
\references{
McNicholas, P.D. (2016), \emph{Mixture Model-Based Classification}. Boca Raton: Chapman & Hall/CRC Press

Browne, R.P. and McNicholas, P.D. (2014). Estimating common principal components in high dimensions. \emph{Advances in Data Analysis and Classification} \bold{8}(2), 217-226.

Browne, R.P. and McNicholas, P.D. (2015), 'A mixture of generalized hyperbolic distributions', Canadian Journal of Statistics 43(2), 176-198. 

Wei, Y., Tang, Y. and McNicholas, P.D. (2019), 'Mixtures of generalized hyperbolic distributions and mixtures of skew-t distributions for model-based clustering with incomplete data', Computational Statistics and Data Analysis 130, 18-41.

Celeux, G., Govaert, G. (1995). Gaussian parsimonious clustering models. \emph{Pattern Recognition} \bold{28}(5), 781-793.
}
\examples{
data("x2")

\dontrun{

### estimate "VVV" "EVE"
ax = pcm(sx3, G=1:3, mnames=c("VVV","EVE"), start=0)
summary(ax)

print(ax)

}

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}