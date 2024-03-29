\name{tpcm}
\alias{tpcm}
\title{Student T Parsimonious Clustering Models}
\description{
Carries out model-based clustering or classification using some or all of the 14 parsimonious Student T clustering models (TPCM).}
\usage{
tpcm(data=NULL, G=1:3, mnames=NULL,
		start=2, label=NULL, 
		veo=FALSE, da=c(1.0),
		nmax=1000, atol=1e-8, mtol=1e-8, mmax=10, burn=5,
		pprogress=FALSE, pwarning=FALSE, stochastic=FALSE, 
		constrained = FALSE, seed=123) 
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
  \item{stochastic}{
	If \code{TRUE} , it will run stochastic E step variant. 
}
\item{constrained}{
	If \code{TRUE}, it will constrain the degrees of freedom for student-t to be the same for all clusters. 
}
  \item{seed}{
	The seed for the run, default is 123
}
}
\details{
The data \code{x} are either clustered or classified using Skew-t mixture models with some or all of the 14 parsimonious covariance structures described in Celeux & Govaert (1995). The algorithms given by Celeux & Govaert (1995) is used for 12 of the 14 models; the "EVE" and "VVE" models use the algorithms given in Browne & McNicholas (2014). Starting values are very important to the successful operation of these algorithms and so care must be taken in the interpretation of results. 
}
\value{
An object of class \code{tpcm} is a list with components:
\item{map}{A vector of integers indicating the maximum \emph{a posteriori} classifications for the best model.}
\item{model_objs}{A list of all estimated models with parameters returned from the C++ call.}
\item{best_model}{A class of vgpcm_best containing; the number of groups for the best model, the covariance structure, and Bayesian Information Criterion (BIC) value.}
\item{loglik}{The log-likelihood values from fitting the best model.}
\item{z}{A matrix giving the raw values upon which \code{map} is based.}
\item{BIC}{A G by mnames by 3 dimensional array with values pertaining to BIC calculations. (legacy)}
\item{gpar}{A list object for each cluster pertaining to parameters. (legacy)}
\item{startobject}{The type of object inputted into \code{start}.}
\item{row_tags}{If there were NAs in the original dataset, a vector of indices referencing the row of the imputed vectors is given.}
\subsection{Best Model}{
An object of class \code{stpcm_best} is a list with components:
}
\item{model_type}{A string containg summarized information about the type of model estimated (Covariance structure and number of groups).}
\item{model_obj}{An internal list containing all parameters returned from the C++ call. }
\item{BIC}{Bayesian Index Criterion (positive scale, bigger is better).}
\item{loglik}{Log liklihood from the estimated model. }
\item{nparam}{Number of a parameters in the mode.}
\item{startobject}{The type of object inputted into \code{start}.}
\item{G}{An integer representing the number of groups.}
\item{cov_type}{A string representing the type of covariance matrix (see 14 models).}
\item{status}{Convergence status of EM algorithm according to Aitken's Acceleration}
\item{map}{A vector of integers indicating the maximum \emph{a posteriori} classifications for the best model.}
\item{row_tags}{If there were NAs in the original dataset, a vector of indices referencing the row of the imputed vectors is given.}
\subsection{Internal Objects}{
All classes contain an internal list called \code{model_obj} or \code{model_objs} with the following components:
}
\item{zigs}{ a posteori matrix }
\item{G}{An integer representing the number of groups.}
\item{sigs}{A vector of covariance matrices for each group} 
\item{mus}{A vector of location vectors for each group}
\item{vgs}{A vector containing estimated gamma parameters for each group}
}
\note{
Dedicated \code{print}, \code{plot} and \code{summary} functions are available for objects of class \code{vgpcm}.

}
\author{
Nik Pocuca, Ryan P. Browne and Paul D. McNicholas.

Maintainer: Paul D. McNicholas <mcnicholas@math.mcmaster.ca>
}
\references{
McNicholas, P.D. (2016), \emph{Mixture Model-Based Classification}. Boca Raton: Chapman & Hall/CRC Press

Browne, R.P. and McNicholas, P.D. (2014). Estimating common principal components in high dimensions. \emph{Advances in Data Analysis and Classification} \bold{8}(2), 217-226.

Andrews, J.L. and McNicholas, P.D. (2012), 'Model-based clustering, classification, and discriminant analysis via mixtures of multivariate t-distributions', Statistics and Computing 22(5), 1021-1029.

Celeux, G., Govaert, G. (1995). Gaussian parsimonious clustering models. \emph{Pattern Recognition} \bold{28}(5), 781-793.
}
\examples{
data("x2")

\dontrun{

### estimate "VVV" "EVE"
ax = tpcm(x2, G=1:3, mnames=c("VVV","EVE"), start=0)
summary(ax)
ax


### estimate all 14 covariance structures 
ax = tpcm(x2, G=1:3, mnames=NULL, start=0)
summary(ax)
ax

}

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}