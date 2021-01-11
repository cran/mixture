\name{get_best_model}
\alias{get_best_model}
\title{Best Model Extractor}
\description{
Carries out model-based clustering or classification using some or all of the 14 parsimonious Gaussian clustering models (GPCM).}
\usage{
get_best_model(gpcm_model)
}
\arguments{
  \item{gpcm_model}{An input of class \code{gpcm}.}
}
\details{
Extracts the best model based on BIC. 
}
\value{
An object of class \code{gpcm_best} is a list with components:
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
}
\author{
Nik Pocuca, Ryan P. Browne and Paul D. McNicholas.

Maintainer: Paul D. McNicholas <mcnicholas@math.mcmaster.ca>
}
\references{
Browne, R.P. and McNicholas, P.D. (2014). Estimating common principal components in high dimensions. \emph{Advances in Data Analysis and Classification} \bold{8}(2), 217-226.

Zhou, H. and Lange, K. (2010). On the bumpy road to the dominant mode. \emph{Scandinavian Journal of Statistics} \bold{37}, 612-631. 

Celeux, G., Govaert, G. (1995). Gaussian parsimonious clustering models. \emph{Pattern Recognition} \bold{28}(5), 781-793.
}
\examples{
\dontrun{

# load dataset and perform model search. 
data(x2)
data_in <- matrix(x2,ncol = 2)
mm <- mixture::gpcm(data = data_in,G = 1:7,
           start = 0,
           veo = FALSE,pprogress=FALSE)

# get best model 
best = get_best_model(mm)
best
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
