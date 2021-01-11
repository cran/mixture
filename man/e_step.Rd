\name{e_step}
\alias{e_step}
\title{Expectation Step}
\description{Calculates the expectation of class memberships, and imputes if missing values for a given dataset.}
\usage{
e_step(data, model_obj, start=0, nu = 1.0) 
}
\arguments{
  \item{data}{
A matrix or data frame such that rows correspond to observations and columns correspond to variables. Note that this function currently only works with multivariate data p > 1. 
}
  \item{start}{ Start values in this context are only used for imputation. Non-missing values have their expectation of class memberships calculated directly.
	If \code{0} then the random soft function is used for initialization.
	If \code{1} then the random hard function is used for initialization.
	If \code{2} then the kmeans function is used for initialization. 
	If \code{is.matrix} then matrix is used as an initialization matrix as along as it has non-negative elements. Note: only models with the same number of columns of this matrix will be fit.
}
\item{model_obj}{
  A gpcm_best, vgpcm_best, stpcm_best, ghpcm_best, and salpcm_best object class.  
}
\item{nu}{
  deterministic annealing for the class membership E-step. 
}
}
\details{
  This will only work on a dataset with the same dimension as estimated in the model. \code{e_step} will also work for missing values, provided that there is at least one non-missing entry. 
}
\value{
Returns a list with the following components:
\item{X}{A matrix of the original dataset plus imputed values if applicable.}
\item{origX}{A matrix of the original dataset including missing values.}
\item{map}{A vector of integers indicating the maximum \emph{a posteriori} classifications for the best model.}
\item{z}{A matrix giving the raw values upon which \code{map} is based.}
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

# lets try imputing some missing data. 
x2NA <- x2
x2NA[5,1] <- NA
x2NA[140,2] <- NA
x2NA[99,1] <- NA

# calculate expectation
expect <- e_step(data=x2NA,start = 0,nu = 1.0,model_obj = best)

# plot imputed entries and compare with original 
plot(x2,col = "grey")
points(expect$X[expect$row_tags+1,],col = "blue", pch = 20,cex = 2) # blue are imputed values.
points(x2[expect$row_tags+1,], col = "red" , pch = 20,cex = 2) # red are original values.
legend(-2,2,legend = c("imputed","original"),col = c("blue","red"),pch = 20)
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
