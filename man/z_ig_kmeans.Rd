\name{z_ig_kmeans}
\alias{z_ig_kmeans}
\title{K-means Initialization}
\description{
Generates an initialization matrix for a dataset X using k-means.}
\usage{
  z_ig_kmeans(X,g)
}
\arguments{
  \item{X}{
A matrix or data frame such that rows correspond to observations and columns correspond to variables. Note that this function currently only works with multivariate data p > 1. Note. NO NAS allowed. 
}
  \item{g}{
  An integer representing the number of groups. 
}
}
\value{
A numeric matrix is returned of size n times g, with row sums adding up to 1. 
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
#data("x2")
#z_init <- z_ig_kmeans(x2,g=3)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
