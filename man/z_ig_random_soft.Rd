\name{z_ig_random_soft}
\alias{z_ig_random_soft}
\title{Random Soft Initialization}
\description{
Generates an initialization matrix of size n times g using random soft.}
\usage{
z_ig_random_soft(n,g)
}
\arguments{
  \item{n}{
  Number of rows, must be positive. 
}
  \item{g}{
  Number of columns, must be positive. 
}

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
z_init <- z_ig_random_soft(100,3)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
