\name{ARI}
\alias{ARI}
\title{Adjusted Rand Index}
\description{
Calculates an adjusted for chance Rand index.}
\usage{
ARI(x,y)
}
\arguments{
  \item{x}{
  predictor class memberships 
}
  \item{y}{
   true class memberships
}

}
\author{
Nik Pocuca, Ryan P. Browne and Paul D. McNicholas.

Maintainer: Paul D. McNicholas <mcnicholas@math.mcmaster.ca>
}
\references{
Hubert, L. and P. Arabie (1985). Comparing partitions. \emph{Journal of Classification} \bold{2}(1), 193-218.
}
\examples{
x <- sample(1:10, size = 100, replace = TRUE)
y <- sample(1:10, size = 100, replace = TRUE)
ARI(x,y)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
