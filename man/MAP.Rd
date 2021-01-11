\name{MAP}
\alias{MAP}
\title{Maximum \emph{a posterori}}
\description{
Generates labels from a classification matrix z}
\usage{
  MAP(z_ig)
}
\arguments{
  \item{z_ig}{
  A classification matrix of positive numbers in which all rows must sum to one. 
}
}
\value{
A numeric matrix is returned of size n times g, with row sums adding up to 1. 
}
\author{
Nik Pocuca, Ryan P. Browne and Paul D. McNicholas.

Maintainer: Paul D. McNicholas <mcnicholas@math.mcmaster.ca>
}
\examples{
  \dontrun{

# Simple example. 
MAP(z_ig_random_soft(100,2))

# import dataset. 
data(x2)
mm <- gpcm(data = as.matrix(x2),G = 1:7,
           start = 2,
           veo = FALSE,pprogress=FALSE)

best = get_best_model(mm)
# You can get labels using the internal object with MAP.
labs <- MAP(best$model_obj[[1]]$zigs)
# or you can just get labels directly. 
labs2 <- best$map
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
