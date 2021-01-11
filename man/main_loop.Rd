\name{main_loop}
\alias{main_loop}
\title{GPCM Internal C++ Call}
\description{
This function is the internal C++ function call within the \code{gpcm} function. 
This is a raw C++ function call,  meaning it has no checks for proper inputs so it may fail to run without giving proper errors. 
Please ensure all arguements are valid. \code{main_loop} is useful for writing parallizations of the gpcm function. All arguement descriptions are given in terms of their corresponding C++ types.   
}
\usage{
main_loop(X, G, model_id, 
        model_type, in_zigs, 
        in_nmax, in_l_tol, in_m_iter_max,
        in_m_tol, anneals, t_burn = 5L) 
}
\arguments{
  \item{X}{
A numeric matrix of values with observations along rows, and dimension along columns. Is allowed to have NAs. (No full rows of NAs)
}
  \item{G}{
	A single positive integer value representing number of groups. 
}
  \item{model_id}{
 An integer representing the model_id, is useful for keeping track within parallizations. Not to be confused with model_type.  
}
  \item{model_type}{
	The type of covariance model you wish to run. Lexicon is given as follows: 
	"0" = "EII", "1" = "VII",  "2" = "EEI" ,  "3" = "EVI", "4" = "VEI",  "5" = "VVI", "6" = "EEE",  
                        "7" = "VEE", "8" = "EVE", "9" = "EEV", "10" = "VVE", "11" = "EVV", "12" = "VEV", "13" = "VVV"
  }
  \item{in_zigs}{
	  A n times G a posteriori matrix resembling the probability of observation i belonging to group G. Rows must sum to one, have the proper dimensions, and be positive.  
  }
  \item{in_nmax}{
	  Positive integer value resembling the maximum amount of iterations for the EM. 
  }
  \item{in_l_tol}{
	  A likelihood tolerance for convergence. 
  }
  \item{in_m_iter_max}{
	  For certain models, where applicable, the number of iterations for the maximization step. 
  }
  \item{in_m_tol}{
	  For certain models, where applicable, the tolerance for the maximization step. 
  }
  \item{anneals}{
	  A vector of doubles representing the deterministic annealing settings.
  }
  \item{t_burn}{
	  A positive integer representing the number of burn steps if missing data (NAs) are detected. 
  }
}
\details{
	Be extremly careful running this function, it is known to crash systems without proper exception handling. Consider using the package \code{parallel} to estimate all possible models at the same time. 
}
\value{
\item{zigs}{ a postereori matrix }
\item{G}{An integer representing the number of groups.}
\item{sigs}{A vector of covariance matrices for each group (note you may have to reshape this)} 
\item{mus}{A vector of mean vectors for each group}
\item{X}{Imputed dataset (if original had NAs)}
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

data("x2")
data_in = as.matrix(x2,ncol = 2)
n_iter = 1000

in_g = 3
n = dim(data_in)[1]
model_string <- "VVE"
in_model_type <- switch(model_string, "EII" = 0,"VII" = 1,  
              "EEI" = 2,  "EVI" = 3,  "VEI" = 4,  "VVI" = 5,  "EEE" = 6,  
              "VEE" = 7,  "EVE" = 8,  "EEV" = 9,  "VVE" = 10,
              "EVV" = 11,"VEV" = 12,"VVV" = 13)

zigs_in <- z_ig_random_soft(n,in_g)

m2 = main_loop(X = data_in, # data in
               G = 3, # number of groups
               model_id = 1, # model id for parallelization later
               model_type = in_model_type,
               in_zigs = zigs_in, # initializaiton
               in_nmax = n_iter, # number of iterations
               in_l_tol = 1e-12, # likilihood tolerance
               in_m_iter_max = 20, # maximium iterations for matrices
               in_m_tol = 1e-8,
               anneals=c(0.5,0.7,0.9,1)) 

plot(data_in,col = MAP(m2$zigs) + 1)
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{multivariate}
\keyword{cluster}
\keyword{classif}
