// GLOBAL COMPILATION FILE FOR EVER HEADER FILE 
// makes compilation time much faster for R when compiling the package. 


#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#define ARMA_HAVE_STD_ISFINITE
#define ARMA_HAVE_STD_ISINF
#define ARMA_HAVE_STD_ISNAN
#define ARMA_HAVE_STD_SNPRINTF
#define ARMA_DONT_PRINT_ERRORS




// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include <Rcpp.h> 

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]


// turn off above and turn on below when compiling the R package version
#include <armadillo>
// turn off above and turn on below when compiling the R package version
//#include "RcppArmadillo.h"
#include <iostream>
#include <vector>
#include "math.h"
#include <limits>
#include <stdlib.h>
#include <exception> 
#include <memory> 
#include <random>
#include "Cluster_Error.hpp"



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]
// [[Rcpp::depends(BH)]]    
#define NDEBUG 1
#include <gsl/gsl_errno.h>


// GAUSSIAN MODEL 
#include "gpcm_arma.hpp"



// GENERALIZED HYPERBOLIC MODEL

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "GH_Model/GH_Mixture_Model.h"

#include "ghpcm_arma.hpp"


// SKEW-T MODEL
#include "ST_Model/ST_Mixture_Model.h"
#include "stpcm_arma.hpp"


// VARIANCE GAMMA MODEL
#include "VG_Model/VG_Mixture_Model.h"
#include "vgpcm_arma.hpp"


#include "T_Model/T_Mixture_Model.hpp"
#include "tpcm_arma.hpp"
