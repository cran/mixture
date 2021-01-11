// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
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
#include "Cluster_Error.hpp"

const double eps = 0.001; 




bool comparison_gp(double a, double b){
    double tolerance  = abs( a - b);
    bool result = tolerance < eps;
    return (result);    
}

// All dev has been done on vscode 
// GENERAL MIXTURE MODEL CLASS
#ifndef Mixture_Model_H
#define Mixture_Model_H
class Mixture_Model 
{
public: 
    int n; // number of observations
    int model_id;
    std::vector<double> n_gs;
    int p; // dimension
    int G;
    std::vector<double> log_dets; 
    std::vector< arma::rowvec > mus; // all mixture models have mean vectors
    std::vector< arma::mat > sigs; // all mixture models have covariance matrices
    std::vector< arma::mat > inv_sigs; // inverse of all sigs 
    arma::mat data; // data for the model
    arma::rowvec pi_gs; // mixing proportions
    arma::mat zi_gs; // posteriori 
    std::vector< arma::mat > Ws; // within cluster scattering matrix per cluster
    std::vector< double > logliks; // std vector containing logliklihoods for each iteration. 
    double tol_l; // tolerance for logliklihood check. 
    double nu; // deterministic annealing method. 

    arma::mat EYE; 

    // Missing Data 
    std::vector < arma::uvec > missing_tags; // holds the row tag, as well as dimension tags for missing data NAs.
    arma::uvec row_tags; // arma unsigned column vector for row tags contain the missing data. 
    // Missing Data methods. 
    void init_missing_tags(void); // initalizes the two variables above by going through the data and pulling location of NAs.  
    void EM_burn(int in_burn_steps); // runs the EM algorithm in_burn_steps number of times only on the Non missing data. 
    void E_step_only_burn(void); // this function is only used for an existing model. See e_step_internal
    void impute_init(void); // initialize x_m values for imputation so that there are no NAs in the dataset. 
    void impute_cond_mean(void); 

    // Constructor 
    Mixture_Model(arma::mat* in_data, int in_G, int model_id);
    // Deconstructor 
    virtual ~Mixture_Model();

    // General Methods
    double calculate_log_liklihood(void); // returns log
    double mahalanobis(arma::rowvec x, arma::rowvec mu, arma::mat inv_sig);  // calculates mh for specific term. 
    double log_density(arma::rowvec x, arma::rowvec mu, arma::mat inv_Sig, double log_det); // calculates a particular log-density for a specific group for a specific x 
    void E_step(); // performs the e-step calculation on the mixture_model, stores data in z_igs. 
    void M_step_props(); // calculate the proportions and n_gs
    void M_step_mus(); // calculates the m step for mean vector
    arma::mat mat_inverse(arma::mat X); // matrix inverse 
    void M_step_Ws(); // calculates both Wk and W
    virtual void m_step_sigs() {  Rcpp::Rcout  << "m_step for general, user should not be here" << std::endl; }; 
    virtual void set_defaults() {  Rcpp::Rcout  << "set defaults virtual, user should not be here" << std::endl; };
    virtual void set_m_iterations(int in_iter, double in_tol) {  Rcpp::Rcout  << "set iterations virtual, user should not be here" << std::endl; };
    virtual void set_m_iterations(int in_iter, double in_tol, arma::mat in_D) {  Rcpp::Rcout  << "set iterations virtual, user should not be here" << std::endl; };
    void track_lg_init(void); // grab the first calculation of the logliklihood and replace the 1.0 from the constructor
    bool track_lg(bool check); // calculate current log liklihood of the model. If bool is check aitken convergence. 

    // Debug functions 
    void sig_eye_init(); // sets each of the groups covariance matrix to identity. 
};
#endif

// spherical family class 
#ifndef SPHERICAL_FAMILY_H
#define SPHERICAL_FAMILY_H
class Spherical_Family: public Mixture_Model 
{
    public:
        using Spherical_Family::Mixture_Model::Mixture_Model;
        arma::mat lambda_sphere(arma::mat in_W,double in_n); // general covariance matrix calculation for spherical family
        // constant identity matrix of size p for spherical family  
        const arma::mat eye_I = arma::mat(p,p,arma::fill::eye); 
};
#endif

#ifndef EII_H
#define EII_H
// EII: Equal volume family 
class EII: public Spherical_Family
{
    public:
        using Spherical_Family::Spherical_Family;
        double lambda; // single volume parameter for all covariance matrices 
        void m_step_sigs(); // maximization step for EII model. 

};
#endif


#ifndef VII_H
#define VII_H
// VII: Equal volume family 
class VII: public Spherical_Family
{
    public:
        using Spherical_Family::Spherical_Family;
        void m_step_sigs(); // maximization step for EII model. 
};
#endif

// DIAGONAL FAMILY 
#ifndef DIAGONAL_FAMILY_H
#define DIAGONAL_FAMILY_H
class Diagonal_Family: public Mixture_Model
{
    public:
        using Mixture_Model::Mixture_Model;
};
#endif

// EEI
#ifndef EEI_H
#define EEI_H
class EEI: public Diagonal_Family
{
    public: 
        using Diagonal_Family::Diagonal_Family;
        void m_step_sigs(void); // Maximization step for EEI model // see page 11 of Calveux 1993
};
#endif 

// VEI 
#ifndef VEI_H
#define VEI_H
class VEI: public Diagonal_Family
{
    public: 
        using Diagonal_Family::Diagonal_Family;
        int m_iter_max; // number of iterations for iterative maximization see pg 11 for celeux
        double m_tol; 
        void m_step_sigs(void);
        void set_m_iterations(int in_iter, double in_tol);
        void set_defaults(void); 
};
#endif 

// EVI
#ifndef EVI_H
#define EVI_H
class EVI: public Diagonal_Family
{
    public:
        using Diagonal_Family::Diagonal_Family;
        void m_step_sigs(void); 
};
#endif 
// VVI 
#ifndef VVI_H
#define VVI_H
class VVI: public Diagonal_Family
{
    public:
        using Diagonal_Family::Diagonal_Family;
        void m_step_sigs(void); 
};
#endif 


class General_Family: public Mixture_Model
{
    public: 
        using Mixture_Model::Mixture_Model;

};


// EEE
#ifndef EEE_H
#define EEE_H
class EEE: public General_Family
{
    public:
        using General_Family::General_Family;
        void m_step_sigs(void);
};
#endif 

// VVV
#ifndef VVV_H
#define VVV_H
class VVV: public General_Family
{
    public: 
        using General_Family::General_Family;
        void m_step_sigs(void); 
};
#endif



// MARK TO DO: GENERAL FAMILY 

// EEV
#ifndef EEV_H
#define EEV_H
class EEV: public General_Family
{
    public:
        using General_Family::General_Family;
        void m_step_sigs(void); 
};
#endif


// VEV
#ifndef VEV_H
#define VEV_H
class VEV: public General_Family
{
    public:
        using General_Family::General_Family;
        int m_iter_max; // number of iterations for iterative maximization see pg 8 of Celeux
        double m_tol; 
        void m_step_sigs(void); 
        void set_m_iterations(int in_iter, double in_tol);
        void set_defaults(void); 
};
#endif

// EVV
#ifndef EVV_H
#define EVV_H
class EVV: public General_Family
{
    public: 
        using General_Family::General_Family; 
        void m_step_sigs(void); 
};
#endif


// VEE
#ifndef VEE_H
#define VEE_H
class VEE: public General_Family
{
    public: 
        using General_Family::General_Family; 
        int m_iter_max; // number of iterations for iterative maximization see pg 8 of Celeux
        double m_tol; 
        void m_step_sigs(void); 
        void set_m_iterations(int in_iter, double in_tol);
        void set_defaults(void); 
};
#endif



// USING RYAN BROWNE AND MCNICHOLAS 2014
// EVE
#ifndef EVE_H
#define EVE_H
class EVE: public General_Family
{
    public:
        using General_Family::General_Family;
        int m_iter_max; // number of matrix iterations 
        double m_tol; 
        arma::mat D;
        void m_step_sigs(void); 
        void set_m_iterations(int in_iter, double in_tol, arma::mat in_D);
        void set_defaults(void); 
};
#endif

// VVE
#ifndef VVE_H
#define VVE_H
class VVE: public General_Family 
{
    public:
        using General_Family::General_Family; 
        int m_iter_max; // number of matrix iterations 
        double m_tol; 
        arma::mat D;
        void m_step_sigs(void); 
        void set_m_iterations(int in_iter, double in_tol, arma::mat in_D);
        void set_defaults(void); 
};
#endif 




// WRAPPERS 



// METHODS 

/*
The following is a table of contents for what is contained in this file. 

// Ctrl F Search 
// GENERAL MIXTURE MODEL CLASS
-- Constructor for general mixture model 
--  general m step for proportions 
--  within cluster scattering matrix calculation every M step 
--  general m step for mus 
--  calculates the log-density of a multivariate normal (incomplete log lik)
--  loglik tracking
// BEGINING OF FAMILY CLASSES AND SPECIFIED MODELS 
- SPHERICAL FAMILY 
-- EII
-- VII
- DIAGONAL FAMILY 
-- EEI 
-- VEI 
-- EVI 
-- VVI
- GENERAL FAMILY 
-- EEE 
-- VEE
*/
#define loggy(x)  Rcpp::Rcout  << x << std::endl //debug 


/*
The following is a table of contents for what is contained in this file. 

// Ctrl F Search 
// GENERAL MIXTURE MODEL CLASS
-- Constructor for general mixture model 
--  general m step for proportions 
--  within cluster scattering matrix calculation every M step 
--  general m step for mus 
--  calculates the log-density of a multivariate normal (incomplete log lik)
--  loglik tracking
// BEGINING OF FAMILY CLASSES AND SPECIFIED MODELS 
- SPHERICAL FAMILY 
-- EII
-- VII
- DIAGONAL FAMILY 
-- EEI 
-- VEI 
-- EVI 
-- VVI
- GENERAL FAMILY 
-- EEE 
-- VEE
*/



// **********************   GENERAL MIXTURE MODEL CLASS ****************************************************************************************   
// Mixture model class contains data, n, p, G, mus, sigs, inv_sigs, zi_gs, pi_gs, Ws, logliks, and tolerance settings                          *
// *********************************************************************************************************************************************      
// Constructor for general mixture model 
Mixture_Model::Mixture_Model(arma::mat* in_data, int in_G, int in_model_id) 
{
  data = *in_data; // pass pointer to data, its always going to be constant anyway
  n = in_data->n_rows;
  p = in_data->n_cols; // dimensions 
  G = in_G;
  model_id = in_model_id; 
  nu = 1.0; 

  // set size of mus based on number of groups
  std::vector<arma::rowvec> in_mus;
  in_mus.assign(G, arma::rowvec(p, arma::fill::zeros));
  mus = in_mus;

  // set size of sigs based on number of groups
  std::vector < arma::mat > in_sigs;
  in_sigs.assign(G, arma::mat(p,p,arma::fill::zeros));
  sigs = in_sigs;
  inv_sigs = in_sigs; 

  // initialize mixing proportions 
  arma::rowvec in_pigs = arma::rowvec(G, arma::fill::zeros);

  // calculate n_gs 
  std::vector<double> inter_n_gs;
  std::vector<double> inter_log_dets; 
  inter_n_gs.assign(G, 0);
  inter_log_dets.assign(G,0.0); 

  // set them all to be equal for now 
  for(arma::uword g = 0; g < in_pigs.n_elem; g++){ 
      in_pigs[g] = 1.0/in_pigs.n_elem; 
      inter_n_gs[g] = in_pigs[g]*n; 
      }

  pi_gs = in_pigs;
  n_gs = inter_n_gs;
  log_dets = inter_log_dets; 

  // set all posterori to be 0 
  zi_gs = arma::mat(n,G,arma::fill::zeros); 

  // set Wks and W to be identity for now 
  std::vector<arma::mat> inter_Ws;
  inter_Ws.assign(G,arma::mat(p,p,arma::fill::eye));
  Ws = inter_Ws; 

  // set up log liklihoods tracking 
  std::vector<double> vect{1.0};
  logliks = vect; 
  tol_l = 1e-6;
  EYE = arma::eye(p,p); 
}



// Deconstructor for general mixture model family 
Mixture_Model::~Mixture_Model() 
{
  // Rcpp::Rcout  << "De-allocating memory" << "\n";
}


// general m step for proportions 
void Mixture_Model::M_step_props()
{
  // initialize mixing proportions and n_gs 
  arma::rowvec in_pigs = arma::rowvec(G, arma::fill::zeros);
  std::vector<double> inter_n_gs;
  inter_n_gs.assign(G, 0);
  
  // calculate new mixing proportions and n_gs 
  for(int g = 0; g < G; g++)
  {
    for(int i = 0; i < n; i++)
    {
        // estimated number of observations
        inter_n_gs[g] += zi_gs.at(i,g); // apparenlty this is unused to the compiler? 
    }
    // mixing proportion
    in_pigs[g] = inter_n_gs[g]/n;
  }

  // set new values 
  n_gs = inter_n_gs; 
  pi_gs = in_pigs; 
}

// general m step for mus 
void Mixture_Model::M_step_mus()
{
  for(int g = 0; g < G; g++){
    arma::rowvec inter_mus = arma::rowvec(p,arma::fill::zeros);
    double denom = n_gs.at(g); 
    for(int i = 0; i < n; i++)
    { 
      arma::rowvec inter_val = data.row(i)/denom;
      inter_mus += zi_gs.at(i,g)*inter_val;
    }
    mus[g] = inter_mus; 
  }
}

// within cluster scattering matrix calculation every M step 
void Mixture_Model::M_step_Ws()
{
  // loop through groups
  for(int g = 0; g < G; g++)
  {
    // reset scatter matrix for group g
    Ws[g] = arma::mat(p,p,arma::fill::zeros);
    // go through observations 
    for(int i = 0; i < n; i++)
    { 
      // grab current posterori
      double zi_g_current = zi_gs.at(i,g);
      // xi diff mu
      arma::rowvec xm_i = (data.row(i) - mus[g]); 
      // transpose and multiply
      arma::mat XMXM = xm_i.t()*xm_i;
      // sum into current g
      Ws[g] += zi_g_current*XMXM; 
    }
    // divide by n_gs as per the cov.wt function in R
    Ws[g] = Ws[g]/n_gs[g];
  }  
}


// calculates a mh for a specific x , mu,and sig. 
double Mixture_Model::mahalanobis(arma::rowvec x, arma::rowvec mu, arma::mat inv_sig)
{
  double mh = 0.0; 
  
  arma::rowvec xm = (x - mu);
  arma::mat inter_m = xm*inv_sig; 
  arma::rowvec inter_x = inter_m*xm.t(); 
  mh = inter_x[0]; 

  return mh; 
}

// calculates log density for a single x 
double Mixture_Model::log_density(arma::rowvec x, arma::rowvec mu, arma::mat inv_Sig, double log_det)
{
  // intialize log-density val 
  double ld_val;

  // Set constants of proportions
  double constants = log(2.0) + log(M_PI);
  double constant_prop = -(p*0.5)*constants;

  // append constants
  constant_prop += -(0.5*log_det); 

  // calculate the mahalanobis term 
  double mh_term = -0.5*mahalanobis(x,mu,inv_Sig); 
  
  // add all terms and return 
  ld_val = constant_prop + mh_term; 

  return ld_val; 
}


double Mixture_Model::calculate_log_liklihood(void)
{
  // n x G matrix for holding densities 

  arma::vec row_sums = arma::vec(n,arma::fill::zeros); 
  double log_lik = 0.0;
  double row_sum_term = 0.0; 

  for(int i = 0; i < n; i++)
  {
    row_sum_term = 0.0; 
    for(int g = 0; g < G; g++)
    {
      row_sum_term += pi_gs[g]*std::exp(log_density(data.row(i),mus[g],inv_sigs[g],log_dets[g]));
    }
    row_sum_term = std::log(row_sum_term); 
    log_lik += row_sum_term; 
  }

  return log_lik; 
}

// GENERAL E - Step for all famalies  
void Mixture_Model::E_step()
{
  // set up inter_mediate step for z_igs. 
  arma::mat inter_zigs = arma::mat(n,G,arma::fill::zeros); 

  // intermediate values 
  arma::rowvec inter_density = arma::rowvec(G,arma::fill::zeros); 
  double inter_row_sum; 

  // calculate density proportions 
  for(int i = 0; i < n; i++) 
  { 
    // clear row sum for every observation 
    inter_row_sum = 0;
    inter_density = arma::rowvec(G,arma::fill::zeros); 

    for(int g = 0; g < G; g ++)
    {
      // numerator in e step term for mixture models
      
      inter_density[g] = std::pow((pi_gs[g])*std::exp(log_density(data.row(i),mus[g], inv_sigs[g],log_dets[g])),nu);
      inter_row_sum += inter_density[g]; 
    }
      // after calculating inter_density assign the row to the z_ig matrix. 
    for(int g = 0; g < G; g ++)
    {
      
      double numer_g = inter_row_sum - inter_density[g]; 
      double denom_g = inter_density[g]; 

      inter_zigs.at(i,g) = 1.0/(1 + numer_g/denom_g);

    }

    double ss = arma::sum(inter_zigs.row(i));
    
    if(isnan(ss)){
      inter_zigs.row(i) = zi_gs.row(i);
      ss = arma::sum(inter_zigs.row(i));
    }

    // make sure that it adds up to one
    int count = 0;
    while(true) {

      if(comparison_gp(ss,1.0)){ 
        break; 
      } 
      double push_sum = 0.0; 

      for(int gv = 0; gv < (G-1); gv++){
        push_sum += inter_zigs.row(i)[gv]; 
      }
      
      inter_zigs.row(i)[G-1] = 1.0 - push_sum; 
      ss = inter_zigs.row(i)[G-1] + push_sum;


      if(count == 10){
        
        inter_zigs.row(i) = zi_gs.row(i); // reset to last a posterori
        // loggy(zi_gs.row(i) space  i );
        break; 
      }
      count ++ ;
    }

  }

  zi_gs = inter_zigs; 
}

// Debug functions
// sets each groups covariance to identity. 
void Mixture_Model::sig_eye_init() {
  arma::mat inter_eye =  arma::eye(p,p);
  for(int g = 0; g < G; g++){
    sigs[g] = inter_eye;
    inv_sigs[g] = inter_eye;
  }
}

// loglik tracking 
void Mixture_Model::track_lg_init(void)
{
  // get log_densities and set the first one This is a simple function and should be done after the first intialization
  //arma::rowvec model_lgs = log_densities(); 
  logliks[0] = calculate_log_liklihood();  //sum(model_lgs);
}

// This function keeps track of the log liklihood. You have to calculate the log densities, then keep track of their progress. 
bool Mixture_Model::track_lg(bool check)
{
  
  if (check) {
    logliks.push_back(calculate_log_liklihood()); 
    return false; 
  }
  else {
    logliks.push_back(calculate_log_liklihood());

    //checking aitkens convergence criterion 
    int last_index = logliks.size()-1;
    double l_p1 =  logliks[last_index];
    double l_t =  logliks[last_index-1];
    double l_m1 = logliks[last_index-2];
    double a_t = (l_p1 - l_t)/(l_t - l_m1);
    double l_Inf = l_t + (l_p1 - l_t)/(1.0-a_t);
    double val = std::abs((l_Inf - l_t));
    return (bool)(val < tol_l); 
  }
}

// ********************** END OF GENERAL MIXTURE MODEL CLASS ****************************************************************************************   
  

// BEGINING OF FAMILY CLASSES AND SPECIFIED MODELS 
// SPHERICAL FAMILY 
arma::mat Spherical_Family::lambda_sphere(arma::mat in_W,double in_n)
{
  double lambda = arma::trace(in_W)/(in_n*p); 
  return lambda*eye_I; 
}
// EII
void EII::m_step_sigs()
{

  // calculate full W matrix
  arma::mat W = arma::mat(p,p,arma::fill::zeros); 
  for(int g = 0; g < G; g++){
    W += Ws[g]*n_gs[g];
  }

  // same covariance function for all groups
  arma::mat inter_mat = lambda_sphere(W,n);
  arma::mat inter_inv_mat = arma::solve(inter_mat,EYE,arma::solve_opts::refine); 
  // set all values for everyting 
  for(int g = 0; g < G; g++)
  {
    sigs[g] = inter_mat;
    inv_sigs[g] = inter_inv_mat;
    log_dets[g] = p*log(arma::trace(W)/((double)(n*p))); 
  }
}

// VII MODEL 
void VII::m_step_sigs()
{ 
  for(int g = 0; g < G; g++)
  {
    // for each calculate lambda_sphere
    arma::mat inter_mat = lambda_sphere(Ws[g],1.0);
    sigs[g] = inter_mat;
    inv_sigs[g] = arma::solve(inter_mat,EYE,arma::solve_opts::refine); 
    log_dets[g] = p*log(arma::trace(Ws[g])/(p)); 
  }
}


// DIAGONAL FAMILY 
// EEI MODEL 
void EEI::m_step_sigs(void)
{
  arma::mat S = arma::mat(p,p,arma::fill::eye); 
  // Set up identity in the B matrix .
  arma::mat B = arma::mat(p,p,arma::fill::eye);

  // Get full scatter matrix 
  arma::mat W = arma::mat(p,p,arma::fill::zeros); 
  for(int g = 0; g < G; g++){
    W += Ws[g]*n_gs[g];
  }

  // set B to be the diagonal of W 
  B.diag() = W.diag();
  // expression simplies down to just this. 
  S = B/n; 
  arma::mat Sinv = arma::solve(S,EYE,arma::solve_opts::refine); 

  for(int g = 0; g < G; g++) {
    sigs[g] = S;
    inv_sigs[g] = Sinv; 
    log_dets[g] = arma::sum(arma::log(S.diag())); 
  }

}


// VEI MODEL 
void VEI::set_m_iterations(int in_iter, double in_tol)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
}

void VEI::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
} 


void VEI::m_step_sigs(void)
{

  // ****  INTIALIZATION STEP BEFORE ITERATIVE METHOD ****
  // Some of the variable models, we need to solve for the sigma matrix iteratively,
  // there is a loop within the EM loop for some of the models, this is one of them. 

  // intialize empty matrix for place holder
  arma::mat S = arma::mat(p,p,arma::fill::eye); 
  // Set up identity in the B matrix .
  arma::mat B = arma::mat(p,p,arma::fill::eye);

  // set up lambdas 
  arma::rowvec lambdas = arma::rowvec(G,arma::fill::zeros);
  // calculate lambdas
  for (int g = 0; g < G; g++)
  {
    lambdas[g] = arma::trace(Ws[g])/(p);
  }


  // Get full scatter matrix 
  arma::mat W = arma::mat(p,p,arma::fill::zeros); 
  for(int g = 0; g < G; g++){
    W += Ws[g]*((double)(n_gs[g]/lambdas[g])/(n));
  }

  // set B to be the diagonal of W 
  arma::mat inter_W = arma::diagmat(W); 
  double denom = pow( arma::det(inter_W), (double)(1.0/p) );

  B = inter_W / denom;
  arma::mat invB = arma::solve(B,EYE,arma::solve_opts::refine);

  // calculate new lambda 
  for (int g = 0; g < G; g ++ ){
    lambdas[g] =  arma::trace( Ws[g]*invB )/(p);
  }

  // calculate tolerance, and iterative method check 
  double first_val = 0.0;
  for(int g = 0; g < G; g ++){
    first_val += pi_gs[g]*(1 + log(lambdas[g]));
  }
  first_val = first_val*p; 

  // intialize convergence ary for tolerance check, make one of them infinity. 
  double conv_ary [2] = {first_val,std::numeric_limits<double>::infinity()};

  // **** END OF INITIALIZATION STEP *******
  
  int iter = 1; 
  // iterate while its less then m_iter_max, but need to add convergence check from mixture... 
  while ( (iter < m_iter_max) &&  std::abs(conv_ary[1] - conv_ary[0]) > m_tol )
  {

    // Get the full flurry matrix 
    W = arma::mat(p,p,arma::fill::zeros); 
    for(int g = 0; g < G; g++){
      W += Ws[g]*((double)(n_gs[g]/lambdas[g])/(n));
    }
    
    // get diagonal and denominator for calculating B
    inter_W = arma::diagmat(W); 
    denom = pow( arma::det(inter_W), (double)(1.0/p) );
    B = inter_W / denom;
    invB = arma::solve(B,EYE,arma::solve_opts::refine);

    // calculate new lambda 
    for (int g = 0; g < G; g ++ ){
      lambdas[g] =  arma::trace(Ws[g]*invB )/(p);
    }
    
    // calculate tolerance, and iterative method check 
    first_val = 0.0;
    for(int g = 0; g < G; g ++){
      first_val += pi_gs[g]*(1 + std::log(lambdas[g]));
    }
    first_val = first_val*p; 

    conv_ary[1]  = conv_ary[0]; 
    conv_ary[0] = first_val; 

    iter++;
  }
  
  for(int g = 0; g < G; g++){
    S = lambdas[g]*B;
    sigs[g] = S; 
    inv_sigs[g] = arma::solve(S,EYE,arma::solve_opts::refine); 
    log_dets[g] =  p*std::log(lambdas[g]); 
  }


}


// EVI
void EVI::m_step_sigs(void){

  // intialize Bk_s and denoms 
  std::vector<arma::mat> Bk_s;
  std::vector<double> denoms; 
  Bk_s.assign(G,arma::mat(p,p,arma::fill::eye));
  denoms.assign(G,0.0);
  

  double lambda = 0;
  // loop through and only compute the BK_s and and assign denominators 
  for(int g = 0; g < G; g++)
  {
    arma::mat D_g = arma::diagmat(Ws[g])*n_gs[g];
    double power = (double)(1.0/p);
    denoms[g] = pow(arma::det(D_g),power);
    Bk_s[g] = D_g/denoms[g];
    lambda += denoms[g];    
  }
  lambda = (double)(lambda)/n; 
  // assign new Sig
  
  for(int g = 0; g < G; g++)
  {
    arma::mat S = lambda*Bk_s[g];
    sigs[g] = S;
    inv_sigs[g] = arma::solve(S,EYE,arma::solve_opts::refine);
    log_dets[g] =  p*std::log(lambda); 
  }

}

// VVI 
void VVI::m_step_sigs(void)
{

  arma::mat S;
  for(int g = 0; g < G; g++){
    S = arma::diagmat(Ws[g]);
    sigs[g] = S;
    inv_sigs[g] = arma::solve(S,EYE);
    log_dets[g] = arma::sum(arma::log(S.diag()));  
  }

}; 


// EEE
void EEE::m_step_sigs(void)
{

  // calculate full W
  arma::mat W = arma::mat(p,p,arma::fill::zeros); 
  for(int g = 0; g < G; g++){
    W += Ws[g]*(double)(n_gs[g]/n);   
  }

  // invert it and set it to all 
  double log_det_W = arma::log_det(W).real(); 
  arma::mat invW = arma::solve(W,EYE,arma::solve_opts::refine);    
  for(int g = 0; g < G; g++){
    sigs[g] = W;
    inv_sigs[g] = invW; 
    log_dets[g] = log_det_W; 
  }
}

// VVV
void VVV::m_step_sigs(void)
{
  // Straight forward. 
  for(int g = 0; g < G; g++)
  {
    sigs[g] = Ws[g]; 
    inv_sigs[g] = arma::solve(Ws[g],EYE,arma::solve_opts::refine);
    log_dets[g] = arma::log_det(Ws[g]).real(); 
  }
}

// EEV
void EEV::m_step_sigs(void)
{
  std::vector<arma::mat> Ls_g(G); 
  std::vector<arma::mat> Omega_g(G);
  std::vector<arma::colvec> eigens(G);

  // exception thrower 
  sym_matrix_error e; 

  for(int g = 0; g < G; g++)
  {
    Ls_g[g] = arma::mat(p,p,arma::fill::zeros);
    Omega_g[g] = arma::mat(p,p,arma::fill::zeros);
    eigens[g] = arma::colvec(p,arma::fill::zeros);
  }

  arma::mat A = arma::mat(p,p,arma::fill::zeros);

  for(int g = 0; g < G; g ++)
  {
      if(!((Ws[g]*n_gs[g]).is_sympd())){
        throw e; 
      }
    arma::eig_sym(eigens[g], Ls_g[g], Ws[g]*n_gs[g],"std");
    Omega_g[g].diag() = eigens[g];
    A += Omega_g[g];
  }

  for(int g = 0; g < G; g++)
  {
    arma::mat S = (Ls_g[g]*A*Ls_g[g].t())*((double)(1.0/n));
    sigs[g] = S;
    inv_sigs[g] = arma::solve(S,EYE,arma::solve_opts::refine); 
    log_dets[g] = arma::log_det(S).real(); 
  }
}

// VEV
void VEV::set_m_iterations(int in_iter, double in_tol)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
}

void VEV::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
} 


void VEV::m_step_sigs(void)
{

  // exception thrower 
  sym_matrix_error e; 

  // ************** INITIALIZATION STEP **************
  std::vector<arma::mat> Ls_g(G); 
  std::vector<arma::mat> Omega_g(G);
  std::vector<arma::colvec> eigens(G);
  std::vector<double> lambdas(G); 

  for(int g = 0; g < G; g++)
  {
    Ls_g[g] = arma::mat(p,p,arma::fill::zeros);
    Omega_g[g] = arma::mat(p,p,arma::fill::zeros);
    eigens[g] = arma::colvec(p,arma::fill::zeros);
  }

  arma::mat A = arma::mat(p,p,arma::fill::zeros);

  for(int g = 0; g < G; g ++)
  {
    if(!( (Ws[g]*pi_gs[g]).is_sympd() ))
    {
      throw e; 
    }
    arma::eig_sym(eigens[g], Ls_g[g], (Ws[g]*pi_gs[g]),"std");
    Omega_g[g].diag() = eigens[g];
  }

  // calculate lambda 
  for(int g = 0; g < G; g++ )
  {
    // first lambda step 
    lambdas[g] = arma::trace(Omega_g[g])/((double)(pi_gs[g]*p)); 
  }

  // calculate A
  for(int g = 0; g < G;  g++)
  {
    A += Omega_g[g]/lambdas[g];
  }  
  double denom = pow(arma::det(A),(double)(1.0/p));
  A = arma::diagmat(A/denom);   
  arma::mat invA = arma::solve(A,EYE,arma::solve_opts::refine);

  for(int g = 0; g < G; g++ )
  {
    lambdas[g] = arma::trace(Omega_g[g]*invA)/((double)(pi_gs[g]*p));
  }

  // calculate log lik for this portion only. 
  double first_val = 0.0; 
  for(int g = 0; g < G; g++){
    first_val +=  (1 + log(lambdas[g]))*pi_gs[g]*((double)p);
  }

 // intialize convergence ary for tolerance check, make one of them infinity. 
  double conv_ary [2] = {first_val,std::numeric_limits<double>::infinity()};

  // ************* END OF INITIALIZATION STEP *******
  int iter = 1;
   // iterate while its less then m_iter_max, but need to add convergence check from mixture... 
  while ( (iter < m_iter_max) &&  abs(conv_ary[1] - conv_ary[0]) > m_tol )
  {
    A = arma::mat(p,p,arma::fill::zeros);
    // calculate A
    for(int g = 0; g < G;  g++)
    {
      A += Omega_g[g]/lambdas[g];
    }  
    double denom = pow(arma::det(A),(double)(1.0/p));
    A = arma::diagmat(A/denom);   
    arma::mat invA = arma::solve(A,EYE,arma::solve_opts::refine);

    for(int g = 0; g < G; g++ )
    {
      lambdas[g] = arma::trace(Omega_g[g]*invA)/((double)(pi_gs[g]*p));
    }

    first_val = 0.0;   
    for(int g = 0; g < G; g++){
      first_val +=  (1 + log(lambdas[g]))*pi_gs[g]*((double)p);
    }

    conv_ary[1] = conv_ary[0];
    conv_ary[0] = first_val; 
    iter++; 
  }

  arma::mat S;
  for(int g = 0; g < G; g++)
  {
    S = lambdas[g]*((Ls_g[g]) * A * Ls_g[g].t());
    sigs[g] = S; 
    inv_sigs[g] = arma::solve(S,EYE,arma::solve_opts::refine); 
    log_dets[g] = p*std::log(lambdas[g]);
  }

}


// EVV
void EVV::m_step_sigs(void)
{
  std::vector<arma::mat> C_g(G); 
  double lambda = 0.0;

  for(int g = 0; g < G; g++){

    arma::mat inter_C = Ws[g]*n_gs[g];
    double denom = pow( arma::det(inter_C), ((double)((1.0)/p)));

    C_g[g] = inter_C/denom; 
    lambda += denom/n; 
  }

  for(int g = 0; g < G; g++){
    arma::mat S = lambda*C_g[g]; 
    sigs[g] = S; 
    inv_sigs[g] = arma::solve(S,EYE,arma::solve_opts::refine); 
    log_dets[g] = p*std::log(lambda); 
  }

}


// VEE
void VEE::set_m_iterations(int in_iter, double in_tol)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
}

void VEE::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
} 

void VEE::m_step_sigs(void)
{
  // ****** BEGINNING OF INITIALIZATION ******

  std::vector<arma::mat> W_k(G); 
  arma::mat W = arma::mat(p,p,arma::fill::zeros); 

  for(int g = 0; g < G; g++)
  {
    W_k[g] = Ws[g]*pi_gs[g];
    W += W_k[g];
  }

  double denom = pow(arma::det(W),((double)((1.0)/p))); 
  arma::mat C = W/denom; 
  arma::mat invC = arma::solve(C,EYE,arma::solve_opts::refine); 

 // calculate new lambdas
  std::vector<double> lambdas(G); 
  for (int g = 0; g < G; g ++ ){
    lambdas[g] =  arma::trace( Ws[g]*invC )/(p);
  }

  // calculate log lik for this portion only. 
  double first_val = 0.0; 
  for(int g = 0; g < G; g++){
    first_val +=  ( (arma::trace( Ws[g]*invC ))/(lambdas[g]) + p*(pi_gs[g]*log(lambdas[g])));
  }

 // intialize convergence ary for tolerance check, make one of them infinity. 
  double conv_ary [2] = {first_val,std::numeric_limits<double>::infinity()};

  // **** END OF INITIALIZATION STEP *******
  
  int iter = 1; 
  // iterate while its less then m_iter_max, but need to add convergence check from mixture... 
  while ( (iter < m_iter_max) &&  (abs(conv_ary[1] - conv_ary[0])) > m_tol )
  {
    C = arma::mat(p,p,arma::fill::zeros); 
    for(int g = 0; g < G; g++)
    {
      C += W_k[g]/(lambdas[g]);
    }
    denom = pow(arma::det(C), (double)(1.0/p)); 
    C = C/denom; 
    invC =  arma::solve(C,EYE,arma::solve_opts::refine); 

    for (int g = 0; g < G; g ++ )
    {
    lambdas[g] =  arma::trace( Ws[g]*invC )/(p);
    }

    first_val = 0.0; 
    for(int g = 0; g < G; g++)
    {
      first_val +=  ( (arma::trace( Ws[g]*invC ))/(lambdas[g]) + p*(pi_gs[g]*std::log(lambdas[g])));
    }

    conv_ary[1] = conv_ary[0]; 
    conv_ary[0] = first_val; 
    iter++; 
  }

  arma::mat S = arma::mat(p,p,arma::fill::zeros); 
  for(int g = 0; g < G; g++){
    S = lambdas[g]*C;
    sigs[g] = S; 
    inv_sigs[g] = arma::solve(S,EYE,arma::solve_opts::refine); 
    log_dets[g] =  p*std::log(lambdas[g]);
  }

}


// RYAN BROWNE ET MCNICHOLAS 2014 
// EVE 

void EVE::set_m_iterations(int in_iter, double in_tol, arma::mat in_D)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
    D = in_D; 
}

void EVE::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
  D = arma::eye(p,p); 
} 



void EVE::m_step_sigs(void){

  // exception thrower 
  sym_matrix_error e; 

  // calculate full Flurry matrix
  arma::mat W = arma::mat(p,p,arma::fill::zeros);
  for(int g =0; g < G; g++)
  {
    W += Ws[g]*pi_gs[g];
  }

  // get eigen values and noramlize them, store them in column vectors. 
  std::vector<arma::colvec> A_gs(G);
  
  // go through groups 
  for(int g = 0; g < G; g++)
  {
    // eigen value calc
    arma::mat tempA_g = D.t() * Ws[g] * D * pi_gs[g];
    A_gs[g] = tempA_g.diag().as_col();
    // calculate denominator for normalization. 
    double denom = pow(arma::prod(A_gs[g]), ((double)((1.0)/((A_gs[g].size())))) );
    A_gs[g] = A_gs[g]/denom;
  }

  // ******** BEGIN INTIALIZATION of MM *************** 

  // MM pt. 1
  arma::mat interZ = arma::mat(p,p,arma::fill::zeros); // holds interZ matrix. 
  double lambda_g; // maximum eigen value placeholder
  arma::colvec eigens; // eigen values placeholder
  arma::mat L_g; // eigen vectors
  arma::mat ADK; // left hand side multiplicaiton matrix placeholder
  std::vector<arma::mat> W_temp_g(G);

  for(int g = 0; g < G; g++)
  {
    //arma::eig_sym
    W_temp_g[g] = Ws[g]*pi_gs[g];
      if(!( (W_temp_g[g]).is_sympd() ))
    {
      throw e;  
    }
    arma::eig_sym(eigens, L_g, W_temp_g[g],"std");
    lambda_g = arma::max(eigens); 
    ADK = arma::diagmat(1.0/A_gs[g]) * D.t();
    interZ += (ADK * W_temp_g[g]) - (lambda_g *ADK);  
  }

  // svd calculation 
  arma::mat U; 
  arma::mat V; 
  arma::vec s; 
  arma::svd(U,s,V,interZ,"std"); 
  D = V * U.t(); 

  // reset 
  interZ = arma::mat(p,p,arma::fill::zeros); 

  // MM pt 2. 
  for(int g = 0; g < G; g++)
  {
    lambda_g = arma::max(1.0/A_gs[g]);
    interZ +=  W_temp_g[g] * D * arma::diagmat(1.0/A_gs[g]) - lambda_g * (W_temp_g[g] * D); 
  }

  // calculate svd and set new D. 
  arma::svd(U,s,V,interZ,"std" ); 
  D = V * U.t(); 
  D = D.t(); 

  double first_val = 0.0; 

  // checking convergence
  arma::vec tempz = arma::vec(G,arma::fill::zeros); 
  for(int g = 0; g < G; g++)
  {
    arma::mat DWA =  (D.t() * W_temp_g[g] * D * arma::diagmat( 1.0 / A_gs[g])) ; 
    tempz[g] = arma::sum(DWA.diag());
  }

  first_val = arma::sum(tempz);

  // *********** END OF INTIALIZATION FOR MM ************

   // intialize convergence ary for tolerance check, make one of them infinity. 
  double conv_ary [2] = {first_val,std::numeric_limits<double>::infinity()};

  int iter = 1; 
  // iterate while its less then m_iter_max, but need to add convergence check from mixture... 
  while ( (iter < m_iter_max) &&  (abs(conv_ary[1] - conv_ary[0])) > m_tol )
  {
    
    // calculate new D 
    
    // MM pt. 1
    arma::mat interZ = arma::mat(p,p,arma::fill::zeros); // holds interZ matrix. 
    double lambda_g; // maximum eigen value placeholder
    arma::colvec eigens; // eigen values placeholder
    arma::mat L_g; // eigen vectors
    arma::mat ADK; // left hand side multiplicaiton matrix placeholder
    std::vector<arma::mat> W_temp_g(G);


    for(int g = 0; g < G; g++)
    {
      //arma::eig_sym
      W_temp_g[g] = Ws[g]*pi_gs[g];
      if(!(W_temp_g[g]).is_sympd())
      {
        throw e; 
      }
      arma::eig_sym(eigens, L_g, W_temp_g[g],"std");
      lambda_g = arma::max(eigens); 
      ADK = arma::diagmat(1.0/A_gs[g]) * D.t();
      interZ += (ADK * W_temp_g[g]) - (lambda_g *ADK);  
    }

    // svd calculation 
    arma::mat U; 
    arma::mat V; 
    arma::vec s; 
    arma::svd(U,s,V,interZ,"std"); 
    D = V * U.t(); 

    // reset 
    interZ = arma::mat(p,p,arma::fill::zeros); 

    // MM pt 2. 
    for(int g = 0; g < G; g++)
    {
      lambda_g = arma::max(1.0/A_gs[g]);
      interZ +=  W_temp_g[g] * D * arma::diagmat(1.0/A_gs[g]) - lambda_g * (W_temp_g[g] * D); 
    }

    // calculate svd and set new D. 
    arma::svd(U,s,V,interZ,"std"); 
    D = V * U.t(); 
    D = D.t(); 

    // CALCULATE NEW A_gs 
    for(int g = 0; g < G; g++)
    {
      // eigen value calc
      arma::mat tempA_g = D.t() * W_temp_g[g]* D ; 
      A_gs[g] = tempA_g.diag().as_col();
      // calculate denominator for normalization. 
      double denom = pow(arma::prod(A_gs[g]), ((double)((1.0)/((A_gs[g].size())))) );
      A_gs[g] = A_gs[g]/denom;
    }


    // check convergence 
    for(int g = 0; g < G; g++)
    {
    arma::mat DWA =  (D.t() * W_temp_g[g] * D * arma::diagmat( 1.0 / A_gs[g])) ; 
    tempz[g] = arma::sum(DWA.diag());
    }
    first_val = arma::sum(tempz);

    conv_ary[1] = conv_ary[0]; 
    conv_ary[0] = first_val; 

    iter++; 

  }

  double lam = 0.0; // volume 
  for(int g = 0; g < G; g++)
  {
    arma::mat DAW = D * arma::diagmat(1/A_gs[g]) * D.t() * W_temp_g[g];
    lam +=  arma::sum(DAW.diag()); 
  }
  lam = lam/p; 

  for(int g = 0; g < G; g++)
  {
    sigs[g] = D * arma::diagmat(lam*A_gs[g]) * D.t();
    inv_sigs[g] = D * arma::diagmat((1.0/lam) * (1.0/A_gs[g]))* D.t(); 
    log_dets[g] = p*std::log(lam);  
    }  

}




// RYAN BROWNE ET MCNICHOLAS 2014 
// EVE 

void VVE::set_m_iterations(int in_iter, double in_tol, arma::mat in_D)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
    D = in_D; 
}

void VVE::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
  D = arma::eye(p,p); 
} 



void VVE::m_step_sigs(void){

  // exception thrower 
  sym_matrix_error e; 

  // calculate full Flurry matrix
  arma::mat W = arma::mat(p,p,arma::fill::zeros);
  for(int g =0; g < G; g++)
  {
    W += Ws[g]*pi_gs[g];
  }

  // get eigen values and noramlize them, store them in column vectors. 
  std::vector<arma::colvec> A_gs(G);
  // go through groups 
  for(int g = 0; g < G; g++)
  {
    // eigen value calc
    arma::mat tempA_g = D.t() * Ws[g] * D * pi_gs[g];
    A_gs[g] = tempA_g.diag().as_col();
    // calculate denominator for normalization. 
    double denom = pow(arma::prod(A_gs[g]), ((double)((1.0)/((A_gs[g].size())))) );
    A_gs[g] = A_gs[g]/denom;
  }

  // ******** BEGIN INTIALIZATION of MM *************** 

  // MM pt. 1
  arma::mat interZ = arma::mat(p,p,arma::fill::zeros); // holds interZ matrix. 
  double lambda_g; // maximum eigen value placeholder
  arma::colvec eigens; // eigen values placeholder
  arma::mat L_g; // eigen vectors
  arma::mat ADK; // left hand side multiplicaiton matrix placeholder
  std::vector<arma::mat> W_temp_g(G);

  for(int g = 0; g < G; g++)
  {
    //arma::eig_sym
    W_temp_g[g] = Ws[g]*pi_gs[g];
    if(!(W_temp_g[g]).is_sympd())
      {
        throw e;
      }
    arma::eig_sym(eigens, L_g, W_temp_g[g],"std");
    lambda_g = arma::max(eigens); 
    ADK = arma::diagmat(1.0/A_gs[g]) * D.t();
    interZ += (ADK * W_temp_g[g]) - (lambda_g *ADK);  
  }

  // svd calculation 
  arma::mat U; 
  arma::mat V; 
  arma::vec s; 
  arma::svd(U,s,V,interZ,"std"); 
  D = V * U.t(); 

  // reset 
  interZ = arma::mat(p,p,arma::fill::zeros); 

  // MM pt 2. 
  for(int g = 0; g < G; g++)
  {
    lambda_g = arma::max(1.0/A_gs[g]);
    interZ +=  W_temp_g[g] * D * arma::diagmat(1.0/A_gs[g]) - lambda_g * (W_temp_g[g] * D); 
  }

  // calculate svd and set new D. 
  arma::svd(U,s,V,interZ,"std"); 
  D = V * U.t(); 
  D = D.t(); 

  double first_val = 0.0; 

  // checking convergence
  arma::vec tempz = arma::vec(G,arma::fill::zeros); 
  for(int g = 0; g < G; g++)
  {
    arma::mat DWA =  (D.t() * W_temp_g[g] * D * arma::diagmat( 1.0 / A_gs[g])) ; 
    tempz[g] = arma::sum(DWA.diag());
  }

  first_val = arma::sum(tempz);

  // *********** END OF INTIALIZATION FOR MM ************

   // intialize convergence ary for tolerance check, make one of them infinity. 
  double conv_ary [2] = {first_val,std::numeric_limits<double>::infinity()};

  int iter = 1; 
  // iterate while its less then m_iter_max, but need to add convergence check from mixture... 
  while ( (iter < m_iter_max) &&  (abs(conv_ary[1] - conv_ary[0])) > m_tol )
  {


    // CALCULATE NEW A_gs 
    for(int g = 0; g < G; g++)
    {
      // eigen value calc
      arma::mat tempA_g = D.t() * W_temp_g[g]* D ; 
      A_gs[g] = tempA_g.diag().as_col();
      // calculate denominator for normalization. 
      double denom = pow(arma::prod(A_gs[g]), ((double)((1.0)/((A_gs[g].size())))) );
      A_gs[g] = A_gs[g]/denom;
    }
    // calculate new D 
    
    // MM pt. 1
    arma::mat interZ = arma::mat(p,p,arma::fill::zeros); // holds interZ matrix. 
    double lambda_g; // maximum eigen value placeholder
    arma::colvec eigens; // eigen values placeholder
    arma::mat L_g; // eigen vectors
    arma::mat ADK; // left hand side multiplicaiton matrix placeholder
    std::vector<arma::mat> W_temp_g(G);


    for(int g = 0; g < G; g++)
    {
      //arma::eig_sym
      W_temp_g[g] = Ws[g]*pi_gs[g];
      if(!(W_temp_g[g].is_sympd())){
        throw e; 
      }
      arma::eig_sym(eigens, L_g, W_temp_g[g],"std");
      lambda_g = arma::max(eigens); 
      ADK = arma::diagmat(1.0/A_gs[g]) * D.t();
      interZ += (ADK * W_temp_g[g]) - (lambda_g *ADK);  
    }

    // svd calculation 
    arma::mat U; 
    arma::mat V; 
    arma::vec s; 
    arma::svd(U,s,V,interZ,"std"); 
    D = V * U.t(); 

    // reset 
    interZ = arma::mat(p,p,arma::fill::zeros); 

    // MM pt 2. 
    for(int g = 0; g < G; g++)
    {
      lambda_g = arma::max(1.0/A_gs[g]);
      interZ +=  W_temp_g[g] * D * arma::diagmat(1.0/A_gs[g]) - lambda_g * (W_temp_g[g] * D); 
    }

    // calculate svd and set new D. 
    arma::svd(U,s,V,interZ,"std"); 
    D = V * U.t(); 
    D = D.t(); 

    // check convergence 
    for(int g = 0; g < G; g++)
    {
    arma::mat DWA =  (D.t() * W_temp_g[g] * D * arma::diagmat( 1.0 / A_gs[g])) ; 
    tempz[g] = arma::sum(DWA.diag());
    }
    first_val = arma::sum(tempz);

    conv_ary[1] = conv_ary[0]; 
    conv_ary[0] = first_val; 

    iter++; 

  }


  arma::vec lam = arma::vec(G,arma::fill::zeros); // volume 
  for(int g = 0; g < G; g++)
  {
    arma::mat DAW = D * arma::diagmat(1/A_gs[g]) * D.t() * Ws[g];
    lam[g] =  arma::sum(DAW.diag())/p; 
  }


  for(int g = 0; g < G; g++)
  {
    sigs[g] = D * arma::diagmat(lam[g]*A_gs[g]) * D.t();
    inv_sigs[g] = D * arma::diagmat((1.0/lam[g]) * (1.0/A_gs[g]))* D.t(); 
    log_dets[g] = arma::log_det(sigs[g]).real(); 
  }  

}



// MISSING DATA METHODS FOR GENERAL Mixture_Model class. 
// grabs missing values and their respective tags and row_tags. 
void Mixture_Model::init_missing_tags(void)
{

  std::vector<arma::uvec> in_missing_tags; // create the missing tags vector 
  arma::uvec in_row_tags; // set up row tags.  

  // loop through rows. 
  for(int i = 0; i < n;  i++ )
  {
    // get the current nantags. 
    arma::uvec nan_tags = arma::find_nonfinite(data.row(i));

    if(nan_tags.n_elem > 0)
    {
      // get row id uvec. 
      arma::uvec row_id = arma::uvec(1); // init
      row_id[0] = i; // set entry 
                    
      // concatonate both row uvec and the nan_tags
      arma::uvec mis_tag_i = arma::join_cols(row_id,nan_tags); 
      in_row_tags = arma::join_cols(in_row_tags,row_id); 

      // add to missing tags list
      in_missing_tags.push_back(mis_tag_i); 
    } 
  }
  // assign row tags and missing tags. 
  row_tags = in_row_tags; 
  missing_tags = in_missing_tags; 
}



// EM BURN_in METHOD 
// takes in number of steps to run the EM algorithm WITHOUT imputation of missing data. 
// this will initialize the model for better imputation 
void Mixture_Model::E_step_only_burn(void)
{

  // impute_cond_mean using z_igs. 
  impute_cond_mean(); 
  E_step(); // then perform E_step on the entire dataset. 
  impute_cond_mean(); // impute conditional mean again. 
  E_step(); // e_step again. 
  impute_cond_mean(); // one more for good luck 
  E_step(); // e_step. Not that every E_step, the parmaeters dont change but the imputation does. 

}





// EM BURN_in METHOD 
// takes in number of steps to run the EM algorithm WITHOUT imputation of missing data. 
// this will initialize the model for better imputation 
void Mixture_Model::EM_burn(int in_burn_steps)
{
  // copy dataset and z_igs. 
  arma::mat* orig_data = new arma::mat(n,p); // create empty arma mat on the heap. 
  arma::mat* orig_zi_gs = new arma::mat(n,G); 
  *orig_data = data; // set orig_data. 
  *orig_zi_gs = zi_gs; // set zi_igs. 

  // remove all data, and zi_gs with missing values. 
  data.shed_rows(row_tags); 
  zi_gs.shed_rows(row_tags); 
 
  n = data.n_rows; 
  
  // intialize all parameters. (all methods are on self)
  M_step_props(); 
  M_step_mus();
  M_step_Ws();
  m_step_sigs();
 
  
  // run EM burn in for in_burn_steps number of steps.  
  for(int i = 0; i < in_burn_steps; i++)
  {
    E_step();
    M_step_props(); 
    M_step_mus();
    M_step_Ws();
    m_step_sigs();
  }
  // Now replace back the original data points and zi_igs. only keep the parmaeters.   
  data = *orig_data; 
  zi_gs = *orig_zi_gs; // done EM burn 
  
}


// imputation cond_mean functions. replaces values based on approach. 
void Mixture_Model::impute_cond_mean(void)
{

  // go through each of the tags and select the row out of the dataset 
  for(arma::uword i_tag = 0; i_tag < row_tags.n_elem; i_tag++)
  {
    arma::uvec current_tag = missing_tags[i_tag];// get current full tag. 
    current_tag.shed_row(0); // remove the row tag.  
    arma::uword row_tag = row_tags.at(i_tag); 

     // create the missing column vector with current tag the from the data. 
    arma::mat c_obs = data.row(row_tag).t();
    arma::mat m_obs = data.row(row_tag).t();
    arma::mat nm_obs = data.row(row_tag).t(); // use column vectors for calculations. 

    // drop the indeces that contain NAS.
    nm_obs.shed_rows(current_tag);
    // select only ones that contain NAS.
    m_obs = m_obs.rows(current_tag); 
    
    // now I have to iterate through g groups to compute the conditional mean imputation. 
    // remember to change this for groups G. it doesnt have to be at 2.  
    for(int g = 0; g < G; g++)
    {
      // current mus. 
      arma::mat c_mu_m = mus[g].t(); // invert because column vector. 
      arma::mat c_mu_nm = mus[g].t(); // invert because column vector.
      arma::mat c_sig = sigs[g]; // self explanitory. (current sig)

      c_mu_m = c_mu_m.rows(current_tag); // select missing rows of mu
      c_mu_nm.shed_rows(current_tag); // select non missing rows of mu 

      arma::mat c_sig_m = c_sig; // set c_sigs because I have to shed. 
      arma::mat c_sig_nm = c_sig;

      c_sig_m.shed_cols(current_tag);  // shed rows and select one you need. 
      c_sig_m = c_sig_m.rows(current_tag); 

      c_sig_nm.shed_cols(current_tag); 
      c_sig_nm.shed_rows(current_tag); // make this square, non missing. 

      // finally compute imputation. 
      double z_ig = zi_gs.at(row_tag,g); // CHANGE THIS LATER IN FULL FUNCTION. 

      // MIXTURES OF CONDITIONAL MEAN IMPUTATION 
      arma::vec x_nm_dif = nm_obs - c_mu_nm; 
      int p_nm = c_sig_nm.n_rows; 
      // note to self during debug, check if you are on first becuase of nans and assign conditional mean imputation 
      if(g == 0){
        m_obs = z_ig*( c_mu_m +  c_sig_m * arma::solve(c_sig_nm,arma::eye(p_nm,p_nm),arma::solve_opts::refine) * x_nm_dif); 
      }
      else{
        m_obs += z_ig*( c_mu_m +  c_sig_m * arma::solve(c_sig_nm,arma::eye(p_nm,p_nm),arma::solve_opts::refine) * x_nm_dif); 
      }

    }
    // assign missing values. 
    for(arma::uword m_i = 0; m_i < m_obs.size(); m_i++ )
    {
      data.at(row_tag,current_tag[m_i]) = m_obs.at(m_i); 
    }
  }

}

void Mixture_Model::impute_init(void)
{
  impute_cond_mean(); // after burn you impute as initalization. 
  E_step(); // calculate e step over entire dataset. 
  // run m_step once. 
  M_step_props(); 
  M_step_mus();
  M_step_Ws();
  m_step_sigs();
}



// create model function, generates a model pointer and returns it as a general mixture_model pointer
Mixture_Model* create_model(arma::mat* Xp,int G, int model_id, int model_type)
{
 switch(model_type)
 {
   case 0: {
      EII* m = new EII(Xp, G, model_id);
      return m;
   }
   case 1:{
      VII* m = new VII(Xp, G, model_id);
      return m;
   }
   case 2:{
      EEI* m = new EEI(Xp, G, model_id); 
      return m;
   }
   case 3:{
      EVI* m = new EVI(Xp,G,model_id);
     return m; 
   }
   case 4:{
      VEI* m = new VEI(Xp,G,model_id); 
      return m; 
   }
   case 5:{
      VVI* m = new VVI(Xp,G,model_id); 
      return m; 
   }
   case 6:{
      EEE* m = new EEE(Xp,G,model_id); 
      return m; 
   }
   case 7:{
      VEE* m = new VEE(Xp, G, model_id); 
      return m; 
   }
   case 8:{
      EVE* m = new EVE(Xp,G,model_id); 
      return m; 
   }
   case 9:{
      EEV* m = new EEV(Xp,G,model_id); 
      return m; 
   }
   case 10:{
      VVE* m = new VVE(Xp,G,model_id); 
      return m; 
   }
   case 11:{
      EVV* m = new EVV(Xp,G, model_id); 
      return m; 
   }
   case 12:{
      VEV* m = new VEV(Xp, G, model_id); 
      return m; 
   }
   default:{ 
      VVV* m = new VVV(Xp, G, model_id);
      return m;
   }
 }
}

// WRAPPERS 

// [[Rcpp::export]]
Rcpp::List main_loop(arma::mat X, // data 
                     int G, int model_id, // number of groups and model id (id is for parrallel use)
                     int model_type,  // covariance model type
                     arma::mat in_zigs, // group memberships from initialization 
                     int in_nmax, // iteration max for EM . 
                     double in_l_tol, // liklihood tolerance 
                     int in_m_iter_max, // iteration max for M step for special models 
                     double in_m_tol, // tolerance for matrix convergence on M step for special mdodels.
                     arma::vec anneals,
                     int t_burn = 5// number of burn in steps for NAs if found. 
                     )
{
  
  // create mixture model class. 
  std::unique_ptr<Mixture_Model> m = std::unique_ptr<Mixture_Model>(create_model(&X,G,model_id,model_type));  

  // Intialize 
  m->zi_gs = in_zigs;

  // check for nas. 
  bool NA_check; 
  m->init_missing_tags();
  NA_check = ( m->row_tags.size() > 0); 


  // wrap iterations up in a try catch just in case something bad happens. 
  try
  {
        // perform missing data check and implement algorithm based on check 
    if(NA_check){

      // check to see if model_id is any of the special ones and set defaults, or pass in arguement. 
      switch (model_type)
      {
        case 4:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol);
          }
          else{
            m->set_defaults();
          } 
          break;    
        }
        case 12:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol);
          }
          else{
            m->set_defaults();
          } 
          break; 
        }
        case 7:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol);
          }
          else{
            m->set_defaults();
          } 
          break; 
        }
        case 8:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol,arma::mat(m->p,m->p,arma::fill::eye));
          }
          else{
            m->set_defaults();
          } 
          break; 
        }
        case 10:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol,arma::mat(m->p,m->p,arma::fill::eye));
          }
          else{
            m->set_defaults();
          } 
        }
        default:
          break;
      }
      // phase 1
      m->EM_burn(t_burn); // defaults already set in here, although I should pull them out. in full function. 
      // phase 2. 
      m->impute_init();
      // phase 3.  
      m->track_lg_init(); 
      arma::uword nmax = (arma::uword)in_nmax; 
      bool convergence_check = false; 
      // main EM with extra setp. 
      for(arma::uword iter = 0; iter < nmax ; iter++)
      {
        if(iter < anneals.n_elem)
        {
          m->nu = anneals[iter]; 
        }
        else{
          m->nu = 1.0; 
        }
        m->E_step();
        m->impute_cond_mean(); // now have imputation step. 
        m->M_step_props(); 
        m->M_step_mus();
        m->M_step_Ws();
        m->m_step_sigs();
        convergence_check = m->track_lg(iter < 5);
        if(convergence_check){
          // Rcpp::Rcout  << "Converged at Iteration " << iter << std::endl;  
          break; 
        }
      }

    }
    else{
      // perform intialization of params. 
      m->M_step_props(); 
      m->M_step_mus();
      m->M_step_Ws();

      // check to see if model_id is any of the special ones and set defaults, or pass in arguement. 
      switch (model_type)
      {
        case 4:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol);
          }
          else{
            m->set_defaults();
          } 
          break;    
        }
        case 12:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol);
          }
          else{
            m->set_defaults();
          } 
          break; 
        }
        case 7:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol);
          }
          else{
            m->set_defaults();
          } 
          break; 
        }
        case 8:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol,arma::mat(m->p,m->p,arma::fill::eye));
          }
          else{
            m->set_defaults();
          } 
          break; 
        }
        case 10:{
          if(in_m_iter_max != 0){
            m->set_m_iterations(in_m_iter_max,in_m_tol,arma::mat(m->p,m->p,arma::fill::eye));
          }
          else{
            m->set_defaults();
          } 
        }
        default:
          break;
      }
      
      m->m_step_sigs();
      m->track_lg_init(); 
      arma::uword nmax = (arma::uword)in_nmax; 
      bool convergence_check = false; 

      for(arma::uword iter = 0; iter < nmax ; iter++)
      {
        if(iter < anneals.n_elem)
        {
          m->nu = anneals[iter]; 
        }
        else{
          m->nu = 1.0; 
        }
        m->E_step();
        m->M_step_props(); 
        m->M_step_mus();
        m->M_step_Ws();
        m->m_step_sigs();
        convergence_check = m->track_lg(iter < 5);
        if(convergence_check){
          // Rcpp::Rcout  << "Converged at Iteration " << iter << std::endl;  
          break; 
        }
      }
    }
  }
  catch(const std::exception& e)
  {
    // Rcpp::Rcout  << "C++ Error has occured during iterations. See message below." << '\n'; 
    // Rcpp::Rcout << e.what() << '\n';
    return Rcpp::List::create(Rcpp::Named("Error") = "Iteration Error, decreasing loglik"); 
    Rcpp::List ret_val = Rcpp::List::create(Rcpp::Named("X") = m->data,
                                            Rcpp::Named("mus") = m->mus, 
                                            Rcpp::Named("sigs") = m->sigs,
                                            Rcpp::Named("G") = m->G, 
                                            Rcpp::Named("zigs") = m->zi_gs,
                                            Rcpp::Named("pi_gs") = m->pi_gs,
                                            Rcpp::Named("n_gs") = m->n_gs,
                                            Rcpp::Named("log_dets") = m->log_dets,
                                            Rcpp::Named("logliks") = m->logliks);
                                            
    
    return ret_val;
  }

    Rcpp::List ret_val = Rcpp::List::create(Rcpp::Named("X") = m->data,
                                            Rcpp::Named("mus") = m->mus, 
                                            Rcpp::Named("sigs") = m->sigs,
                                            Rcpp::Named("G") = m->G, 
                                            Rcpp::Named("zigs") = m->zi_gs,
                                            Rcpp::Named("pi_gs") = m->pi_gs,
                                            Rcpp::Named("n_gs") = m->n_gs,
                                            Rcpp::Named("log_dets") = m->log_dets,
                                            Rcpp::Named("logliks") = m->logliks);
  
  return ret_val;
}



// [[Rcpp::export]]
Rcpp::List e_step_internal(arma::mat X, // data 
                           int G, int model_id, // number of groups and model id (id is for parrallel use)
                           int model_type,  // covariance model type
                           Rcpp::List in_m_obj, // internal object from output
                           arma::mat init_zigs, 
                           double in_nu = 1.0)
{
  // declare params. that are passed in from the internal object.  
  std::vector< arma::rowvec > mus = in_m_obj["mus"]; 
  std::vector< arma::mat > sigs = in_m_obj["sigs"];
  std::vector<double> n_gs = in_m_obj["n_gs"];
  std::vector<double> log_dets = in_m_obj["log_dets"];
  arma::rowvec pi_gs = in_m_obj["pi_gs"];

  // create model and set existing parameters. 
  std::unique_ptr<Mixture_Model> m = std::unique_ptr<Mixture_Model>(create_model(&X,G,model_id,model_type));  
  m->data = X; 
  m->mus = mus; 
  m->sigs = sigs; 
  m->log_dets = log_dets; 
  m->pi_gs = pi_gs; 
  m->n_gs = n_gs; 
  m->zi_gs = init_zigs; 
  m->init_missing_tags(); //Additional graphical and num

  // invert symmetric matrices. 
  for(int g = 0; g < G; g++)
  {
    m->inv_sigs[g] = arma::solve(sigs[g],m->EYE,arma::solve_opts::refine); 
  }

  // perform e_step and imputation. 
  m->E_step_only_burn(); 

  Rcpp::List ret_val = Rcpp::List::create(Rcpp::Named("X") = m->data,
                                            Rcpp::Named("row_tags") = m->row_tags, 
                                            Rcpp::Named("origX") = X,
                                            Rcpp::Named("zigs") = m->zi_gs); 

  
  return ret_val;
}
