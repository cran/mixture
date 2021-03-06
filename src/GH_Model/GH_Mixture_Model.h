

#include <armadillo> 
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/bessel_prime.hpp>
#include <boost/math/special_functions/digamma.hpp> 
#include <boost/math/special_functions/gamma.hpp> 
#include <iostream> 
#include <vector> 
#include "math.h"
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_result.h> 
#include <exception> 
#include <random> 
#include "../Cluster_Error.hpp"

#define loggy(x) Rcpp::Rcout << x << std::endl 
#define bessy(nu,x) LG_k_bessel(nu,x)
#define bessy_prime(nu,x) (bessy(nu+eps,x) - bessy(nu,x))/(eps)
#define space << " " << 
const double two_eps = 0.002;
const double pi_my = 3.1415926535897932;

#pragma once 
std::default_random_engine generator_gh;


#pragma once 
bool comparison(double a, double b);

#pragma omp declare reduction( + : arma::rowvec : omp_out += omp_in ) \
                 initializer( omp_priv = omp_orig )   


#pragma omp declare reduction( + : arma::vec : omp_out += omp_in ) \
                 initializer( omp_priv = omp_orig )   


#pragma omp declare reduction( + : arma::mat : omp_out += omp_in ) \
                 initializer( omp_priv = omp_orig )   

#pragma once
class GH_Mixture_Model 
{
    public:
        // Constructor
        // data is passed in by (p x n) dimension where n is number of observations  
        GH_Mixture_Model(arma::mat* in_data, int in_G, int model_id ); 
        // Deconstructor
        virtual ~GH_Mixture_Model(); 

        int n; // number of observations
        int model_id; // id number, can be assigned for parallel or other testing purposes. 
        std::vector<double> n_gs;
        int p; // dimension
        int G; // number of groups 
        std::vector<double> log_dets;
        std::vector< arma::vec > mus; // all mixture models have location vectors
        std::vector< arma::vec > alphas; // skewness vectors. 
        std::vector< arma::mat > sigs; // all mixture models have covariance matrices
        std::vector< arma::mat > inv_sigs; // inverse of all sigs
        arma::mat data; // data for the model
        arma::vec pi_gs; // mixing proportions
        arma::mat zi_gs; // posteriori
        arma::mat prev_zi_gs;  // previous a-posterori
        std::vector< double > prev_abar_gs; // previous latent variables
        std::vector< double > prev_bbar_gs; // previous latent variables
        std::vector< double > prev_cbar_gs; // previous latent variables
        std::vector< arma::mat > Ws; // within cluster scattering matrix per cluster
        std::vector< double > logliks; // std vector containing logliklihoods for each iteration.
        size_t log_iter_max = 1000;
        double tol_l; // tolerance for logliklihood check.
        double nu; // deterministic annealing method.
        // gamma distribution parameters. 
        std::vector< double > omegas; 
        std::vector< double > lambdas;  
        arma::mat EYE; 

        std::vector< arma::vec > a_is; // expectation of latent parameter 
        std::vector< arma::vec > b_is;  // expectation of lg of y 
        std::vector< arma::vec > c_is; // expectation of inverse of y
        std::vector< double > abar_gs; //
        std::vector< double > bbar_gs; //  
        std::vector< double > cbar_gs; // 
        
        // General Methods
        void random_soft_init(void); // self explanitory  
        double calculate_log_liklihood(void); // calcuates the logliklihood function over all 
        void M_step_alphas(void); // calsculate alpha_gs under infinite logliklihood problem. 


        // mahalanobis distance calculation with skewed approach 
        double mahalanobis(arma::vec x, // vector comes in as 1 x p
                        arma::vec mu, 
                        arma::vec alpha,
                        double y_s,
                        double inv_y, 
                        arma::mat inv_sig);
        
        // log density calculation. 
        double log_density(arma::vec x, // vector comes in as 1 x p  
                        arma::vec mu,
                        arma::vec alpha,
                        double y_ig, // latent variables
                        double lg_y_ig, 
                        double inv_y_ig,
                        arma::mat inv_Sig, 
                        double log_det, 
                        double omega_g,
                        double lambda_g); 
        // calculates a particular log-density for a specific group for a specific x

        void E_step(); // performs the e-step calculation on the T_Mixture_Model, stores data in z_igs. 
        // stochastic methods
        void SE_step(void); // performs the stochastic estep .
        void RE_step(void);  
        void (GH_Mixture_Model::*e_step)(void); 

        void set_E_step(bool stochastic){
            if(stochastic){
                this->e_step = &GH_Mixture_Model::SE_step; 
            }
            else{
                // e_step is already regular. 
            }
        };
        
        void M_step_props(); // calculate the proportions and n_gs
        void M_step_init_gaussian(void); // initializes mu, sig, and alphas according to gaussian settings. alpha is really small.  
        double k_bessel(double nu, double x); 
        double LG_k_bessel(double nu, double x);


        // GH EStep for y , log(y) and 1/y  
        void E_step_latent(void); 
        void M_step_mus(void); // calculates the m step for mean vector

        // GH gamma update step 
        void M_step_gamma(void); 
        double q_function(int g);
        double q_deriv(int g);
        double q_deriv_2(int g);



        arma::mat mat_inverse(arma::mat X); // matrix inverse
        void M_step_Ws(); // calculates both Wk and W
        virtual void m_step_sigs() {
            //Rcpp::Rcout  << "m_step for general, user should not be here" << std::endl;
        };
        virtual void set_defaults() {
            //Rcpp::Rcout  << "set defaults virtual, user should not be here" << std::endl;
        };
        virtual void set_m_iterations(int in_iter, double in_tol) {
            //Rcpp::Rcout  << "set iterations virtual, user should not be here" << std::endl;
        };
        virtual void set_m_iterations(int in_iter, double in_tol, arma::mat in_D) {
            //Rcpp::Rcout  << "set iterations virtual, user should not be here" << std::endl;
        };
        void track_lg_init(void); // grab the first calculation of the logliklihood and replace the 1.0 from the constructor
        bool track_lg(bool check); // calculate current log liklihood of the model. If bool is check aitken convergence.
        bool check_aitkens(void); // check aitken's convergence criterion. 


        // Debug functions
        void sig_eye_init(); // sets each of the groups covariance matrix to identity.

        // missing data 
        double nu_d; // deterministic annealing method.
        
        // Missing Data
        std::vector < arma::uvec > missing_tags; // holds the col tag, as well as dimension tags for missing data NAs.
        arma::uvec col_tags; // arma unsigned column vector for col tags contain the missing data.
        // Missing Data methods.
        void init_missing_tags(void); // initalizes the two variables above by going through the data and pulling location of NAs.
        void EM_burn(int in_burn_steps); // runs the EM algorithm in_burn_steps number of times only on the Non missing data.
        void E_step_only_burn(void); // this function is only used for an existing model. See e_step_internal
        void impute_init(void); // initialize x_m values for imputation so that there are no NAs in the dataset.
        void impute_cond_mean(void);

        // multi-threading methods 
        int num_of_threads; 
        void M_step_mus_threaded(void); 
        void M_step_Ws_threaded(void); 
        void E_step_latent_threaded(void); 
        void E_step_threaded(void); 
        void track_lg_init_threaded(void); // grab the first calculation of the logliklihood and replace the 1.0 from the constructor
        bool track_lg_threaded(bool check); // calculate current log liklihood of the model. If bool is check aitken convergence.
        double calculate_log_liklihood_threaded(void); // calcuates the logliklihood function over all 
    
}; 



bool comparison(double a, double b){
    double tolerance  = abs( a - b);
    bool result = tolerance < eps;
    return (result);    
}


double GH_Mixture_Model::LG_k_bessel(double nu, double x){


  // set up safety net for this function. 
  gsl_sf_result result;
  int status;    
  
  status = gsl_sf_bessel_lnKnu_e(nu,abs(x), &result );
    
    if(isnan(result.val)){
      status = 1; 
    }

  if(status == 0){

      return(result.val); 

  } else {
    // try scaled version 
    status = gsl_sf_bessel_Knu_scaled_e(nu, abs(x), & result); 
    
    if(isnan(result.val)){
      status = 1; 
    }

    if(status == 0) {
      return( log(result.val/exp(abs(x))) );

    } else {
      double approx_result = 0.0;

      try{ 

          approx_result = 0.5*(log(M_PI) - log(2.0) - log(nu) ) - nu*log(M_E*x) + nu*log(2.0*nu); 
        
          if(isnan(approx_result)){
            // overflow has occured. 
            return (log(1e-100));
          }

          return(approx_result);
      }
      catch(...) {
        // if all else fails just return 1.0 
          return (log(1e-100));
      }

    }

  }

  return (log(1e-100)); 
}


// CONSTURCTOR
GH_Mixture_Model::GH_Mixture_Model(arma::mat* in_data, int in_G, int model_id ){

    data = *in_data; 

    // set usual parameters 
    p = data.n_rows;
    n = data.n_cols;  
    G = in_G; 

    // skewed parameteers initializations. 
    std::vector< arma::vec > mus_temp; // location 
    std::vector< arma::vec > alphas_temp; // skewness 
    std::vector< arma::mat > sigs_temp;  // covariance 
    std::vector< double > omegas_temp; 
    std::vector< double > lambdas_temp; 
    std::vector< arma::vec > y_is_temp; 

    const arma::vec zv = arma::vec(p,arma::fill::zeros); 
    const arma::mat zS = arma::mat(p,p,arma::fill::eye); 
    const arma::vec zy = arma::vec(n,arma::fill::ones); 
    
    abar_gs = std::vector<double>();
    bbar_gs = std::vector<double>();
    cbar_gs = std::vector<double>(); 


    for(int g = 0; g < G; g++){
       mus_temp.push_back(zv); 
       alphas_temp.push_back(zv); 
       sigs_temp.push_back(zS); 
       omegas_temp.push_back(0.0); 
       lambdas_temp.push_back(0.0); 
       y_is_temp.push_back(zy); 
       abar_gs.push_back(0.0);
       bbar_gs.push_back(0.0);
       cbar_gs.push_back(0.0); 
    }

    // set all parameters to zero upon initialization. 
    mus = mus_temp; 
    alphas = alphas_temp;
    sigs = sigs_temp; 
    inv_sigs = sigs_temp; 
    omegas = omegas_temp;
    lambdas = lambdas_temp;  
    Ws = sigs_temp; 
    log_dets = lambdas_temp; 
    zi_gs = arma::mat(n,p,arma::fill::zeros); 
    a_is = y_is_temp; 
    b_is = y_is_temp; 
    c_is = y_is_temp; 

    // set up log liklihoods tracking 
    std::vector<double> vect{1.0};
    logliks = vect; 
    tol_l = 1e-6;
    nu_d = 1.0; 
    EYE = arma::eye(p,p); 
    e_step = &GH_Mixture_Model::RE_step; 
}


// DECONSTRUCTOR 
GH_Mixture_Model::~GH_Mixture_Model(){


}



// initialize random soft. 
void GH_Mixture_Model::random_soft_init()
{


  arma::mat z_ig_temp = arma::mat(n,G,arma::fill::randu)*100.0;

  double row_sum;  
  for(int i = 0; i < n; i++)
  {
    row_sum = 0.0; 

    for(int g = 0; g < G; g++){ 
      row_sum += z_ig_temp.at(i,g);
    }

    z_ig_temp.row(i) = z_ig_temp.row(i)/row_sum; 
    
    if( arma::sum(zi_gs.row(i)) != 1.0 ){
      z_ig_temp.row(i) = z_ig_temp.row(i)/arma::sum(z_ig_temp.row(i)); 
    }

  }

  zi_gs = z_ig_temp;
}


bool GH_Mixture_Model::check_aitkens(void) {
        int last_index = logliks.size();
        double l_p1 =  logliks[last_index-1];
        double l_t =  logliks[last_index-2];
        if( isnan(l_p1) || isinf(l_p1) ){
          infinite_loglik_except e; 
          throw e; 
        }
        if(l_p1 > l_t){
          loglik_decreasing e;
          throw e; 
        }
        double l_m1 = logliks[last_index-3];
        double a_t = (l_p1 - l_t)/(l_t - l_m1);
        double l_Inf = l_t + (l_p1 - l_t)/(1.0-a_t);
        double val = std::abs((l_Inf - l_t));
        return (bool)(val < tol_l);
}



// loglik tracking 
void GH_Mixture_Model::track_lg_init(void)
{
  // get log_densities and set the first one This is a simple function and should be done after the first intialization
  //arma::rowvec model_lgs = log_densities(); 
  logliks[0] = calculate_log_liklihood();  //sum(model_lgs);
}

// This function keeps track of the log liklihood. You have to calculate the log densities, then keep track of their progress. 
bool GH_Mixture_Model::track_lg(bool check)
{
  
  if (check) {
    logliks.push_back(calculate_log_liklihood());    
    return false; 
  }
  else {

    double c_loglik = calculate_log_liklihood();

    if( isnan(c_loglik) || isinf(c_loglik) ){
        
        if(logliks.size() < 10){
          infinite_loglik_except e;
          throw e; 
        }

        zi_gs = prev_zi_gs; 
        abar_gs = prev_abar_gs;
        bbar_gs = prev_bbar_gs;
        cbar_gs = prev_cbar_gs; 

        M_step_props();
        M_step_alphas(); // new function to update mus 
        M_step_Ws(); 
        m_step_sigs(); 
        M_step_gamma(); 

        for( size_t i = 0; i < 50; i++) {

          if(check_aitkens() == true){ return true; }

          else 
          {

            E_step_latent();
            M_step_props();
            M_step_alphas(); // new function to update alphas
            M_step_Ws(); 
            m_step_sigs(); 
            M_step_gamma(); 

            c_loglik = calculate_log_liklihood();

            if(isnan(c_loglik) || isinf(c_loglik)){
              zi_gs = prev_zi_gs; 
              abar_gs = prev_abar_gs;
              bbar_gs = prev_bbar_gs;
              cbar_gs = prev_cbar_gs; 
              
              E_step_latent();
              M_step_props();
              M_step_alphas(); // new function to update alphas
              M_step_Ws(); 
              m_step_sigs(); 
              M_step_gamma(); 
              
              infinite_loglik_with_return_except e; 
              throw e;
            }
            check_aitkens();
            logliks.push_back(c_loglik);

          }

        }

      return(check_aitkens());

    } else {
        prev_zi_gs = zi_gs;
        prev_abar_gs = abar_gs;
        prev_bbar_gs = bbar_gs; 
        prev_cbar_gs = cbar_gs;
        logliks.push_back(c_loglik);
    }

    //checking aitkens convergence criterion 
    int last_index = logliks.size();
    double l_p1 =  logliks[last_index-1];
    double l_t =  logliks[last_index-2];
    double l_m1 = logliks[last_index-3];
    double a_t = (l_p1 - l_t)/(l_t - l_m1);
    double l_Inf = l_t + (l_p1 - l_t)/(1.0-a_t);
    double val = std::abs((l_Inf - l_t));
    return (bool)(val < tol_l); 
  }
}


double GH_Mixture_Model::mahalanobis(arma::vec x, // vector comes in as 1 x p
                        arma::vec mu, 
                        arma::vec alpha,
                        double y_s,
                        double inv_y, 
                        arma::mat inv_sig)
{

	arma::vec xma = (x - mu - alpha*y_s); 

	double res = arma::trace(inv_sig*(xma*xma.t()));  

  double mh = res*(1.0/y_s);

  return (mh); 
}


        // log density calculation. 
double GH_Mixture_Model::log_density(arma::vec x, // vector comes in as 1 x p  
                        arma::vec mu,
                        arma::vec alpha,
                        double y_ig, // latent variables
                        double lg_y_ig, 
                        double inv_y_ig, 
                        arma::mat inv_Sig, 
                        double log_det, 
                        double omega_g,
                        double lambda_g) 
{


  const double nu = lambda_g - p/2.0;
  const double rho = arma::trace(inv_Sig*alpha*alpha.t()); 
  arma::vec xm = x - mu;
  double delta = arma::trace(inv_Sig*xm*xm.t());

  if( comparison(delta,0) ){
    delta = 0.0001;
  }

  double bess_input = sqrt( (delta + omega_g)*(rho + omega_g) );
  double leading_terms = - (p/2.0)*log(2.0*M_PI) - 0.5*log_det - LG_k_bessel(abs(lambda_g),omega_g);
  double middle_terms = arma::trace(inv_Sig*(x-mu)*alpha.t());
  double third_term = (nu/2.0)*( log(delta + omega_g) - log( rho + omega_g));
  double bessel_term = LG_k_bessel(abs(nu),bess_input);

  if( isnan(bessel_term)){
    bessel_term = log(1e-10);
  }

  double result = leading_terms + middle_terms +third_term + bessel_term; 
  
  return(result);

}




double GH_Mixture_Model::calculate_log_liklihood(void) {

  double lglik = 0.0;
  double row_sum_term = 0.0; 
  // go through observations.
  for(int i = 0; i < n; i++)
  {
    // go through groups
    row_sum_term = 0.0; 
    for(int g = 0; g < G; g++) 
    {
      row_sum_term += pi_gs[g]*exp(log_density(data.col(i),mus[g],alphas[g], // parameters and observations
                          a_is[g].at(i),c_is[g].at(i),b_is[g].at(i), // latent variables
                          inv_sigs[g],log_dets[g],omegas[g],lambdas[g] // other parameters 
                          )); 
    } 
    row_sum_term = std::log(row_sum_term); 
    lglik += row_sum_term; 
  }

  return(lglik); 
}



void GH_Mixture_Model::M_step_props() {

  // initialize mixing proportions and n_gs 
  arma::vec in_pigs = arma::vec(G, arma::fill::zeros);
  std::vector<double> inter_n_gs;
  inter_n_gs.assign(G, 0);
  
  // calculate new mixing proportions and n_gs 
  for(int g = 0; g < G; g++)
  {
    for(int i = 0; i < n; i++)
    {
        // estimated number of observations
        inter_n_gs[g] += zi_gs.at(i,g);  
    }
    if(inter_n_gs[g] < 1){
      below_1_ng_except e; 
      throw e; 
    }
    // mixing proportion
    in_pigs[g] = inter_n_gs[g]/n;
  }

  // set new values 
  n_gs = inter_n_gs; 
  pi_gs = in_pigs; 

}




void GH_Mixture_Model::M_step_init_gaussian(void) {

  // calculate fake mus and sigs. 
  for(int g = 0; g < G; g++) {
    arma::vec inter_mu = arma::vec(p,arma::fill::zeros); 
    arma::mat inter_Sig = arma::mat(p,p,arma::fill::zeros); 

    int i = 0;
    for(i = 0; i < n; i++ ){
      inter_mu += zi_gs.at(i,g)*data.col(i); 
    }
    mus[g] = inter_mu/n_gs[g]; 
    const arma::vec mu_g = mus[g]; 

    arma::vec XM = arma::vec(p,arma::fill::zeros); 
    for(i = 0; i < n; i++){
      XM = data.col(i) - mu_g; 
      inter_Sig += zi_gs.at(i,g)*((XM)*(XM.t())); 
    }

    sigs[g] = inter_Sig/n_gs[g]; 
    inv_sigs[g] = arma::solve(sigs[g],EYE,arma::solve_opts::refine);
    log_dets[g] = log(arma::det(sigs[g]));

    // add a tiny amount of skewness just like michael did. 
    alphas[g] = arma::vec(p,arma::fill::ones)*1; 
    // add a large amount for the gamma parameter
    omegas[g] = 3.0; 
    lambdas[g] = 3.0; 
  }

}



#pragma once 
void GH_Mixture_Model::E_step(){
  (this->*e_step)();
}


#pragma once
void GH_Mixture_Model::SE_step(void) // performs the stochastic estep . 
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
    inter_row_sum = 0.0;
    inter_density = arma::rowvec(G,arma::fill::zeros); 

    for(int g = 0; g < G; g ++)
    {
      // numerator in e step term for mixture models

      double log_dens = log_density(data.col(i),mus[g],alphas[g], // basic parameters
                                                        a_is[g][i],c_is[g][i],b_is[g][i], // latent parameters
                                                        inv_sigs[g],log_dets[g],omegas[g],lambdas[g]);
                                                        
      inter_density[g] = std::pow((pi_gs[g])*exp(log_dens),nu_d); // other 

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

      if(comparison(ss,1.0)){ 
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
        break; 
      }
      count ++ ;
    }


  }
  zi_gs = inter_zigs; 

  inter_zigs = arma::mat(n,G,arma::fill::zeros); 

  // go through current z_ig. 
  for(int i = 0; i < n; i++){
    std::vector<double> params = arma::conv_to< std::vector<double> >::from(zi_gs.row(i)); 
    std::discrete_distribution<int> di (params.begin(), params.end());
    int assign_class = di(generator_gh); 
    inter_zigs.at(i,assign_class) = 1; 
  }

  zi_gs = inter_zigs; 

}


// GENERAL E - Step for all famalies  
void GH_Mixture_Model::RE_step()
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
    inter_row_sum = 0.0;
    inter_density = arma::rowvec(G,arma::fill::zeros); 

    for(int g = 0; g < G; g ++)
    {
      // numerator in e step term for mixture models

      double log_dens = log_density(data.col(i),mus[g],alphas[g], // basic parameters
                                                        a_is[g][i],c_is[g][i],b_is[g][i], // latent parameters
                                                        inv_sigs[g],log_dets[g],omegas[g],lambdas[g]);
                                                        
      inter_density[g] = std::pow((pi_gs[g])*exp(log_dens),nu_d); // other 

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

      if(comparison(ss,1.0)){ 
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
        break; 
      }
      count ++ ;
    }


  }
  zi_gs = inter_zigs; 
}



// LATENT E STEP 
void GH_Mixture_Model::E_step_latent(void)
{

  int g, i;  

  double a_bar_g, b_bar_g, c_bar_g; 

  for(g = 0; g < G; g++){


    a_bar_g = 0.0;
    b_bar_g = 0.0; 
    c_bar_g = 0.0;

    for(i = 0; i < n; i++){

      // calculate a_is and other terms! 
      const arma::vec x = data.col(i); 
      arma::vec xm = (x - mus[g]);

      const double delta = arma::trace(inv_sigs[g]*xm*xm.t());  

      double alpha_term = arma::trace(inv_sigs[g]*alphas[g]*alphas[g].t()); 

      const double rho = (alpha_term);

      const double product_terms = sqrt((omegas[g] + delta)*(omegas[g] + rho)); 

      // loggy(gammas[g]);
      const double nu = lambdas[g] - p/2; 

      const double K_dfp1_prod = LG_k_bessel(abs(nu+1),product_terms); 
      const double K_df_prod = LG_k_bessel(abs(nu),product_terms);
  
      // calculate a_is.
      double y_ig = exp(0.5*(log(omegas[g] + delta) - log(omegas[g] + rho)) + K_dfp1_prod - K_df_prod);  //log(K_dfp1_prod) - log(K_df_prod); 

      if(y_ig < 1.0e20){
        a_is[g].at(i) = y_ig;
      }

      double inv_y_ig = 0.5*(log(omegas[g] + rho) - log(omegas[g] + delta)) +  K_dfp1_prod - K_df_prod;// log(K_dfp1_prod) - log(K_df_prod); 
      double b_current = exp(inv_y_ig) - 2.0*(nu)/(omegas[g] + delta);

      if(b_current < 1.0e20){
         b_is[g].at(i) = b_current; 
      }

      double bprime = (LG_k_bessel(abs(nu+eps),abs(product_terms)) - LG_k_bessel(abs(nu),abs(product_terms)))/eps ;

      double log_y_ig = 0.5*(log(omegas[g] + delta) - log(omegas[g] + rho)) + bprime;   

      if(log_y_ig < 1.0e20) {
        c_is[g].at(i) = log_y_ig; 
      }
      
    }

    a_bar_g  = arma::sum(zi_gs.col(g) % a_is.at(g)); 
    b_bar_g = arma::sum(zi_gs.col(g) % b_is.at(g));
    c_bar_g = arma::sum(zi_gs.col(g) % c_is.at(g));

    a_bar_g = a_bar_g/n_gs[g]; 
    b_bar_g = b_bar_g/n_gs[g]; 
    c_bar_g = c_bar_g/n_gs[g]; 

    abar_gs[g] = a_bar_g;
    bbar_gs[g] = b_bar_g;
    cbar_gs[g] = c_bar_g; 

  }
}


// function under infinite liklihood problem. 
void GH_Mixture_Model::M_step_alphas(void) {

  arma::vec mu_g, alpha_g;  
  for(int g = 0; g < G; g++){
 
    
    mu_g = mus[g]; 
    alpha_g = arma::vec(p,arma::fill::zeros); 

    double a_sum = arma::sum(a_is.at(g) % zi_gs.col(g)); 

    for(int i = 0; i < n; i++){
      
      // denominator term is constant for both calculations
      double z_ig  = zi_gs.at(i,g);
      arma::vec x_i = data.col(i);
    
      alpha_g += (z_ig*(x_i - mu_g));
   
    }
    alphas[g] = alpha_g/a_sum;
  }

}



void GH_Mixture_Model::M_step_mus(void) {

  arma::vec mu_g, alpha_g;  
  for(int g = 0; g < G; g++){
 
    // these are used throughout the loop
    const double a_bar = abar_gs.at(g);
    const double b_bar = bbar_gs.at(g);
    
    mu_g = arma::vec(p,arma::fill::zeros); 
    alpha_g = mu_g; 


    double b_sum = arma::sum(b_is.at(g) % zi_gs.col(g)); 

    double denom = a_bar*b_sum - n_gs.at(g);

    for(int i = 0; i < n; i++){
      
      // denominator term is constant for both calculations
      double z_ig  = zi_gs.at(i,g);
      arma::vec x_i = data.col(i);

      mu_g += (x_i*z_ig*(a_bar*b_is.at(g).at(i) - 1));
    
      alpha_g += (x_i*z_ig*(b_bar - b_is[g].at(i) ));
    }

    alphas[g] = alpha_g/denom;
    mus[g] = mu_g/denom; 
  }

}



void GH_Mixture_Model::M_step_Ws(void) {
  
  for(int g = 0; g < G; g++)
  {
    // set up some parameters ahead of time
    arma::mat W_g = arma::mat(p,p,arma::fill::zeros); 
    const arma::vec bs = b_is.at(g); 
    const arma::vec as = a_is.at(g);
    const arma::vec alpha_g = alphas.at(g); 
    const arma::vec mu_g = mus.at(g); 

    // flurry matrix calculation 

    for(int i = 0; i < n; i++)
    {
      const arma::vec xm = data.col(i) - mu_g; 
      W_g +=  zi_gs.at(i,g)*( bs.at(i)*xm*xm.t() - xm*alpha_g.t() - alpha_g*xm.t() +  as.at(i)*alpha_g*alpha_g.t() ); 
    
    }

    Ws.at(g) = W_g/n_gs[g]; // I divide by n_gs here for simplicity, this is similar to the cov.wt R, 
    // do not confuse this statement with the true flury matrix.  
  }

}

double GH_Mixture_Model::q_function(int g){
  double result = -bessy(lambdas[g],omegas[g]) + (lambdas[g] - 1)*cbar_gs[g] - 0.5*omegas[g]*(abar_gs[g] + bbar_gs[g]); 
  return(result); 
}


// computes the first order forward derivarive approximation. 
double GH_Mixture_Model::q_deriv(int g){
  double result = 0.0; 
  double first_deriv_term = -(bessy(lambdas[g],omegas[g]+eps) - bessy(lambdas[g],omegas[g]))/eps; 
  result = first_deriv_term - 0.5*(abar_gs[g] + bbar_gs[g]); 
  return(result); 
}

double GH_Mixture_Model::q_deriv_2(int g){
  const double lam_g = lambdas[g]; 
  const double om_g = omegas[g]; 
  double result = -(( bessy(lam_g,om_g + 2.0*eps) - 2.0*bessy(lam_g, om_g + eps) + bessy(lam_g,om_g))/(eps*eps)); 
  return (result); 
}


void GH_Mixture_Model::M_step_gamma(void) {

  double omega_new = 0.0; 
  double lambda_new = 0.0;

  for(int g = 0; g < G; g++){
    double bess_prime = (LG_k_bessel(abs(lambdas[g] + eps),omegas[g]) - LG_k_bessel(abs(lambdas[g]),omegas[g]))/eps; 
    lambda_new = cbar_gs[g]*lambdas[g]/bess_prime; 
    omega_new = omegas[g] - q_deriv(g)/q_deriv_2(g); 
    //if(lambda_new < 0){ lambda_new = abs(lambda_new);}
    if(isnan(lambda_new) || isinf(lambda_new)){}
    else{ lambdas[g] = lambda_new;}
    if(isnan(omega_new) || isinf(omega_new)){}
    else{omegas[g] = omega_new; }
  }

}


// MISSING DATA METHODS FOR GENERAL Mixture_Model class. 
// grabs missing values and their respective tags and col_tags. 
void GH_Mixture_Model::init_missing_tags(void)
{

  std::vector<arma::uvec> in_missing_tags; // create the missing tags vector 
  arma::uvec in_col_tags; // set up row tags.  

  // loop through rows. 
  for(int i = 0; i < n;  i++ )
  {
    // get the current nantags. 
    arma::uvec nan_tags = arma::find_nonfinite(data.col(i));

    if(nan_tags.n_elem > 0)
    {
      // get row id uvec. 
      arma::uvec col_id = arma::uvec(1); // init
      col_id[0] = i; // set entry 
                    
      // concatonate both row uvec and the nan_tags
      arma::uvec mis_tag_i = arma::join_cols(col_id,nan_tags); 
      in_col_tags = arma::join_cols(in_col_tags,col_id); 

      // add to missing tags list
      in_missing_tags.push_back(mis_tag_i); 
    } 
  }
  // assign row tags and missing tags. 
  col_tags = in_col_tags; 
  missing_tags = in_missing_tags; 
}


// EM BURN_in METHOD 
// takes in number of steps to run the EM algorithm WITHOUT imputation of missing data. 
// this will initialize the model for better imputation 
void GH_Mixture_Model::EM_burn(int in_burn_steps)
{
  // copy dataset and z_igs. 
  arma::mat* orig_data = new arma::mat(p,n); // create empty arma mat on the heap. 
  arma::mat* orig_zi_gs = new arma::mat(n,G); 

  std::vector < arma::vec  > orig_a_is = a_is; 
  std::vector < arma::vec  > orig_b_is = b_is; 
  std::vector < arma::vec  > orig_c_is = c_is; 


  *orig_data = data; // set orig_data. 
  *orig_zi_gs = zi_gs; // set zi_igs. 

  // remove all data, and zi_gs with missing values. 
  data.shed_cols(col_tags); 
  zi_gs.shed_rows(col_tags); 

  for(int g = 0; g < G; g++ )
  {

      a_is[g].shed_rows(col_tags);
      b_is[g].shed_rows(col_tags);
      c_is[g].shed_rows(col_tags); 

  }
 
  n = data.n_cols; 
  
  // intialize all parameters. (all methods are on self)
  M_step_props();
  M_step_init_gaussian();
  E_step_latent();
  M_step_mus();
  M_step_Ws(); 
  m_step_sigs(); 
  M_step_gamma(); 


  // run EM burn in for in_burn_steps number of steps.  
  for(int i = 0; i < in_burn_steps; i++)
  {
        E_step(); 
        E_step_latent();
        M_step_props();
        M_step_mus();
        M_step_Ws(); 
        m_step_sigs(); 
        M_step_gamma(); 
  }

  // Now replace back the original data points and zi_igs. only keep the parmaeters.   
  data = *orig_data; 
  zi_gs = *orig_zi_gs; // done EM burn 
  a_is = orig_a_is; 
  b_is = orig_b_is; 
  c_is = orig_c_is; 
  n = data.n_cols; 

}


// imputation cond_mean functions. replaces values based on approach. 
void GH_Mixture_Model::impute_cond_mean(void)
{

  // go through each of the tags and select the row out of the dataset 
  for(size_t i_tag = 0; i_tag < col_tags.n_elem; i_tag++)
  {
    arma::uvec current_tag = missing_tags[i_tag];// get current full tag. 
    current_tag.shed_row(0); // remove the row tag.  
    arma::uword col_tag = col_tags.at(i_tag); 

     // create the missing column vector with current tag the from the data. 
    arma::mat c_obs = data.col(col_tag);
    arma::mat m_obs = data.col(col_tag);
    arma::mat nm_obs = data.col(col_tag); // use column vectors for calculations. 

    // drop the indeces that contain NAS.
    nm_obs.shed_rows(current_tag);
    // select only ones that contain NAS.
    m_obs = m_obs.rows(current_tag); 

    // now I have to iterate through g groups to compute the conditional mean imputation. 
    // remember to change this for groups G. it doesnt have to be at 2.  
    for(int g = 0; g < G; g++)
    {
      // current mus. 
      arma::mat c_mu_m = mus[g] + alphas[g]*a_is[g].at(i_tag); // invert because column vector. 
      arma::mat c_mu_nm = mus[g] + alphas[g]*a_is[g].at(i_tag); // invert because column vector.
      arma::mat c_sig = sigs[g]*a_is[g].at(i_tag); // self explanitory. (current sig)

      c_mu_m = c_mu_m.rows(current_tag); // select missing rows of mu
      c_mu_nm.shed_rows(current_tag); // select non missing rows of mu 

      arma::mat c_sig_m = c_sig; // set c_sigs because I have to shed. 
      arma::mat c_sig_nm = c_sig;

      c_sig_m.shed_cols(current_tag);  // shed rows and select one you need. 
      c_sig_m = c_sig_m.rows(current_tag); 


      c_sig_nm.shed_cols(current_tag); 
      c_sig_nm.shed_rows(current_tag); // make this square, non missing. 

      // finally compute imputation. 
      double z_ig = zi_gs.at(col_tag,g); // CHANGE THIS LATER IN FULL FUNCTION. 

      // MIXTURES OF CONDITIONAL MEAN IMPUTATION 
      arma::vec x_nm_dif = nm_obs - c_mu_nm; 
      // note to self during debug, check if you are on first becuase of nans and assign conditional mean imputation 
      int p_nm = c_sig_nm.n_rows;
      if(g == 0){
        m_obs = z_ig*( c_mu_m +  c_sig_m * arma::solve(c_sig_nm,arma::eye(p_nm,p_nm),arma::solve_opts::refine) * x_nm_dif); 
      }
      else{
        m_obs += z_ig*( c_mu_m +  c_sig_m * arma::solve(c_sig_nm,arma::eye(p_nm,p_nm),arma::solve_opts::refine) * x_nm_dif); 
      }

    }
    // assign missing values. 
    for(size_t m_i = 0; m_i < m_obs.size(); m_i++ )
    {
      data.at(current_tag[m_i],col_tag) = m_obs.at(m_i); 
    }
  }

}

void GH_Mixture_Model::impute_init(void)
{
  impute_cond_mean(); // after burn you impute as initalization. 
  E_step(); // calculate e step over entire dataset. 
  E_step_latent();
  // run m_step once. 
  M_step_props(); 
  M_step_mus();
  M_step_Ws();
  m_step_sigs();
}

void GH_Mixture_Model::E_step_only_burn(void) {

  // impute_cond_mean using z_igs. 
  impute_cond_mean(); 
  E_step(); // then perform E_step on the entire dataset. 
  impute_cond_mean(); // impute conditional mean again. 
  E_step(); // e_step again. 
  impute_cond_mean(); // one more for good luck 
  E_step(); // e_step. Not that every E_step, the parmaeters dont change but the imputation does. 
}



#include "GH_VVV.hpp"
#include "GH_VVI.hpp"
#include "GH_VVE.hpp"
#include "GH_VII.hpp"
#include "GH_VEV.hpp"
#include "GH_VEI.hpp"
#include "GH_VEE.hpp"
#include "GH_EVV.hpp"
#include "GH_EVI.hpp"
#include "GH_EVE.hpp"
#include "GH_EII.hpp"
#include "GH_EEV.hpp"
#include "GH_EEI.hpp"
#include "GH_EEE.hpp"

