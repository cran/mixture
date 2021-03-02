#include "T_Mixture_Model.hpp"
#include <boost/math/special_functions/digamma.hpp> 
#include <boost/math/special_functions/gamma.hpp> 
#include "T_Solve.hpp"
#include <random> 

// #include <Eigen/Dense>

// **********************   GENERAL MIXTURE MODEL CLASS ****************************************************************************************   
// Mixture model class contains data, n, p, G, mus, sigs, inv_sigs, zi_gs, pi_gs, Ws, logliks, and tolerance settings                          *
// *********************************************************************************************************************************************      
// Constructor for general mixture model 
#pragma once
T_Mixture_Model::T_Mixture_Model(arma::mat* in_data, int in_G, int in_model_id) 
{
  data = *in_data; // pass pointer to data, its always going to be constant anyway
  n = in_data->n_rows; 
  p = in_data->n_cols; // dimensions 
  G = in_G;
  model_id = in_model_id; 
  nu = 1.0; 
  EYE = arma::eye(p,p); 

  // set size of mus based on number of groups
  std::vector<arma::rowvec> in_mus;
  in_mus.assign(G, arma::rowvec(p, arma::fill::zeros));
  mus = in_mus;

  // set size of sigs based on number of groups
  std::vector<arma::mat> in_sigs; 
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

  // t distributions. 
  vgs.assign(G,50.0); 
  wgs.assign(G,arma::vec(n,arma::fill::zeros));


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
  inter_Ws.assign(G,EYE);
  Ws = inter_Ws; 

  // set up log liklihoods tracking 
  std::vector<double> vect{1.0};
  logliks = vect; 
  tol_l = 1e-6;
  e_step = &T_Mixture_Model::RE_step; 
  m_step_vgs = &T_Mixture_Model::M_step_vgs_regular; 

}


// Deconstructor for general mixture model family 
#pragma once
T_Mixture_Model::~T_Mixture_Model() 
{
  // Rcpp::Rcout  << "De-allocating memory" << "\n";
}

// general m step for proportions 
#pragma once
void T_Mixture_Model::M_step_props()
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
#pragma once
void T_Mixture_Model::M_step_mus()
{
  for(int g = 0; g < G; g++){
    arma::rowvec inter_mus = arma::rowvec(p,arma::fill::zeros);
    
    double denom = 0.0; 
    for(int i = 0; i < n; i++)
    { 
      arma::rowvec inter_val = data.row(i);
      inter_mus += zi_gs.at(i,g)*wgs[g][i]*inter_val;
      denom += zi_gs.at(i,g)*wgs[g][i];
    }
    mus[g] = inter_mus/denom; 
  }
}

// within cluster scattering matrix calculation every M step 
#pragma once
void T_Mixture_Model::M_step_Ws()
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
      double zi_g_current = zi_gs.at(i,g)*wgs[g][i];
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
#pragma once
double T_Mixture_Model::mahalanobis(arma::rowvec x, arma::rowvec mu, arma::mat inv_sig)
{
  double mh = 0.0; 
  
  arma::rowvec xm = (x - mu);
  arma::mat inter_m = xm*inv_sig; 
  arma::rowvec inter_x = inter_m*xm.t(); 
  mh = inter_x[0]; 

  return mh; 
}

// calculates log density for a single x 
#pragma once
double T_Mixture_Model::log_density(arma::rowvec x, arma::rowvec mu, arma::mat inv_Sig, double log_det, double vg)
{

  // t distribution log density calculation, take in vg. 
  double vg_term = 0.5*(vg + p);
  // compute the log of gamma and other terms. 
  double numerator_term = boost::math::lgamma(vg_term) - 0.5*log_det; 
  // denominator term of t distribution. 
  double denominator_term = - 0.5*p*std::log(M_PI*vg) - boost::math::lgamma(0.5*vg) - vg_term*std::log(1 + mahalanobis(x,mu,inv_Sig)/vg);

  return(numerator_term + denominator_term);
}

#pragma once
double T_Mixture_Model::calculate_log_liklihood(void)
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
      row_sum_term += pi_gs[g]*std::exp(log_density(data.row(i),mus[g],inv_sigs[g],log_dets[g],vgs[g]));
    }
    row_sum_term = std::log(row_sum_term); 
    log_lik += row_sum_term; 
  }

  return log_lik; 
}

#pragma once 
void T_Mixture_Model::E_step(){
  (this->*e_step)();
}

// GENERAL E - Step for all famalies 
#pragma once
void T_Mixture_Model::RE_step()
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
      
      inter_density[g] = std::pow((pi_gs[g])*std::exp(log_density(data.row(i),mus[g], inv_sigs[g],log_dets[g],vgs[g])),nu);
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

      if(comparison_t(ss,1.0)){ 
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

#pragma once
void T_Mixture_Model::SE_step(void) // performs the stochastic estep . 
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
      
      inter_density[g] = std::pow((pi_gs[g])*std::exp(log_density(data.row(i),mus[g], inv_sigs[g],log_dets[g],vgs[g])),nu);
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

      if(comparison_t(ss,1.0)){ 
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

  inter_zigs = arma::mat(n,G,arma::fill::zeros); 

  // go through current z_ig. 
  for(int i = 0; i < n; i++){
    std::vector<double> params = arma::conv_to< std::vector<double> >::from(zi_gs.row(i)); 
    std::discrete_distribution<int> di (params.begin(), params.end());
    int assign_class = di(generator); 
    inter_zigs.at(i,assign_class) = 1; 
  }

  zi_gs = inter_zigs; 

}

// GENERAL E - Step for wgs
#pragma once
void T_Mixture_Model::E_step_ws(void){
  // go through groups and observations. 
  arma::vec inter_wig;

  for(int g = 0; g < G; g++){
    
    inter_wig = arma::vec(n,arma::fill::zeros);

    for(int i = 0; i < n; i++){
      // calculate
      inter_wig[i] = (vgs[g] + p) / (vgs[g] + mahalanobis(data.row(i), mus[g],inv_sigs[g])) ; 
    
    }

    wgs[g] = inter_wig; 
  }
}


#pragma once 
void T_Mixture_Model::M_step_vgs(){
  (this->*m_step_vgs)();
}

#pragma once 
void T_Mixture_Model::M_step_vgs_regular(void){

  for(int g = 0; g < G; g++){
   
    double vg_term = 0.5*(vgs[g] + p);
    double eta_g = 1.0 - std::log(vg_term) + digam(vg_term); 

    double ziggy_wg_terms = 0.0; 

    for(int i = 0; i < n; i++){
        ziggy_wg_terms += zi_gs.at(i,g)*( std::log(wgs[g][i]) - wgs[g][i] ); 
    }
    ziggy_wg_terms /= n_gs[g]; 
    eta_g += ziggy_wg_terms; 
    
    // std::cout << "abar: " << abar_gs[g] << " bbar: " <<  bbar_gs[g] << " cbar: " << cbar_gs[g] << std::endl;    
    try {
      double vgs_g = vgs_solve(eta_g,vgs[g], 0.1);
      if(!std::isnan(vgs_g) && !comparison_t(vgs_g,0)){
        vgs[g] = vgs_g; 
      }
    } catch(const std::exception& e){
      
    }; 
    
  }
}

#pragma once 
void T_Mixture_Model::M_step_vgs_constrained(void){
   
  double vg_term = 0.5*(vgs[0] + p);
  double eta_g = 1.0 - std::log(vg_term) + digam(vg_term); 

  for(int g = 0; g < G; g++){

    double ziggy_wg_terms = 0.0; 

    for(int i = 0; i < n; i++){
        ziggy_wg_terms += zi_gs.at(i,g)*( std::log(wgs[g][i]) - wgs[g][i] ); 
    }

    eta_g += ziggy_wg_terms/n; 
  }

  try {
      double vgs_g = vgs_solve(eta_g,vgs[0], 0.1);
      if(!std::isnan(vgs_g) && !comparison_t(vgs_g,0)){
        
        for(int g = 0; g < G; g++){
            vgs[g] = vgs_g;     
        }
        
      } 
    } catch(const std::exception& e) { }; 

}



// Debug functions
// sets each groups covariance to identity. 
#pragma once
void T_Mixture_Model::sig_eye_init() {
  arma::mat inter_eye =  EYE;
  for(int g = 0; g < G; g++){
    sigs[g] = inter_eye;
    inv_sigs[g] = inter_eye;
  }
}

// loglik tracking 
#pragma once
void T_Mixture_Model::track_lg_init(void)
{
  // get log_densities and set the first one This is a simple function and should be done after the first intialization
  //arma::rowvec model_lgs = log_densities(); 
  logliks[0] = calculate_log_liklihood();  //sum(model_lgs);
}

// This function keeps track of the log liklihood. You have to calculate the log densities, then keep track of their progress. 
#pragma once
bool T_Mixture_Model::track_lg(bool check)
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


#pragma once 
void T_Mixture_Model::z_ig_init_random(void) {
  
  // arma::mat inter_zigs = arma::mat(n,G,arma::fill::zeros);

  // for(int i = 0; i < n; i++){

  //   double inter_row_sum = 0.0; 

  //   for(int g = 0; g < G; g++){
  //     inter_zigs(i,g) = std::abs(Rcpp::rand() % (G + 50)); 
  //     inter_row_sum += inter_zigs.at(i,g);
  //   }
  //   // check for stupid nans
  
  //   for(int g = 0; g < G; g++){
      
  //     double numer_g = inter_row_sum - inter_zigs(i,g); 
  //     double denom_g = inter_zigs(i,g); 
  //     inter_zigs(i,g) = 1.0/(1 + numer_g/denom_g);

  //     if(isnan(inter_zigs(i,g))){
  //       inter_zigs.row(i) = inter_zigs.row(i-1);
  //       break; 
  //     }

  //   }    

  // }

  // zi_gs = inter_zigs; 
}




// ********************** END OF GENERAL MIXTURE MODEL CLASS ****************************************************************************************   
  
