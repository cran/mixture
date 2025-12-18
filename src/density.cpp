// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppGSL)]]

#include <RcppArmadillo.h>
#include <cmath>
#include <gsl/gsl_sf_bessel.h>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/gamma.hpp>

using namespace Rcpp;

const double eps = 0.00001; 

// -----------------------------------------------------------------
// Helper functions
static bool comparison(double a, double b){
    double tolerance  = abs( a - b);
    bool result = tolerance < eps;
    return (result);    
}

static double mahalanobis(arma::rowvec x, arma::rowvec mu, arma::mat inv_sig){
  arma::rowvec xm = (x - mu);
  double mh = arma::as_scalar(xm * inv_sig * xm.t());
  return mh;
}

static double log_bessel_k(double nu, double x){
  return ( log( R::bessel_k(abs(x), nu, 2.0) ) - abs(x) ); 
}

static double quadratic_form(arma::vec v, arma::mat Q){
  double result = ((arma::mat)(v.t() * Q * v)).at(0,0); 
  return abs(result); 
}

static double quadratic_form_2(arma::vec v, arma::vec v2, arma::mat Q){
  double result = ((arma::mat)(v.t() * Q * v2)).at(0,0); 
  return abs(result); 
}

struct infinite_loglik_except : std::exception 
{
  const char* what() const noexcept {return "logliklihood is infinite";}
}; 

// -----------------------------------------------------------------
// Gaussian density
// [[Rcpp::export]]
Rcpp::List dmg(arma::rowvec x, // vector 1 x p
               arma::rowvec mu,
               arma::mat Sig,
               bool LOG = false) {
  
  int p = x.n_elem;
  arma::mat inv_Sig = arma::inv(Sig);
  double log_det = log(arma::det(Sig));
  
  double constants = log(2.0) + log(M_PI);
  double constant_prop = -(p*0.5)*constants;

  constant_prop += -(0.5*log_det); 
  double mh_term = -0.5*mahalanobis(x,mu,inv_Sig); 
  
  double result;
  if (LOG == false){
    result = exp(constant_prop + mh_term); 
  } else {
    result = constant_prop + mh_term; 
  }
  
  return Rcpp::List::create(Rcpp::Named("result") = result);
}

// -----------------------------------------------------------------
// Generalized Hyperbolic density
// [[Rcpp::export]]
Rcpp::List dmgh(arma::vec x, // vector comes in as 1 x p  
                arma::vec mu,
                arma::vec alpha,
                arma::mat Sig, 
                double omega,
                double lambda,
                bool LOG = false) {
  
  int p = x.n_elem;
  
  arma::mat inv_Sig = arma::inv(Sig);
  double log_det = log(arma::det(Sig));
  
  const double nu = lambda - ((double)p) * 0.5;
  const double rho = quadratic_form(alpha, inv_Sig); 
  arma::vec xm = x - mu;
  
  double delta = quadratic_form(xm, inv_Sig);
  
  if (comparison(delta, 0.0)){
    delta = 1.0e-7; 
  }
  
  double bess_input = sqrt( (delta + omega)*(rho + omega) );
  
  double leading_terms = - ( ((double)p) * 0.5)*log(2.0*M_PI) - 0.5*log_det - log_bessel_k(lambda, omega); 
  double middle_terms = quadratic_form_2(alpha, xm, inv_Sig); 
  double third_term = (nu * 0.5)*( log(delta + omega) - log( rho + omega));
  double bessel_term = log_bessel_k(nu,bess_input);
  
  double result;
  if (LOG == false){
    result = exp(leading_terms + middle_terms +third_term + bessel_term); 
  } else {
    result = leading_terms + middle_terms +third_term + bessel_term; 
  }

  if(std::isnan(result) || std::isinf(result)){
    infinite_loglik_except e; 
    throw e;
  }

  return Rcpp::List::create(Rcpp::Named("result") = result);
  
}
// -----------------------------------------------------------------
// Skew-t density
// [[Rcpp::export]]
Rcpp::List dmst(arma::vec x, // vector comes in as 1 x p  
                arma::vec mu,
                arma::vec alpha,
                arma::mat Sig, 
                double v,
                bool LOG = false) {
  
  int p = x.n_elem;
  arma::mat inv_Sig = arma::inv(Sig);
  double log_det = log(arma::det(Sig));
  
  const double nu = (-v - p)/2.0;
  const double rho = arma::trace(inv_Sig*alpha*alpha.t()); 
  arma::vec xm = x - mu;
  double delta = arma::trace(inv_Sig*xm*xm.t());

  if( comparison(delta,0) ){
    delta = 0.0001;
  }
  
  double bess_input = sqrt( (delta + v)*(rho));
  double leading_terms = - (p/2.0)*log(2.0*M_PI) - 0.5*log_det + 0.5*v*log(v) - (v/2.0 - 1.0)*log(2.0) - boost::math::lgamma(v/2.0);  //
  double middle_terms = arma::trace(inv_Sig*(x-mu)*alpha.t());
  double third_term = (nu/2.0)*( log(delta + v) - log(rho));
  double bessel_term = log_bessel_k(nu,bess_input);
  
  if(std::isnan(bessel_term)){
    bessel_term = log(1e-10);
  }
  
  double result;
  if (LOG == false){
    result = exp(leading_terms + middle_terms +third_term + bessel_term); 
  } else {
    result = leading_terms + middle_terms +third_term + bessel_term; 
  }
  
  return Rcpp::List::create(Rcpp::Named("result") = result);
}
// -----------------------------------------------------------------
// Variance Gamma density
// [[Rcpp::export]]
Rcpp::List dmvg(arma::vec x, // vector 1 x p
                      arma::vec mu,
                      arma::vec alpha,
                      arma::mat Sig,
                      double gamma,
                      bool LOG = false) {
  
  int p = x.n_elem;
  arma::mat inv_Sig = arma::inv(Sig);
  double log_det = log(arma::det(Sig));
  
  double nu = gamma - p / 2.0;
  double rho = arma::trace(inv_Sig * alpha * alpha.t());
  arma::vec xm = x - mu;
  double delta = arma::as_scalar(xm.t() * inv_Sig * xm);
  
  if (comparison(delta, 0.0)) {
    delta = 1e-4;
  }
  
  double bess_input = std::sqrt(delta * (rho + 2.0 * gamma));
  double leading_terms = std::log(2.0) + gamma * std::log(gamma) - (p / 2.0) * std::log(2.0 * M_PI) - 0.5 * log_det;
  double middle_terms = arma::as_scalar(alpha.t() * inv_Sig * (x - mu)) - std::lgamma(gamma);
  double third_term = (nu / 2.0) * (std::log(delta) - std::log(rho + 2 * gamma));
  double bessel_term = log_bessel_k(nu, bess_input);
  
  if (std::isnan(bessel_term)) {
    bessel_term = std::log(1e-10);
  }
  
  double result;
  if (LOG == false){
    result = exp(leading_terms + middle_terms +third_term + bessel_term); 
  } else {
    result = leading_terms + middle_terms +third_term + bessel_term; 
  }
  
  return Rcpp::List::create(Rcpp::Named("result") = result);
}
