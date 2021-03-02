
// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#define ARMA_HAVE_STD_ISFINITE
#define ARMA_HAVE_STD_ISINF
#define ARMA_HAVE_STD_ISNAN
#define ARMA_HAVE_STD_SNPRINTF

// UNCOMMENT THE FOLLOWING IN FINAL PACKAGE. 
// #define ARMA_DONT_PRINT_ERRORS uncomment in actual package. 

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
//#include "RcppArmadillo.h"
//#include <Rcpp.h> 

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
#include "../Cluster_Error.hpp"

#pragma once
const double t_eps = 0.001; 

#pragma once
bool comparison_t(double a, double b){
    double tolerance  = abs( a - b);
    bool result = tolerance < t_eps;
    return (result);    
}

#pragma once 
std::default_random_engine generator;


// All dev has been done on vscode 
// GENERAL MIXTURE MODEL CLASS
#ifndef T_Mixture_Model_H
#define T_Mixture_Model_H
class T_Mixture_Model 
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
    void z_ig_init_random(void);

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
    T_Mixture_Model(arma::mat* in_data, int in_G, int model_id);
    // Deconstructor 
    virtual ~T_Mixture_Model();

    // General Methods
    double calculate_log_liklihood(void); // returns log
    double mahalanobis(arma::rowvec x, arma::rowvec mu, arma::mat inv_sig);  // calculates mh for specific term. 
    double log_density(arma::rowvec x, arma::rowvec mu, arma::mat inv_Sig, double log_det, double vg); // calculates a particular log-density for a specific group for a specific x 
    
    void E_step(); // performs the e-step calculation on the T_Mixture_Model, stores data in z_igs. 
    // stochastic methods
    void SE_step(void); // performs the stochastic estep .
    void RE_step(void);  
    void (T_Mixture_Model::*e_step)(void); 

    void set_E_step(bool stochastic){
        if(stochastic){
            this->e_step = &T_Mixture_Model::SE_step; 
        }
        else{
            // e_step is already regular. 
        }
    };


    // t distribution 
    std::vector<double> vgs; 
    std::vector<arma::vec> wgs; 
    void E_step_ws(void); 
    void M_step_vgs(void); 
    void M_step_vgs_constrained(void); 
    void M_step_vgs_regular(void); 
    void (T_Mixture_Model::*m_step_vgs)(void); 

    void set_m_step_vgs(bool constrained){
        if(constrained){
            this->m_step_vgs = &T_Mixture_Model::M_step_vgs_constrained; 
        }
        else{
            // m_step is already regular. 
        }
    };

    void M_step_props(); // calculate the proportions and n_gs
    void M_step_mus(); // calculates the m step for mean vector
    arma::mat mat_inverse(arma::mat X); // matrix inverse 
    void M_step_Ws(); // calculates both Wk and W
    virtual void m_step_sigs() { }; 
    virtual void set_defaults() {   };
    virtual void set_m_iterations(int in_iter, double in_tol) {  };
    virtual void set_m_iterations(int in_iter, double in_tol, arma::mat in_D) {  };

    void track_lg_init(void); // grab the first calculation of the logliklihood and replace the 1.0 from the constructor
    bool track_lg(bool check); // calculate current log liklihood of the model. If bool is check aitken convergence. 

    // Debug functions 
    void sig_eye_init(); // sets each of the groups covariance matrix to identity. 
};
#endif

// spherical family class 
#ifndef T_SPHERICAL_FAMILY_H
#define T_SPHERICAL_FAMILY_H
class T_Spherical_Family: public T_Mixture_Model 
{
    public:
        using T_Spherical_Family::T_Mixture_Model::T_Mixture_Model;
        arma::mat lambda_sphere(arma::mat in_W,double in_n); // general covariance matrix calculation for spherical family
        // constant identity matrix of size p for spherical family  
        const arma::mat eye_I = arma::mat(p,p,arma::fill::eye); 
};
#endif



#ifndef T_EII_H
#define T_EII_H
// EII: Equal volume family MODEL 1
class T_EII: public T_Spherical_Family
{
    public:
        using T_Spherical_Family::T_Spherical_Family;
        double lambda; // single volume parameter for all covariance matrices 
        void m_step_sigs(); // maximization step for EII model. 

};
#endif


#ifndef T_VII_H
#define T_VII_H
// T_VII: Equal volume family  MODEL 2
class T_VII: public T_Spherical_Family
{
    public:
        using T_Spherical_Family::T_Spherical_Family;
        void m_step_sigs(); // maximization step for EII model. 
};
#endif

// T_DIAGONAL FAMILY 
#ifndef T_DIAGONAL_FAMILY_H
#define T_DIAGONAL_FAMILY_H
class T_Diagonal_Family: public T_Mixture_Model
{
    public:
        using T_Mixture_Model::T_Mixture_Model;
};
#endif

// T_EII
#ifndef T_EEI_H
#define T_EEI_H // MODEL 3
class T_EEI: public T_Diagonal_Family
{
    public: 
        using T_Diagonal_Family::T_Diagonal_Family;
        void m_step_sigs(void); // Maximization step for T_EII model // see page 11 of Calveux 1993
};
#endif 

// T_VEI 
#ifndef T_VEI_H
#define T_VEI_H
class T_VEI: public T_Diagonal_Family // MODEL 4
{
    public: 
        using T_Diagonal_Family::T_Diagonal_Family;
        int m_iter_max; // number of iterations for iterative maximization see pg 11 for celeux
        double m_tol; 
        void m_step_sigs(void);
        void set_m_iterations(int in_iter, double in_tol);
        void set_defaults(void); 
};
#endif 

// T_EVI
#ifndef T_EVI_H
#define T_EVI_H
class T_EVI: public T_Diagonal_Family // MODEL 5
{
    public:
        using T_Diagonal_Family::T_Diagonal_Family;
        void m_step_sigs(void); 
};
#endif 
// T_VVI 
#ifndef T_VVI_H
#define T_VVI_H
class T_VVI: public T_Diagonal_Family // MODEL 6
{
    public:
        using T_Diagonal_Family::T_Diagonal_Family;
        void m_step_sigs(void); 
};
#endif 


class T_General_Family: public T_Mixture_Model
{
    public: 
        using T_Mixture_Model::T_Mixture_Model;

};


// EEE
#ifndef T_EEE_H
#define T_EEE_H
class T_EEE: public T_Mixture_Model // MODEL 7
{
    public:
        using T_Mixture_Model::T_Mixture_Model;
        void m_step_sigs(void);

};
#endif 

// VVV
#ifndef T_VVV_H
#define T_VVV_H
class T_VVV: public T_General_Family // MODEL 8
{
    public: 
        using T_General_Family::T_General_Family;
        void m_step_sigs(void); 
};
#endif



// MARK TO DO: GENERAL FAMILY 

// EEV
#ifndef T_EEV_H
#define T_EEV_H
class T_EEV: public T_General_Family // MODEL 9
{
    public:
        using T_General_Family::T_General_Family;
        void m_step_sigs(void); 
};
#endif


// VEV
#ifndef T_VEV_H
#define T_VEV_H
class T_VEV: public T_General_Family // MODEL 10
{
    public:
        using T_General_Family::T_General_Family;
        int m_iter_max; // number of iterations for iterative maximization see pg 8 of Celeux
        double m_tol; 
        void m_step_sigs(void); 
        void set_m_iterations(int in_iter, double in_tol);
        void set_defaults(void); 
};
#endif

// EVV
#ifndef T_EVV_H
#define T_EVV_H
class T_EVV: public T_General_Family // MODEL 11
{
    public: 
        using T_General_Family::T_General_Family; 
        void m_step_sigs(void); 
};
#endif


// VEE
#ifndef T_VEE_H
#define T_VEE_H
class T_VEE: public T_General_Family // MODEL 12
{
    public: 
        using T_General_Family::T_General_Family; 
        int m_iter_max; // number of iterations for iterative maximization see pg 8 of Celeux
        double m_tol; 
        void m_step_sigs(void); 
        void set_m_iterations(int in_iter, double in_tol);
        void set_defaults(void); 
};
#endif



// USING RYAN BROWNE AND MCNICHOLAS 2014
// EVE
#ifndef T_EVE_H
#define T_EVE_H
class T_EVE: public T_General_Family // MODEL 13
{
    public:
        using T_General_Family::T_General_Family;
        int m_iter_max; // number of matrix iterations 
        double m_tol; 
        arma::mat D;
        void m_step_sigs(void); 
        void set_m_iterations(int in_iter, double in_tol, arma::mat in_D);
        void set_defaults(void); 
};
#endif

// VVE
#ifndef T_VVE_H
#define T_VVE_H
class T_VVE: public T_General_Family // MODEL 14
{
    public:
        using T_General_Family::T_General_Family; 
        int m_iter_max; // number of matrix iterations 
        double m_tol; 
        arma::mat D;
        void m_step_sigs(void); 
        void set_m_iterations(int in_iter, double in_tol, arma::mat in_D);
        void set_defaults(void); 
};
#endif 


#include "General_Methods.hpp"
#include "Covariance_Methods.hpp"
#include "Missing_Data_Methods.hpp"


