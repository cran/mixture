#include "GH_Mixture_Model.h"

#pragma once
class GH_EII: public GH_Mixture_Model {

public:
    using GH_Mixture_Model::GH_Mixture_Model;
    arma::mat lambda_sphere(arma::mat in_W,double in_n)
    {
        double lambda = arma::trace(in_W)/(in_n*p); 
        return lambda*arma::mat(p,p,arma::fill::eye); 
    }
    void m_step_sigs()  {
        
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
    
};

