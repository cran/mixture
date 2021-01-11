
#include "GH_Mixture_Model.h"

#pragma once
class GH_EEE: public GH_Mixture_Model
{    

public:
    using GH_Mixture_Model::GH_Mixture_Model;

    void m_step_sigs(void)
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

};






