#include "GH_Mixture_Model.h"

#pragma once
class GH_VII: public GH_Mixture_Model {

public:

    using GH_Mixture_Model::GH_Mixture_Model;
    arma::mat lambda_sphere(arma::mat in_W,double in_n)
    {
        double lambda = arma::trace(in_W)/(in_n*p); 
        return lambda*arma::mat(p,p,arma::fill::eye); 
    }

    // VII MODEL 
    void m_step_sigs()
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
    
};

