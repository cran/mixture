
#include "GH_Mixture_Model.h"

#pragma once
class GH_EVV: public GH_Mixture_Model
{    

public:
    using GH_Mixture_Model::GH_Mixture_Model;

    void m_step_sigs(void)
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

};






