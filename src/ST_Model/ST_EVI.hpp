
#include "ST_Mixture_Model.h"

#pragma once
class ST_EVI: public ST_Mixture_Model
{
    
public:
    using ST_Mixture_Model::ST_Mixture_Model;

    void m_step_sigs(void)
    {

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

};






