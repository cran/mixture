
#include "ST_Mixture_Model.h"

#pragma once
class ST_EEV: public ST_Mixture_Model
{    

public:
    using ST_Mixture_Model::ST_Mixture_Model;

    void m_step_sigs(void)
    {

        std::vector<arma::mat> Ls_g(G); 
        std::vector<arma::mat> Omega_g(G);
        std::vector<arma::colvec> eigens(G);

        for(int g = 0; g < G; g++)
        {
            Ls_g[g] = arma::mat(p,p,arma::fill::zeros);
            Omega_g[g] = arma::mat(p,p,arma::fill::zeros);
            eigens[g] = arma::colvec(p,arma::fill::zeros);
        }

        arma::mat A = arma::mat(p,p,arma::fill::zeros);

        for(int g = 0; g < G; g ++)
        {
            arma::eig_sym(eigens[g], Ls_g[g], Ws[g]*n_gs[g]);
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

};






