
#pragma once
class VG_EEI: public VG_Mixture_Model
{
public:
    using VG_Mixture_Model::VG_Mixture_Model;
    // Maximization step for EEI model // see page 11 of Calveux 1993

    // DIAGONAL FAMILY 
    // EEI MODEL 
    void m_step_sigs(void)
    {
        arma::mat S = arma::mat(p,p,arma::fill::eye); 
        // Set up identity in the B matrix .
        arma::mat B = arma::mat(p,p,arma::fill::eye);

        // Get full scatter matrix 
        arma::mat W = arma::mat(p,p,arma::fill::zeros); 
        for(int g = 0; g < G; g++){
            W += Ws[g]*n_gs[g];
        }

        // set B to be the diagonal of W 
        B.diag() = W.diag();
        // expression simplies down to just this. 
        S = B/n; 
        arma::mat Sinv = arma::solve(S,EYE,arma::solve_opts::refine); 

        for(int g = 0; g < G; g++) {
            sigs[g] = S;
            inv_sigs[g] = Sinv; 
            log_dets[g] = arma::sum(arma::log(S.diag())); 
        }

    }

};

