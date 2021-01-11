
#pragma once
class VG_VVV: public VG_Mixture_Model {

public:
    using VG_Mixture_Model::VG_Mixture_Model;
    void m_step_sigs()  {
        
        for(int g = 0; g < G;  g++){
            try{
                sigs[g] = Ws[g];
                if( cond(Ws[g]) < 1.0e14){
                    inv_sigs[g] = arma::solve(Ws[g],EYE,arma::solve_opts::refine);
                }
                log_dets[g] = arma::log_det(Ws[g]).real();
            } catch(...){ }
        }
    }

};
