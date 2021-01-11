

#pragma once
class VG_VEE: public VG_Mixture_Model
{    

public:
    using VG_Mixture_Model::VG_Mixture_Model;

    int m_iter_max; // number of iterations for iterative maximization see pg 8 of Celeux
    double m_tol;

    // VEV
    void set_m_iterations(int in_iter, double in_tol)
    {     
        m_iter_max = in_iter; // number of iterations for optimizing matrix 
        m_tol = in_tol; // set tolerance          
    }

    void set_defaults(void){
    m_iter_max = 20; // default 20 max 
    m_tol = 1e-8; // default tolerance 
    } 



    void m_step_sigs(void)
    {
      // ****** BEGINNING OF INITIALIZATION ******

        std::vector<arma::mat> W_k(G); 
        arma::mat W = arma::mat(p,p,arma::fill::zeros); 

        for(int g = 0; g < G; g++)
        {
            W_k[g] = Ws[g]*pi_gs[g];
            W += W_k[g];
        }

        double denom = pow(arma::det(W),((double)((1.0)/p))); 
        arma::mat C = W/denom; 
        arma::mat invC = arma::solve(C,EYE,arma::solve_opts::refine); 

        // calculate new lambdas
        std::vector<double> lambdas(G); 
        for (int g = 0; g < G; g ++ ){
            lambdas[g] =  arma::trace( Ws[g]*invC )/(p);
        }

        // calculate log lik for this portion only. 
        double first_val = 0.0; 
        for(int g = 0; g < G; g++){
            first_val +=  ( (arma::trace( Ws[g]*invC ))/(lambdas[g]) + p*(pi_gs[g]*log(lambdas[g])));
        }

        // intialize convergence ary for tolerance check, make one of them infinity. 
        double conv_ary [2] = {first_val,std::numeric_limits<double>::infinity()};

        // **** END OF INITIALIZATION STEP *******
        
        int iter = 1; 
        // iterate while its less then m_iter_max, but need to add convergence check from mixture... 
        while ( (iter < m_iter_max) &&  (abs(conv_ary[1] - conv_ary[0])) > m_tol )
        {
            C = arma::mat(p,p,arma::fill::zeros); 
            for(int g = 0; g < G; g++)
            {
            C += W_k[g]/(lambdas[g]);
            }
            denom = pow(arma::det(C), (double)(1.0/p)); 
            C = C/denom; 
            invC =  arma::solve(C,EYE,arma::solve_opts::refine); 

            for (int g = 0; g < G; g ++ )
            {
            lambdas[g] =  arma::trace( Ws[g]*invC )/(p);
            }

            first_val = 0.0; 
            for(int g = 0; g < G; g++)
            {
            first_val +=  ( (arma::trace( Ws[g]*invC ))/(lambdas[g]) + p*(pi_gs[g]*std::log(lambdas[g])));
            }

            conv_ary[1] = conv_ary[0]; 
            conv_ary[0] = first_val; 
            iter++; 
        }

        arma::mat S = arma::mat(p,p,arma::fill::zeros); 
        for(int g = 0; g < G; g++){
            S = lambdas[g]*C;
            sigs[g] = S; 
            inv_sigs[g] = arma::solve(S,EYE); 
            log_dets[g] =  p*std::log(lambdas[g]);
        }

    }



};





