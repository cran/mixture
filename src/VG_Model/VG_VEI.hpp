

#pragma once
class VG_VEI: public VG_Mixture_Model
{
    
public:
    using VG_Mixture_Model::VG_Mixture_Model;
    int m_iter_max; // number of iterations for iterative maximization see pg 11 for celeux
    double m_tol;


    void set_m_iterations(int in_iter, double in_tol)
    {
        m_iter_max = in_iter; // number of iterations for optimizing matrix 
        m_tol = in_tol; // set tolerance          
    }

    void set_defaults(void)
    {
        m_iter_max = 20; // default 20 max 
        m_tol = 1e-8; // default tolerance 
    }

    void m_step_sigs(void)
    {
        // ****  INTIALIZATION STEP BEFORE ITERATIVE METHOD ****
        // Some of the variable models, we need to solve for the sigma matrix iteratively,
        // there is a loop within the EM loop for some of the models, this is one of them. 

        // intialize empty matrix for place holder
        arma::mat S = arma::mat(p,p,arma::fill::eye); 
        // Set up identity in the B matrix .
        arma::mat B = arma::mat(p,p,arma::fill::eye);

        // set up lambdas 
        arma::rowvec lambdas = arma::rowvec(G,arma::fill::zeros);
        // calculate lambdas
        for (int g = 0; g < G; g++)
        {
            lambdas[g] = arma::trace(Ws[g])/(p);
        }


        // Get full scatter matrix 
        arma::mat W = arma::mat(p,p,arma::fill::zeros); 
        for(int g = 0; g < G; g++){
            W += Ws[g]*((double)(n_gs[g]/lambdas[g])/(n));
        }

        // set B to be the diagonal of W 
        arma::mat inter_W = arma::diagmat(W); 
        double denom = pow( arma::det(inter_W), (double)(1.0/p) );

        B = inter_W / denom;
        arma::mat invB = arma::solve(B,EYE,arma::solve_opts::refine);

        // calculate new lambda 
        for (int g = 0; g < G; g ++ ){
            lambdas[g] =  arma::trace( Ws[g]*invB )/(p);
        }

        // calculate tolerance, and iterative method check 
        double first_val = 0.0;
        for(int g = 0; g < G; g ++){
            first_val += pi_gs[g]*(1 + log(lambdas[g]));
        }
        first_val = first_val*p; 

        // intialize convergence ary for tolerance check, make one of them infinity. 
        double conv_ary [2] = {first_val,std::numeric_limits<double>::infinity()};

        // **** END OF INITIALIZATION STEP *******
        
        int iter = 1; 
        // iterate while its less then m_iter_max, but need to add convergence check from mixture... 
        while ( (iter < m_iter_max) &&  std::abs(conv_ary[1] - conv_ary[0]) > m_tol )
        {

            // Get the full flurry matrix 
            W = arma::mat(p,p,arma::fill::zeros); 
            for(int g = 0; g < G; g++){
            W += Ws[g]*((double)(n_gs[g]/lambdas[g])/(n));
            }
            
            // get diagonal and denominator for calculating B
            inter_W = arma::diagmat(W); 
            denom = pow( arma::det(inter_W), (double)(1.0/p) );
            B = inter_W / denom;
            invB = arma::solve(B,EYE,arma::solve_opts::refine);

            // calculate new lambda 
            for (int g = 0; g < G; g ++ ){
            lambdas[g] =  arma::trace(Ws[g]*invB )/(p);
            }
            
            // calculate tolerance, and iterative method check 
            first_val = 0.0;
            for(int g = 0; g < G; g ++){
            first_val += pi_gs[g]*(1 + std::log(lambdas[g]));
            }
            first_val = first_val*p; 

            conv_ary[1]  = conv_ary[0]; 
            conv_ary[0] = first_val; 

            iter++;
        }
        
        for(int g = 0; g < G; g++){
            S = lambdas[g]*B;
            sigs[g] = S; 
            inv_sigs[g] = arma::solve(S,EYE,arma::solve_opts::refine); 
            log_dets[g] =  p*std::log(lambdas[g]); 
        }

    }

};






