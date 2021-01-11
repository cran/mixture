

#pragma once
class VG_VEV: public VG_Mixture_Model
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

        // ************** INITIALIZATION STEP **************
        std::vector<arma::mat> Ls_g(G); 
        std::vector<arma::mat> Omega_g(G);
        std::vector<arma::colvec> eigens(G);
        std::vector<double> lambdas(G); 

        for(int g = 0; g < G; g++)
        {
            Ls_g[g] = arma::mat(p,p,arma::fill::zeros);
            Omega_g[g] = arma::mat(p,p,arma::fill::zeros);
            eigens[g] = arma::colvec(p,arma::fill::zeros);
        }

        arma::mat A = arma::mat(p,p,arma::fill::zeros);

        for(int g = 0; g < G; g ++)
        {
            arma::eig_sym(eigens[g], Ls_g[g], (Ws[g]*pi_gs[g]));
            Omega_g[g].diag() = eigens[g];
        }

        // calculate lambda 
        for(int g = 0; g < G; g++ )
        {
            // first lambda step 
            lambdas[g] = arma::trace(Omega_g[g])/((double)(pi_gs[g]*p)); 
        }

        // calculate A
        for(int g = 0; g < G;  g++)
        {
            A += Omega_g[g]/lambdas[g];
        }  
        double denom = pow(arma::det(A),(double)(1.0/p));
        A = arma::diagmat(A/denom);   
        arma::mat invA = arma::solve(A,EYE,arma::solve_opts::refine);

        for(int g = 0; g < G; g++ )
        {
            lambdas[g] = arma::trace(Omega_g[g]*invA)/((double)(pi_gs[g]*p));
        }

        // calculate log lik for this portion only. 
        double first_val = 0.0; 
        for(int g = 0; g < G; g++){
            first_val +=  (1 + log(lambdas[g]))*pi_gs[g]*((double)p);
        }

        // intialize convergence ary for tolerance check, make one of them infinity. 
        double conv_ary [2] = {first_val,std::numeric_limits<double>::infinity()};

        // ************* END OF INITIALIZATION STEP *******
        int iter = 1;
        // iterate while its less then m_iter_max, but need to add convergence check from mixture... 
        while ( (iter < m_iter_max) &&  abs(conv_ary[1] - conv_ary[0]) > m_tol )
        {
            A = arma::mat(p,p,arma::fill::zeros);
            // calculate A
            for(int g = 0; g < G;  g++)
            {
            A += Omega_g[g]/lambdas[g];
            }  
            double denom = pow(arma::det(A),(double)(1.0/p));
            A = arma::diagmat(A/denom);   
            arma::mat invA = arma::solve(A,EYE,arma::solve_opts::refine);

            for(int g = 0; g < G; g++ )
            {
            lambdas[g] = arma::trace(Omega_g[g]*invA)/((double)(pi_gs[g]*p));
            }

            first_val = 0.0;   
            for(int g = 0; g < G; g++){
            first_val +=  (1 + log(lambdas[g]))*pi_gs[g]*((double)p);
            }

            conv_ary[1] = conv_ary[0];
            conv_ary[0] = first_val; 
            iter++; 
        }

        arma::mat S;
        for(int g = 0; g < G; g++)
        {
            S = lambdas[g]*((Ls_g[g]) * A * Ls_g[g].t());
            sigs[g] = S; 
            inv_sigs[g] = arma::solve(S,EYE,arma::solve_opts::refine); 
            log_dets[g] = p*std::log(lambdas[g]);
        }

    }



};





