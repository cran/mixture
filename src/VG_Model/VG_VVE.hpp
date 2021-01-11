


#pragma once
class VG_VVE: public VG_Mixture_Model
{    

public:
    using VG_Mixture_Model::VG_Mixture_Model;

    int m_iter_max; // number of iterations for iterative maximization see pg 8 of Celeux
    double m_tol;
    arma::mat D;

    // VEV
    void set_m_iterations(int in_iter, double in_tol, arma::mat in_D)
    {     
        m_iter_max = in_iter; // number of iterations for optimizing matrix 
        m_tol = in_tol; // set tolerance          
        D = in_D;   
    }

    void set_defaults(void){
        m_iter_max = 20; // default 20 max 
        m_tol = 1e-8; // default tolerance 
        D = arma::eye(p,p);
    } 


    void m_step_sigs(void)
    {

        // calculate full Flurry matrix
        arma::mat W = arma::mat(p,p,arma::fill::zeros);
        for(int g =0; g < G; g++)
        {
            W += Ws[g]*pi_gs[g];
        }

        // get eigen values and noramlize them, store them in column vectors. 
        std::vector<arma::colvec> A_gs(G);
        // go through groups 
        for(int g = 0; g < G; g++)
        {
            // eigen value calc
            arma::mat tempA_g = D.t() * Ws[g] * D * pi_gs[g];
            A_gs[g] = tempA_g.diag().as_col();
            // calculate denominator for normalization. 
            double denom = pow(arma::prod(A_gs[g]), ((double)((1.0)/((A_gs[g].size())))) );
            A_gs[g] = A_gs[g]/denom;
        }

        // ******** BEGIN INTIALIZATION of MM *************** 

        // MM pt. 1
        arma::mat interZ = arma::mat(p,p,arma::fill::zeros); // holds interZ matrix. 
        double lambda_g; // maximum eigen value placeholder
        arma::colvec eigens; // eigen values placeholder
        arma::mat L_g; // eigen vectors
        arma::mat ADK; // left hand side multiplicaiton matrix placeholder
        std::vector<arma::mat> W_temp_g(G);

        for(int g = 0; g < G; g++)
        {
            //arma::eig_sym
            W_temp_g[g] = Ws[g]*pi_gs[g];
            arma::eig_sym(eigens, L_g, W_temp_g[g]);
            lambda_g = arma::max(eigens); 
            ADK = arma::diagmat(1.0/A_gs[g]) * D.t();
            interZ += (ADK * W_temp_g[g]) - (lambda_g *ADK);  
        }

        // svd calculation 
        arma::mat U; 
        arma::mat V; 
        arma::vec s; 
        arma::svd(U,s,V,interZ,"std"); 
        D = V * U.t(); 

        // reset 
        interZ = arma::mat(p,p,arma::fill::zeros); 

        // MM pt 2. 
        for(int g = 0; g < G; g++)
        {
            lambda_g = arma::max(1.0/A_gs[g]);
            interZ +=  W_temp_g[g] * D * arma::diagmat(1.0/A_gs[g]) - lambda_g * (W_temp_g[g] * D); 
        }

        // calculate svd and set new D. 
        arma::svd(U,s,V,interZ,"std"); 
        D = V * U.t(); 
        D = D.t(); 

        double first_val = 0.0; 

        // checking convergence
        arma::vec tempz = arma::vec(G,arma::fill::zeros); 
        for(int g = 0; g < G; g++)
        {
            arma::mat DWA =  (D.t() * W_temp_g[g] * D * arma::diagmat( 1.0 / A_gs[g])) ; 
            tempz[g] = arma::sum(DWA.diag());
        }

        first_val = arma::sum(tempz);

        // *********** END OF INTIALIZATION FOR MM ************

        // intialize convergence ary for tolerance check, make one of them infinity. 
        double conv_ary [2] = {first_val,std::numeric_limits<double>::infinity()};

        int iter = 1; 
        // iterate while its less then m_iter_max, but need to add convergence check from mixture... 
        while ( (iter < m_iter_max) &&  (abs(conv_ary[1] - conv_ary[0])) > m_tol )
        {


            // CALCULATE NEW A_gs 
            for(int g = 0; g < G; g++)
            {
            // eigen value calc
            arma::mat tempA_g = D.t() * W_temp_g[g]* D ; 
            A_gs[g] = tempA_g.diag().as_col();
            // calculate denominator for normalization. 
            double denom = pow(arma::prod(A_gs[g]), ((double)((1.0)/((A_gs[g].size())))) );
            A_gs[g] = A_gs[g]/denom;
            }
            // calculate new D 
            
            // MM pt. 1
            arma::mat interZ = arma::mat(p,p,arma::fill::zeros); // holds interZ matrix. 
            double lambda_g; // maximum eigen value placeholder
            arma::colvec eigens; // eigen values placeholder
            arma::mat L_g; // eigen vectors
            arma::mat ADK; // left hand side multiplicaiton matrix placeholder
            std::vector<arma::mat> W_temp_g(G);

            for(int g = 0; g < G; g++)
            {
            //arma::eig_sym
            W_temp_g[g] = Ws[g]*pi_gs[g];
            arma::eig_sym(eigens, L_g, W_temp_g[g]);
            lambda_g = arma::max(eigens); 
            ADK = arma::diagmat(1.0/A_gs[g]) * D.t();
            interZ += (ADK * W_temp_g[g]) - (lambda_g *ADK);  
            }

            // svd calculation 
            arma::mat U; 
            arma::mat V; 
            arma::vec s; 
            arma::svd(U,s,V,interZ,"std"); 
            D = V * U.t(); 

            // reset 
            interZ = arma::mat(p,p,arma::fill::zeros); 

            // MM pt 2. 
            for(int g = 0; g < G; g++)
            {
            lambda_g = arma::max(1.0/A_gs[g]);
            interZ +=  W_temp_g[g] * D * arma::diagmat(1.0/A_gs[g]) - lambda_g * (W_temp_g[g] * D); 
            }

            // calculate svd and set new D. 
            arma::svd(U,s,V,interZ,"std"); 
            D = V * U.t(); 
            D = D.t(); 

            // check convergence 
            for(int g = 0; g < G; g++)
            {
            arma::mat DWA =  (D.t() * W_temp_g[g] * D * arma::diagmat( 1.0 / A_gs[g])) ; 
            tempz[g] = arma::sum(DWA.diag());
            }
            first_val = arma::sum(tempz);

            conv_ary[1] = conv_ary[0]; 
            conv_ary[0] = first_val; 

            iter++; 

        }


        arma::vec lam = arma::vec(G,arma::fill::zeros); // volume 
        for(int g = 0; g < G; g++)
        {
            arma::mat DAW = D * arma::diagmat(1/A_gs[g]) * D.t() * Ws[g];
            lam[g] =  arma::sum(DAW.diag())/p; 
        }


        for(int g = 0; g < G; g++)
        {
            sigs[g] = D * arma::diagmat(lam[g]*A_gs[g]) * D.t();
            inv_sigs[g] = D * arma::diagmat((1.0/lam[g]) * (1.0/A_gs[g]))* D.t(); 
            log_dets[g] = arma::log_det(sigs[g]).real(); 
        }  



    }




};





