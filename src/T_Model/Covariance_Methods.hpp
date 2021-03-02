#include "T_Mixture_Model.hpp"

// BEGINING OF FAMILY CLASSES AND SPECIFIED MODELS 
// SPHERICAL FAMILY 
#pragma once
arma::mat T_Spherical_Family::lambda_sphere(arma::mat in_W,double in_n)
{
  double lambda = arma::trace(in_W)/(in_n*p); 
  return lambda*eye_I; 
}
// T_EII
#pragma once
void T_EII::m_step_sigs()
{

  // calculate full W matrix
  arma::mat W = arma::mat(p,p,arma::fill::zeros); 
  for(int g = 0; g < G; g++){
    W += Ws[g]*n_gs[g];
  }

  // same covariance function for all groups
  arma::mat inter_mat = lambda_sphere(W,n);
  arma::mat inter_inv_mat = arma::solve(inter_mat,EYE,arma::solve_opts::refine); 
  // set all values for everyting 
  for(int g = 0; g < G; g++)
  {
    sigs[g] = inter_mat;
    inv_sigs[g] = inter_inv_mat;
    log_dets[g] = p*log(arma::trace(W)/((double)(n*p))); 
  }
}

// T_VII MODEL 
#pragma once
void T_VII::m_step_sigs()
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


// DIAGONAL FAMILY 
// EEI MODEL 
#pragma once
void T_EEI::m_step_sigs(void)
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


// VEI MODEL
#pragma once
void T_VEI::set_m_iterations(int in_iter, double in_tol)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
}

#pragma once
void T_VEI::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
} 

#pragma once
void T_VEI::m_step_sigs(void)
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


// EVI
#pragma once
void T_EVI::m_step_sigs(void){

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

// VVI 
#pragma once
void T_VVI::m_step_sigs(void)
{

  arma::mat S;
  for(int g = 0; g < G; g++){
    S = arma::diagmat(Ws[g]);
    sigs[g] = S;
    inv_sigs[g] = arma::solve(S,EYE);
    log_dets[g] = arma::sum(arma::log(S.diag()));  
  }

}; 


// EEE
#pragma once
void T_EEE::m_step_sigs(void)
{

  // calculate full W
  arma::mat W = arma::mat(p,p,arma::fill::zeros); 
  for(int g = 0; g < G; g++){
    W += Ws[g]*(double)(n_gs[g]/n);   
  }

  // invert it and set it to all 
  double log_det_W = arma::log_det(W).real(); 
  arma::mat invW = arma::solve(W,EYE,arma::solve_opts::refine);    
  for(int g = 0; g < G; g++){
    sigs[g] = W;
    inv_sigs[g] = invW; 
    log_dets[g] = log_det_W; 
  }
}

// VVV
void T_VVV::m_step_sigs(void)
{
  // Straight forward. 
  for(int g = 0; g < G; g++)
  {
    sigs[g] = Ws[g]; 
    inv_sigs[g] = arma::solve(Ws[g],EYE,arma::solve_opts::refine);
    log_dets[g] = arma::log_det(Ws[g]).real(); 
  }
}

// EEV
#pragma once
void T_EEV::m_step_sigs(void)
{
  std::vector<arma::mat> Ls_g(G); 
  std::vector<arma::mat> Omega_g(G);
  std::vector<arma::colvec> eigens(G);

  // exception thrower 
  sym_matrix_error e; 

  for(int g = 0; g < G; g++)
  {
    Ls_g[g] = arma::mat(p,p,arma::fill::zeros);
    Omega_g[g] = arma::mat(p,p,arma::fill::zeros);
    eigens[g] = arma::colvec(p,arma::fill::zeros);
  }

  arma::mat A = arma::mat(p,p,arma::fill::zeros);

  for(int g = 0; g < G; g ++)
  {
      if(!((Ws[g]*n_gs[g]).is_sympd())){
        throw e; 
      }
    arma::eig_sym(eigens[g], Ls_g[g], Ws[g]*n_gs[g],"std");
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

// VEV
#pragma once
void T_VEV::set_m_iterations(int in_iter, double in_tol)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
}

#pragma once
void T_VEV::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
} 

#pragma once
void T_VEV::m_step_sigs(void)
{

  // exception thrower 
  sym_matrix_error e; 

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
    if(!( (Ws[g]*pi_gs[g]).is_sympd() ))
    {
      throw e; 
    }
    arma::eig_sym(eigens[g], Ls_g[g], (Ws[g]*pi_gs[g]),"std");
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


// EVV
#pragma once
void T_EVV::m_step_sigs(void)
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


// VEE
#pragma once
void T_VEE::set_m_iterations(int in_iter, double in_tol)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
}

#pragma once
void T_VEE::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
} 

#pragma once
void T_VEE::m_step_sigs(void)
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
    inv_sigs[g] = arma::solve(S,EYE,arma::solve_opts::refine); 
    log_dets[g] =  p*std::log(lambdas[g]);
  }

}


// RYAN BROWNE ET MCNICHOLAS 2014 
// EVE 

#pragma once
void T_EVE::set_m_iterations(int in_iter, double in_tol, arma::mat in_D)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
    D = in_D; 
}

#pragma once
void T_EVE::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
  D = arma::eye(p,p); 
} 


#pragma once
void T_EVE::m_step_sigs(void){

  // exception thrower 
  sym_matrix_error e; 

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
      if(!( (W_temp_g[g]).is_sympd() ))
    {
      throw e;  
    }
    arma::eig_sym(eigens, L_g, W_temp_g[g],"std");
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
  arma::svd(U,s,V,interZ,"std" ); 
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
      if(!(W_temp_g[g]).is_sympd())
      {
        throw e; 
      }
      arma::eig_sym(eigens, L_g, W_temp_g[g],"std");
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

  double lam = 0.0; // volume 
  for(int g = 0; g < G; g++)
  {
    arma::mat DAW = D * arma::diagmat(1/A_gs[g]) * D.t() * W_temp_g[g];
    lam +=  arma::sum(DAW.diag()); 
  }
  lam = lam/p; 

  for(int g = 0; g < G; g++)
  {
    sigs[g] = D * arma::diagmat(lam*A_gs[g]) * D.t();
    inv_sigs[g] = D * arma::diagmat((1.0/lam) * (1.0/A_gs[g]))* D.t(); 
    log_dets[g] = p*std::log(lam);  
    }  

}




// RYAN BROWNE ET MCNICHOLAS 2014 
// EVE 
#pragma once
void T_VVE::set_m_iterations(int in_iter, double in_tol, arma::mat in_D)
{     
    m_iter_max = in_iter; // number of iterations for optimizing matrix 
    m_tol = in_tol; // set tolerance          
    D = in_D; 
}
#pragma once
void T_VVE::set_defaults(void){
  m_iter_max = 20; // default 20 max 
  m_tol = 1e-8; // default tolerance 
  D = arma::eye(p,p); 
} 


#pragma once
void T_VVE::m_step_sigs(void){

  // exception thrower 
  sym_matrix_error e; 

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
    if(!(W_temp_g[g]).is_sympd())
      {
        throw e;
      }
    arma::eig_sym(eigens, L_g, W_temp_g[g],"std");
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
      if(!(W_temp_g[g].is_sympd())){
        throw e; 
      }
      arma::eig_sym(eigens, L_g, W_temp_g[g],"std");
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
