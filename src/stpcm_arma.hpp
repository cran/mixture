

// create model function, generates a model pointer and returns it as a general mixture_model pointer
ST_Mixture_Model* st_create_model(arma::mat* Xp,int G, int model_id, int model_type)
{
 switch(model_type)
 {
   case 0: {
      ST_EII* m = new ST_EII(Xp, G, model_id);
      return m;
   }
   case 1:{
      ST_VII* m = new ST_VII(Xp, G, model_id);
      return m;
   }
   case 2:{
      ST_EEI* m = new ST_EEI(Xp, G, model_id); 
      return m;
   }
   case 3:{
      ST_EVI* m = new ST_EVI(Xp,G,model_id);
     return m; 
   }
   case 4:{
      ST_VEI* m = new ST_VEI(Xp,G,model_id); 
      return m; 
   }
   case 5:{
      ST_VVI* m = new ST_VVI(Xp,G,model_id); 
      return m; 
   }
   case 6:{
      ST_EEE* m = new ST_EEE(Xp,G,model_id); 
      return m; 
   }
   case 7:{
      ST_VEE* m = new ST_VEE(Xp, G, model_id); 
      return m; 
   }
   case 8:{
      ST_EVE* m = new ST_EVE(Xp,G,model_id); 
      return m; 
   }
   case 9:{
      ST_EEV* m = new ST_EEV(Xp,G,model_id); 
      return m; 
   }
   case 10:{
      ST_VVE* m = new ST_VVE(Xp,G,model_id); 
      return m; 
   }
   case 11:{
      ST_EVV* m = new ST_EVV(Xp,G, model_id); 
      return m; 
   }
   case 12:{
      ST_VEV* m = new ST_VEV(Xp, G, model_id); 
      return m; 
   }
   default:{ 
      ST_VVV* m = new ST_VVV(Xp, G, model_id);
      return m;
   }
 }
}

#pragma once
void set_model_defaults_st(std::unique_ptr<ST_Mixture_Model> & m, 
                        int model_type,
                        int in_m_iter_max,
                        double in_m_tol )
{
  
  switch (model_type)
    {
      case 4:{
        if(in_m_iter_max != 0){
          m->set_m_iterations(in_m_iter_max,in_m_tol);
        }
        else{
          m->set_defaults();
        } 
        break;    
      }
      case 12:{
        if(in_m_iter_max != 0){
          m->set_m_iterations(in_m_iter_max,in_m_tol);
        }
        else{
          m->set_defaults();
        } 
        break; 
      }
      case 7:{
        if(in_m_iter_max != 0){
          m->set_m_iterations(in_m_iter_max,in_m_tol);
        }
        else{
          m->set_defaults();
        } 
        break; 
      }
      case 8:{
        if(in_m_iter_max != 0){
          m->set_m_iterations(in_m_iter_max,in_m_tol,arma::mat(m->p,m->p,arma::fill::eye));
        }
        else{
          m->set_defaults();
        } 
        break; 
      }
      case 10:{
        if(in_m_iter_max != 0){
          m->set_m_iterations(in_m_iter_max,in_m_tol,arma::mat(m->p,m->p,arma::fill::eye));
        }
        else{
          m->set_defaults();
        } 
      }
      default:
        break;
    }

}

Rcpp::List create_result_list_st(std::unique_ptr<ST_Mixture_Model> & m, bool check_na_)
/*
  Creates the return list object that is pushed out to the R from C++. 
*/
{


  Rcpp::List result_list = Rcpp::List::create(Rcpp::Named("mus") = m->mus, 
                                          Rcpp::Named("alphas") = m->alphas, 
                                          Rcpp::Named("sigs") = m->sigs,
                                          Rcpp::Named("G") = m->G, 
                                          Rcpp::Named("vs") = m->vs,
                                          Rcpp::Named("zigs") = m->zi_gs,
                                          Rcpp::Named("pi_gs") = m->pi_gs,
                                          Rcpp::Named("n_gs") = m->n_gs,
                                          Rcpp::Named("log_dets") = m->log_dets,
                                          Rcpp::Named("logliks") = m->logliks);

  if(check_na_){
    result_list["X"] = m->data.t(); 
  }

  return result_list; 
}



// WRAPPERS 

// [[Rcpp::export]]
Rcpp::List main_loop_st(arma::mat X, // data 
                     int G, int model_id, // number of groups and model id (id is for parrallel use)
                     int model_type,  // covariance model type
                     arma::mat in_zigs, // group memberships from initialization 
                     int in_nmax, // iteration max for EM . 
                     double in_l_tol, // liklihood tolerance 
                     int in_m_iter_max, // iteration max for M step for special models 
                     double in_m_tol, // tolerance for matrix convergence on M step for special models.
                     arma::vec anneals, // for deterministic annealing 
                     std::string latent_step = "standard", // e step method. 
                     int t_burn = 5// number of burn in steps for NAs if found. 
                     )
{
  
  int stochastic_check = 0; 
  // for stochastic variant
  if (19 < model_type ){
    model_type -= 20;  
    stochastic_check = 1;     
  }
  

  // create mixture model class. 
  std::unique_ptr<ST_Mixture_Model> m = std::unique_ptr<ST_Mixture_Model>(st_create_model(&X,G,model_id,model_type));  

  if( model_id == 2){
    stochastic_check = 2; 
    // reconstruct the label vector on this side and use it as a guide.
    int k = 0;  
    for(int i = 0; i < m->n; i++){
            for( k = 0; k < G; k++){
              if(in_zigs.at(i,k) == 5){
                m->semi_labels.at(i) = k + 1;
                in_zigs.at(i,k) = 1.0; 
              }
            }
    }
  }


  m->set_E_step(stochastic_check); 

  // check latent step and assign it. 
  if(!latent_step.compare("random")){
    // assign latent step. 
    // Rcpp::Rcout << "Assigning random latent method. \n" ;
    m->e_step_latent = &ST_Mixture_Model::SE_step_latent;

  } // in all other cases use a standard sampler. 

  if(isnan(in_l_tol)){
    m->tol_l = 1e-6;
  }
  else{
    m->tol_l = in_l_tol;
  }

  gsl_set_error_handler_off();

  // Intialize 
  m->zi_gs = in_zigs;

  // check for nas. 
  bool NA_check; 
  m->init_missing_tags();
  NA_check = ( m->col_tags.size() > 0); 


  // wrap iterations up in a try catch just in case something bad happens. 
  try
  {
        // perform missing data check and implement algorithm based on check 
    if(NA_check){

      set_model_defaults_st(m, model_type, in_m_iter_max, in_m_tol); 
      // phase 1
      m->EM_burn(t_burn); // defaults already set in here, although I should pull them out. in full function. 
      // phase 2. 
      m->impute_init();
      m->M_step_props();
      // phase 3.  
      m->E_step_latent(); 
      m->M_step_mus();
      m->M_step_Ws(); 
      m->set_defaults();
      m->m_step_sigs(); 
      m->M_step_gamma(); 
      m->track_lg_init(); 

      arma::uword nmax = (arma::uword)in_nmax; 
      bool convergence_check = false; 
      // main EM with extra setp. 
      for(arma::uword iter = 0; iter < nmax ; iter++)
      {
        if(iter < anneals.n_elem)
        {
          m->nu_d = anneals[iter]; 
        }
        else{
          m->nu_d = 1.0; 
        }

        m->E_step(); 
        m->M_step_props();
        m->E_step_latent();
        m->impute_cond_mean(); // now have imputation step. 
        m->M_step_mus();
        m->M_step_Ws(); 
        m->m_step_sigs(); 
        m->M_step_gamma(); 

        convergence_check = m->track_lg(iter < 5);
        if(convergence_check){
          // Rcpp::Rcout  << "Converged at Iteration " << iter << std::endl;  
          break; 
        }
      }

    }
    else{

      set_model_defaults_st(m, model_type, in_m_iter_max, in_m_tol); 

      // perform intialization of params. 
      m->M_step_props(); 
      m->M_step_init_gaussian(); 
      m->track_lg_init(); 
      m->E_step(); 
      m->M_step_props(); 
      m->E_step_latent();
      m->M_step_props();
      m->M_step_mus();
      m->M_step_Ws(); 
      m->m_step_sigs(); 
      m->M_step_gamma();  
      m->track_lg(false);     

      arma::uword nmax = (arma::uword)in_nmax; 
      bool convergence_check = false; 

      for(arma::uword iter = 0; iter < nmax ; iter++)
      {
        if(iter < anneals.n_elem)
        {
          m->nu = anneals[iter]; 
        }
        else{
          m->nu = 1.0; 
        }
        
        // set the previous state. 
        m->set_previous_state(); 

        m->E_step(); 
        m->M_step_props(); 
        m->E_step_latent();
        m->M_step_mus();
        m->M_step_Ws(); 
        m->m_step_sigs(); 
        m->M_step_gamma(); 

        m->check_decreasing_loglik(&iter, nmax); 

        convergence_check = m->track_lg(iter < 5);
        if(convergence_check){
          // Rcpp::Rcout  << "Converged at Iteration " << iter << std::endl;  
          break; 
        }
      }
    }
  }
    catch(const std::exception& e)
  {

    // Rcpp::Rcout << "Error " << e.what() << " \n"; 
    // check for bad logliklihood. 
    if(is_string_comparison(e,"logliklihood was infinite, back to previous step and returned results"))
    {
      Rcpp::List ret_val = create_result_list_st(m, NA_check); 
      return ret_val;
    }

    return Rcpp::List::create(Rcpp::Named("Error") = e.what()); 
  }

    // create return object and exit c++. 
    Rcpp::List ret_val = create_result_list_st(m, NA_check); 
    return ret_val;
}






// [[Rcpp::export]]
Rcpp::List st_e_step_internal(arma::mat X, // data 
                           int G, int model_id, // number of groups and model id (id is for parrallel use)
                           int model_type,  // covariance model type
                           Rcpp::List in_m_obj, // internal object from output
                           arma::mat init_zigs, 
                           double in_nu = 1.0)
{
  // declare params. that are passed in from the internal object.  
  std::vector< arma::rowvec > mus_t = in_m_obj["mus"]; 
  std::vector< arma::rowvec > alphas_t = in_m_obj["alphas"]; 
  std::vector< arma::mat > sigs = in_m_obj["sigs"];
  std::vector<double> vgs = in_m_obj["vgs"];
  std::vector<double> n_gs = in_m_obj["n_gs"];
  std::vector<double> log_dets = in_m_obj["log_dets"];
  arma::rowvec pi_gs_t = in_m_obj["pi_gs"];

  // flip 
  std::vector< arma::colvec > mus; 
  std::vector< arma::colvec > alphas; 

  for(int g = 0; g < G; g++)
  {
    mus.push_back( (arma::vec)((arma::mat)mus_t[g]).t() ); 
    alphas.push_back( (arma::vec)((arma::mat)alphas_t[g]).t() ); 
  }


  arma::vec pi_gs = (arma::vec)(((arma::mat)pi_gs_t).t()); 


  // create model and set existing parameters. 
  X = X.t(); 
  std::unique_ptr<ST_Mixture_Model> m = std::unique_ptr<ST_Mixture_Model>(st_create_model(&X,G,model_id,model_type));  
  m->mus = mus;
  m->alphas = alphas;  
  m->sigs = sigs; 
  m->log_dets = log_dets; 
  m->pi_gs = pi_gs; 
  m->vs = vgs; 
  m->n_gs = n_gs; 
  m->zi_gs = init_zigs; 
  m->init_missing_tags(); //Additional graphical and num

  // invert symmetric matrices. 
  for(int g = 0; g < G; g++)
  {
    m->inv_sigs[g] = arma::inv_sympd(sigs[g]); 
  }

  // perform e_step and imputation. 
  m->E_step_only_burn(); 

  Rcpp::List ret_val = Rcpp::List::create(Rcpp::Named("X") = m->data.t(),
                                            Rcpp::Named("col_tags") = m->col_tags, 
                                            Rcpp::Named("origX") = X,
                                            Rcpp::Named("zigs") = m->zi_gs); 

  
  return ret_val;
}










