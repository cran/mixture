 #include <string> 

// create model function, generates a model pointer and returns it as a general mixture_model pointer
VG_Mixture_Model* vg_create_model(arma::mat* Xp,int G, int model_id, int model_type)
{
 switch(model_type)
 {
   case 0: {
      VG_EII* m = new VG_EII(Xp, G, model_id);
      return m;
   }
   case 1:{
      VG_VII* m = new VG_VII(Xp, G, model_id);
      return m;
   }
   case 2:{
      VG_EEI* m = new VG_EEI(Xp, G, model_id); 
      return m;
   }
   case 3:{
      VG_EVI* m = new VG_EVI(Xp,G,model_id);
     return m; 
   }
   case 4:{
      VG_VEI* m = new VG_VEI(Xp,G,model_id); 
      return m; 
   }
   case 5:{
      VG_VVI* m = new VG_VVI(Xp,G,model_id); 
      return m; 
   }
   case 6:{
      VG_EEE* m = new VG_EEE(Xp,G,model_id); 
      return m; 
   }
   case 7:{
      VG_VEE* m = new VG_VEE(Xp, G, model_id); 
      return m; 
   }
   case 8:{
      VG_EVE* m = new VG_EVE(Xp,G,model_id); 
      return m; 
   }
   case 9:{
      VG_EEV* m = new VG_EEV(Xp,G,model_id); 
      return m; 
   }
   case 10:{
      VG_VVE* m = new VG_VVE(Xp,G,model_id); 
      return m; 
   }
   case 11:{
      VG_EVV* m = new VG_EVV(Xp,G, model_id); 
      return m; 
   }
   case 12:{
      VG_VEV* m = new VG_VEV(Xp, G, model_id); 
      return m; 
   }
   default:{ 
      VG_VVV* m = new VG_VVV(Xp, G, model_id);
      return m;
   }
 }
}

// WRAPPERS 

// [[Rcpp::export]]
Rcpp::List main_loop_vg(arma::mat X, // data 
                     int G, int model_id, // number of groups and model id (id is for parrallel use)
                     int model_type,  // covariance model type
                     arma::mat in_zigs, // group memberships from initialization 
                     int in_nmax, // iteration max for EM . 
                     double in_l_tol, // liklihood tolerance 
                     int in_m_iter_max, // iteration max for M step for special models 
                     double in_m_tol, // tolerance for matrix convergence on M step for special mdodels.
                     arma::vec anneals,
                     int t_burn = 5// number of burn in steps for NAs if found. 
                     )
{
  


  int stochastic_check = 0;   // for stochastic variant
  if (19 < model_type ){
    model_type -= 20;  
    stochastic_check = 1;     
  }
  // create mixture model class. 
  std::unique_ptr<VG_Mixture_Model> m = std::unique_ptr<VG_Mixture_Model>(vg_create_model(&X,G,model_id,model_type));  

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

      // check to see if model_id is any of the special ones and set defaults, or pass in arguement. 
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
        m->E_step_latent();
        m->M_step_props();
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
      // perform intialization of params. 
      m->M_step_props(); 
      m->M_step_init_gaussian();
      m->M_step_mus();
      m->M_step_Ws();
      

      // check to see if model_id is any of the special ones and set defaults, or pass in arguement. 
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
      
      m->m_step_sigs();
      m->M_step_gamma(); 
      m->track_lg_init(); 

      for(size_t iter = 0; iter < (size_t)t_burn; iter++){
        m->E_step_latent();
        m->M_step_props();
        m->M_step_mus();
        m->M_step_Ws(); 
        m->m_step_sigs(); 
        m->M_step_gamma(); 
      }

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

        m->E_step(); 
        m->E_step_latent();
        m->M_step_props();
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
  }
  catch(const std::exception& e)
  {


    if(0 == std::string(e.what()).compare("logliklihood was infinite, back to previous step and returned results")){
    Rcpp::List ret_val = Rcpp::List::create(
                                            Rcpp::Named("mus") = m->mus, 
                                            Rcpp::Named("alphas") = m->alphas, 
                                            Rcpp::Named("sigs") = m->sigs,
                                            Rcpp::Named("G") = m->G, 
                                            Rcpp::Named("gammas") = m->gammas,
                                            Rcpp::Named("zigs") = m->zi_gs,
                                            Rcpp::Named("pi_gs") = m->pi_gs,
                                            Rcpp::Named("n_gs") = m->n_gs,
                                            Rcpp::Named("log_dets") = m->log_dets,
                                            Rcpp::Named("logliks") = m->logliks);
    }

    return Rcpp::List::create(Rcpp::Named("Error") = e.what()); 
  }

    Rcpp::List ret_val = Rcpp::List::create(
                                            Rcpp::Named("mus") = m->mus, 
                                            Rcpp::Named("alphas") = m->alphas, 
                                            Rcpp::Named("sigs") = m->sigs,
                                            Rcpp::Named("G") = m->G, 
                                            Rcpp::Named("gammas") = m->gammas,
                                            Rcpp::Named("zigs") = m->zi_gs,
                                            Rcpp::Named("pi_gs") = m->pi_gs,
                                            Rcpp::Named("n_gs") = m->n_gs,
                                            Rcpp::Named("log_dets") = m->log_dets,
                                            Rcpp::Named("logliks") = m->logliks);

  
  return ret_val;
}









// [[Rcpp::export]]
Rcpp::List vg_e_step_internal(arma::mat X, // data 
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
  std::vector<double> gammas = in_m_obj["gammas"];
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
  std::unique_ptr<VG_Mixture_Model> m = std::unique_ptr<VG_Mixture_Model>(vg_create_model(&X,G,model_id,model_type));  
  m->mus = mus;
  m->alphas = alphas;  
  m->sigs = sigs; 
  m->log_dets = log_dets; 
  m->pi_gs = pi_gs; 
  m->gammas = gammas; 
  m->n_gs = n_gs; 
  m->zi_gs = init_zigs; 
  m->init_missing_tags(); //Additional graphical and num

  // invert symmetric matrices. 
  for(int g = 0; g < G; g++)
  {
    m->inv_sigs[g] = arma::solve(sigs[g],m->EYE); 
  }

  // perform e_step and imputation. 
  m->E_step_only_burn(); 

  Rcpp::List ret_val = Rcpp::List::create(Rcpp::Named("X") = m->data.t(),
                                            Rcpp::Named("col_tags") = m->col_tags, 
                                            Rcpp::Named("origX") = X,
                                            Rcpp::Named("zigs") = m->zi_gs); 

  
  return ret_val;
}





