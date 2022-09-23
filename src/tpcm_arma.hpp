 #include <string> 

// create model function, generates a model pointer and returns it as a general mixture_model pointer
T_Mixture_Model* t_create_model(arma::mat* Xp,int G, int model_id, int model_type)
{
 switch(model_type)
 {
   case 0: {
      T_EII* m = new T_EII(Xp, G, model_id);
      return m;
   }
   case 1:{
      T_VII* m = new T_VII(Xp, G, model_id);
      return m;
   }
   case 2:{
      T_EEI* m = new T_EEI(Xp, G, model_id); 
      return m;
   }
   case 3:{
      T_EVI* m = new T_EVI(Xp,G,model_id);
     return m; 
   }
   case 4:{
      T_VEI* m = new T_VEI(Xp,G,model_id); 
      return m; 
   }
   case 5:{
      T_VVI* m = new T_VVI(Xp,G,model_id); 
      return m; 
   }
   case 6:{
      T_EEE* m = new T_EEE(Xp,G,model_id); 
      return m; 
   }
   case 7:{
      T_VEE* m = new T_VEE(Xp, G, model_id); 
      return m; 
   }
   case 8:{
      T_EVE* m = new T_EVE(Xp,G,model_id); 
      return m; 
   }
   case 9:{
      T_EEV* m = new T_EEV(Xp,G,model_id); 
      return m; 
   }
   case 10:{
      T_VVE* m = new T_VVE(Xp,G,model_id); 
      return m; 
   }
   case 11:{
      T_EVV* m = new T_EVV(Xp,G, model_id); 
      return m; 
   }
   case 12:{
      T_VEV* m = new T_VEV(Xp, G, model_id); 
      return m; 
   }
   default:{ 
      T_VVV* m = new T_VVV(Xp, G, model_id);
      return m;
   }
 }
}

// WRAPPERS 

// [[Rcpp::export]]
Rcpp::List main_loop_t(arma::mat X, // data 
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
  
  bool constrained_check = false; 
  
  if(model_id == 21){
    // set constrained update for vgs
    constrained_check = true;
  }
  if(model_id == 23){
    constrained_check = true; 
    model_id -= 1;
  }

  model_id -= 20; 

  int stochastic_check = 0; 

  // for stochastic variant
  if (19 < model_type ){
    model_type -= 20;  
    stochastic_check = 1;     
  }
  // create mixture model class. 
  std::unique_ptr<T_Mixture_Model> m = std::unique_ptr<T_Mixture_Model>(t_create_model(&X,G,model_id,model_type));  


  // check if it semi-supervised. 
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
  m->set_m_step_vgs(constrained_check); 

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
  NA_check = ( m->row_tags.size() > 0); 


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
      m->E_step_ws();  
      m->M_step_mus();
      m->M_step_Ws(); 
      m->set_defaults();
      m->m_step_sigs(); 
      m->M_step_vgs();
      m->track_lg_init(); 

      arma::uword nmax = (arma::uword)in_nmax; 
      bool convergence_check = false; 
      // main EM with extra setp. 
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
        m->M_step_props();
        m->impute_cond_mean(); // now have imputation step. 
        m->E_step_ws(); 
        m->M_step_mus();
        m->M_step_Ws(); 
        m->m_step_sigs(); 
        m->M_step_vgs();

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
      m->E_step_ws(); 
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
      m->M_step_vgs();
      m->track_lg_init(); 

      for(size_t iter = 0; iter < (size_t)t_burn; iter++){
        m->E_step_ws(); 
        m->M_step_props();
        m->M_step_mus();
        m->M_step_Ws(); 
        m->m_step_sigs(); 
        m->M_step_vgs();
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
        m->E_step_ws(); 
        m->M_step_props();
        m->M_step_mus();
        m->M_step_Ws(); 
        m->m_step_sigs(); 
        m->M_step_vgs();

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
                                            Rcpp::Named("sigs") = m->sigs,
                                            Rcpp::Named("G") = m->G, 
                                            Rcpp::Named("vgs") = m->vgs,
                                            Rcpp::Named("zigs") = m->zi_gs,
                                            Rcpp::Named("pi_gs") = m->pi_gs,
                                            Rcpp::Named("n_gs") = m->n_gs,
                                            Rcpp::Named("log_dets") = m->log_dets,
                                            Rcpp::Named("logliks") = m->logliks);


      if(NA_check){
        ret_val["X"] = m->data.t(); 
      }     
      return ret_val; 
    }

    return Rcpp::List::create(Rcpp::Named("Error") = e.what()); 
  }

    Rcpp::List ret_val = Rcpp::List::create(
                                            Rcpp::Named("mus") = m->mus, 
                                            Rcpp::Named("sigs") = m->sigs,
                                            Rcpp::Named("G") = m->G, 
                                            Rcpp::Named("vgs") = m->vgs,
                                            Rcpp::Named("zigs") = m->zi_gs,
                                            Rcpp::Named("pi_gs") = m->pi_gs,
                                            Rcpp::Named("n_gs") = m->n_gs,
                                            Rcpp::Named("log_dets") = m->log_dets,
                                            Rcpp::Named("logliks") = m->logliks);

    if(NA_check){
      ret_val["X"] = m->data.t(); 
    }
              
  return ret_val;
}









// [[Rcpp::export]]
Rcpp::List t_e_step_internal(arma::mat X, // data 
                           int G, int model_id, // number of groups and model id (id is for parrallel use)
                           int model_type,  // covariance model type
                           Rcpp::List in_m_obj, // internal object from output
                           arma::mat init_zigs, 
                           double in_nu = 1.0)
{
  // declare params. that are passed in from the internal object.  
  std::vector< arma::rowvec > mus_t = in_m_obj["mus"]; 
  std::vector< arma::mat > sigs = in_m_obj["sigs"];
  std::vector<double> vgs = in_m_obj["vgs"];
  std::vector<double> n_gs = in_m_obj["n_gs"];
  std::vector<double> log_dets = in_m_obj["log_dets"];
  arma::rowvec pi_gs_t = in_m_obj["pi_gs"];


  arma::rowvec pi_gs = pi_gs_t; 

  // create model and set existing parameters. 
  std::unique_ptr<T_Mixture_Model> m = std::unique_ptr<T_Mixture_Model>(t_create_model(&X,G,model_id,model_type));  
  m->mus = mus_t;
  m->sigs = sigs; 
  m->log_dets = log_dets; 
  m->pi_gs = pi_gs; 
  m->vgs = vgs; 
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

  Rcpp::List ret_val = Rcpp::List::create(Rcpp::Named("X") = m->data,
                                            Rcpp::Named("row_tags") = m->row_tags, 
                                            Rcpp::Named("origX") = X,
                                            Rcpp::Named("zigs") = m->zi_gs); 

  
  return ret_val;
}





