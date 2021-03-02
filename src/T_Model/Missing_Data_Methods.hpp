#include "Covariance_Methods.hpp"



// MISSING DATA METHODS FOR GENERAL R_Mixture_Model class. 
// grabs missing values and their respective tags and row_tags.
#pragma once 
void T_Mixture_Model::init_missing_tags(void)
{

  std::vector<arma::uvec> in_missing_tags; // create the missing tags vector 
  arma::uvec in_row_tags; // set up row tags.  

  // loop through rows. 
  for(int i = 0; i < n;  i++ )
  {
    // get the current nantags. 
    arma::uvec nan_tags = arma::find_nonfinite(data.row(i));

    if(nan_tags.n_elem > 0)
    {
      // get row id uvec. 
      arma::uvec row_id = arma::uvec(1); // init
      row_id[0] = i; // set entry 
                    
      // concatonate both row uvec and the nan_tags
      arma::uvec mis_tag_i = arma::join_cols(row_id,nan_tags); 
      in_row_tags = arma::join_cols(in_row_tags,row_id); 

      // add to missing tags list
      in_missing_tags.push_back(mis_tag_i); 
    } 
  }
  // assign row tags and missing tags. 
  row_tags = in_row_tags; 
  missing_tags = in_missing_tags; 
}



// EM BURN_in METHOD 
// takes in number of steps to run the EM algorithm WITHOUT imputation of missing data. 
// this will initialize the model for better imputation 
#pragma once
void T_Mixture_Model::E_step_only_burn(void)
{

  // impute_cond_mean using z_igs. 
  impute_cond_mean(); 
  E_step(); // then perform E_step on the entire dataset. 
  E_step_ws(); // then perform E_step on the entire dataset. 
  impute_cond_mean(); // impute conditional mean again. 
  E_step(); // e_step again.
  E_step_ws(); // then perform E_step on the entire dataset.  
  impute_cond_mean(); // one more for good luck 
  E_step(); // e_step. Not that every E_step, the parmaeters dont change but the imputation does. 
  E_step_ws(); // then perform E_step on the entire dataset. 
}





// EM BURN_in METHOD 
// takes in number of steps to run the EM algorithm WITHOUT imputation of missing data. 
// this will initialize the model for better imputation 
#pragma once
void T_Mixture_Model::EM_burn(int in_burn_steps)
{
  // copy dataset and z_igs. 
  arma::mat* orig_data = new arma::mat(n,p); // create empty arma mat on the heap. 
  arma::mat* orig_zi_gs = new arma::mat(n,G); 
  *orig_data = data; // set orig_data. 
  *orig_zi_gs = zi_gs; // set zi_igs. 

  // remove all data, and zi_gs with missing values. 
  data.shed_rows(row_tags); 
  zi_gs.shed_rows(row_tags); 
 
  n = data.n_rows; 
  
  // intialize all parameters. (all methods are on self)
  M_step_props();
  E_step_ws(); // then perform E_step on the entire dataset.  
  M_step_mus();
  M_step_Ws();
  m_step_sigs();
  M_step_vgs();
 
  
  // run EM burn in for in_burn_steps number of steps.  
  for(int i = 0; i < in_burn_steps; i++)
  {
    E_step();
    E_step_ws(); // then perform E_step on the entire dataset.  
    M_step_props(); 
    M_step_mus();
    M_step_Ws();
    m_step_sigs();
    M_step_vgs();
  }
  // Now replace back the original data points and zi_igs. only keep the parmaeters.   
  data = *orig_data; 
  zi_gs = *orig_zi_gs; // done EM burn 
  
}


// imputation cond_mean functions. replaces values based on approach. 
#pragma once
void T_Mixture_Model::impute_cond_mean(void)
{

  // go through each of the tags and select the row out of the dataset 
  for(arma::uword i_tag = 0; i_tag < row_tags.n_elem; i_tag++)
  {
    arma::uvec current_tag = missing_tags[i_tag];// get current full tag. 
    current_tag.shed_row(0); // remove the row tag.  
    arma::uword row_tag = row_tags.at(i_tag); 

     // create the missing column vector with current tag the from the data. 
    arma::mat c_obs = data.row(row_tag).t();
    arma::mat m_obs = data.row(row_tag).t();
    arma::mat nm_obs = data.row(row_tag).t(); // use column vectors for calculations. 

    // drop the indeces that contain NAS.
    nm_obs.shed_rows(current_tag);
    // select only ones that contain NAS.
    m_obs = m_obs.rows(current_tag); 
    
    // now I have to iterate through g groups to compute the conditional mean imputation. 
    // remember to change this for groups G. it doesnt have to be at 2.  
    for(int g = 0; g < G; g++)
    {
      // current mus. 
      arma::mat c_mu_m = mus[g].t(); // invert because column vector. 
      arma::mat c_mu_nm = mus[g].t(); // invert because column vector.
      arma::mat c_sig = sigs[g]; // self explanitory. (current sig)

      c_mu_m = c_mu_m.rows(current_tag); // select missing rows of mu
      c_mu_nm.shed_rows(current_tag); // select non missing rows of mu 

      arma::mat c_sig_m = c_sig; // set c_sigs because I have to shed. 
      arma::mat c_sig_nm = c_sig;

      c_sig_m.shed_cols(current_tag);  // shed rows and select one you need. 
      c_sig_m = c_sig_m.rows(current_tag); 

      c_sig_nm.shed_cols(current_tag); 
      c_sig_nm.shed_rows(current_tag); // make this square, non missing. 

      // finally compute imputation. 
      double z_ig = zi_gs.at(row_tag,g); // CHANGE THIS LATER IN FULL FUNCTION. 

      // MIXTURES OF CONDITIONAL MEAN IMPUTATION 
      arma::vec x_nm_dif = nm_obs - c_mu_nm; 
      arma::mat EYE_nm = EYE; 
      EYE_nm.shed_cols(current_tag); 
      EYE_nm.shed_rows(current_tag); 
      // note to self during debug, check if you are on first becuase of nans and assign conditional mean imputation 
      if(g == 0){
        m_obs = z_ig*( c_mu_m +  c_sig_m * arma::solve(c_sig_nm,EYE_nm,arma::solve_opts::refine) * x_nm_dif); 
      }
      else{
        m_obs += z_ig*( c_mu_m +  c_sig_m * arma::solve(c_sig_nm,EYE_nm,arma::solve_opts::refine) * x_nm_dif); 
      }

    }
    // assign missing values. 
    for(arma::uword m_i = 0; m_i < m_obs.size(); m_i++ )
    {
      data.at(row_tag,current_tag[m_i]) = m_obs.at(m_i); 
    }
  }

}

void T_Mixture_Model::impute_init(void)
{
  impute_cond_mean(); // after burn you impute as initalization. 
  E_step(); // calculate e step over entire dataset. 
  E_step_ws(); // then perform E_step on the entire dataset.  
  // run m_step once. 
  M_step_props(); 
  M_step_mus();
  M_step_Ws();
  m_step_sigs();
  M_step_vgs();
}
