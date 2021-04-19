
.packageName<-'mixture' # do I really need this? 

set.seed(1)

    # MARK :// ARGS 
    # G , number of groups
    # data , matrix X, 
    # mnames = model names. (could be NULL) 
    # start , just do kmeans for now., then you will add annealing, random start, and general functions 
    # labels 
    # veo, number of variables in the model exceeds the number of observations, still fit. 
    # nmax, The maximum number of iterations each EM algorithm is allowed to use
    # atol, A number specifying the epsilon value for the convergence criteria used in the
    # mtol, A number specifying the epsilon value for the convergence criteria used in the M-step in the GEM algorithms.
    # mmax, The maximum number of iterations each M-step is allowed in the GEM algorithms.
    # pprogress, If TRUE print the progress of the function.
    # pwarning If TRUE print the warnings.
    

# MAIN GPCM FUNCTION
gpcm <- function(data=NULL,  G=1:3, mnames=NULL, # main inputs with mnames being the Model Name 
				start=2, label=NULL, # starting inputs , start = 0: random soft, start = 2, random hard. start = 3 mkmeans. 
				veo=FALSE, da=c(1.0), # veo (variables exceed observations), da is deterministic annealing  
				nmax=1000, atol=1e-8, mtol=1e-8, mmax=10, burn=5, # convergence settings for matrix and loglik
				pprogress=FALSE, pwarning=TRUE, stochastic = FALSE)  # progress settings 
{

	# Do some sanity checks. 
	if (is.null(data)) stop('Hey, we need some data, please! data is null')
	if (!is.matrix(data)) stop('The data needs to be in matrix form')
	if (!is.numeric(data)) stop('The data is required to be numeric')
	if (nrow(data) == 1) stop('nrow(data) is equal to 1')
	if (ncol(data) == 1) stop('ncol(data) is equal to 1; This function currently only works with multivariate data p > 1')
	
	# check for full NAs vector. as mixture version 1.6+ can handle missing data.  
	apply(data,1,checkNA)

	# some more sanity checks. 
	if (is.null(G)) stop('G is NULL')
	G = as.integer(ceiling(G))
	if (!is.integer(G)) stop('G is not a integer')
	if ( any(G < 1)) stop('G is not a positive integer')
	n <- dim(data)[1]
	p <- dim(data)[2]

	row_tags <- c()
	# grab row_tags if na 
	if(any(is.na(data))){
		for(i in 1:n)
		{
			if(any(is.na(data[i,])))
			{
				row_tags <- append(i,row_tags)
			}
		}
	}

	# IF MODEL NAME NOT SPECIFIED DO EM ON ALL POSSIBLE MODELS 
	if (is.null(mnames) )  mnames = c("EII", "VII", "EEI", "VEI", "EVI", "VVI", "EEE", "EEV", "VEV", "VVV", "EVE", "VVE", "VEE", "EVV")

	# CREATE MULTIDIMENSIONAL ARRAY CONTAINING THE FOLLOWING:
	# G x MODEL_LENGTH x INFO (loglik, npar, BIC)
	info_BIC <- c() # BIC for each model 
	info_loglik <- c() # logliklihoods for each model
	info_npar <- c() # gives number of parameters vector
	info_model_lexicon <- c() # gives header tag for the full model, useful for print statements and summary
	info_model <- c() # stores full model data 

	# SET START OBJECT
	# handeling z_ig matrices. 
	if ( is.matrix(start))
	{
		if(length(G) != 1){
			stop("Initialization z matrix should only be for a single G")
		}
		if(dim(start)[1] != dim(data)[1])
		{
			stop("Initialization z matrix should have the same number of rows as data")
		}
		if(dim(start)[2] != G)
		{
			stop("Initialization z matrix should have G number of columns")
		}
		startobject <- "matrix"
	}
	else if ( start == 0){
		startobject <- "random_soft" # check for validity of start object. 
	} 
	else if ( start == 1){
		startobject <- "random_hard" 
	}
	else if ( start == 2){
	  if ( any(is.na(data)) ) { stop("You cannot use kmeans on missing values, try soft initialization first, then after a kmeans start.") }
		startobject <- "kmeans"
	} 
	else{
		stop("start setting is not valid")
	}


	# deterministic annealing sanity checks
	if(any(is.na(da))){ stop("deterministic annealing should contain no NAs") } # check NAs
	else if(!is.vector(da)) { stop("deterministic annealing should be a vector")} # check vector 
	else if(!is.numeric(da)) { stop("deterministic annealing should be numeric")} # check numeric 

	# burn setting check 
	if(!is.numeric(burn)){ stop("burn-in setting has to be a number")}
	if(round(burn) <= 0) { stop("burn-in setting has to be a positive round number")}
	if(burn != round(burn)){
		if(!pwarning){
			warning("Warning: rounding burn-in setting number")
		}
		burn <- round(burn) 
	}

	# label sanity checks 
	if(!is.null(label)) { # if there is no null
		if(!is.vector(label)){ stop("label has to be a vector")} #check that its a vector. 
		else if(any(is.na(label))) { stop("No NAs allowed in label vector.")} # check for NAs
		else if(any(label < 0)) { stop("labels have to be greater or equal to 0")} # make sure they are greater than 0
		else if(any(label > G)) { stop("labels cant be bigger than G groups")}  # make sure they are greater than G
		else if(length(label) != n) { stop("label vector has to be the same size as the number of observtions. ")} # check length. 
		else if(length(G) != 1) { stop("Labels can only be inputted in for a single G") } # check for single input G. 
	}

	# go through all G's 
	for(G_i in G)
	{
		# go through all models
		for(model_name in mnames)
		{
			if(pprogress) {	cat("Running GPCM Model:",model_name,"G=",G_i,"\n")}
			# calculate number of parameters once. 
			number_of_params <- npar.model(model_name,G=G_i,p=p)

			check_veo <- TRUE

			if(number_of_params > n){
				if(veo){
					if(!pwarning){
					warning("Model: ",model_name , " G: " , G_i ," ","Number of Parameters exceed number of observations.\n")
					}
				}
				else{
					check_veo <- FALSE
				}
			}

			# check to see if number of parameters exceed observations. 
			if(check_veo){
				# get model_type 
				model_id <- model_to_id(model_name)
				model_id_stochastic_check <- model_id

				if(G_i > 1){
					# set up intialization matrix depending on choice. 
					in_zigs <- switch(startobject,
					"random_soft"=z_ig_random_soft(n,G_i),
					"random_hard"=z_ig_random_hard(n,G_i),
					"kmeans"=z_ig_kmeans(data,G_i),
					"matrix"=start)
				
					# handle labels within z_ig matrix.  
					if(!is.null(label)){
						model_id_stochastic_check <- 2

						if(stochastic)
						{
							stop('Under current version, you cannot have semi-supervision and stochastic EM')
						}
						
						# start observation count
						i <- 1
						# go through the entire label system 
						for(label_i in label){
							# if its not zero, replace the one entry in z_ig with just 0s and 1 one. 
							if(label_i != 0){
								# create vector of zeros and put a 1 based on the label i 
								classif_vector <- rep(0,G_i) # rep 0. 
								classif_vector[label_i] <- 5 # put five
								in_zigs[i,] <- classif_vector # replace
							}
							i <- i + 1
						}
					}
				}
				else{
					in_zigs <- as.matrix(rep(1.0,n))
				}

				# run model with settings 
				model_results_i <- main_loop(X=data, G=G_i, in_zigs=in_zigs,# dataset G_i, z_igs
										model_id=model_id_stochastic_check, model_type=model_id + 20*stochastic, # for all intensive purposes these are the same. (They will change later)
										in_nmax=nmax, in_l_tol=atol, # em number of iterations 
										in_m_iter_max=mmax,in_m_tol=mtol, # m_step matrix iterations and convergence settings 
										anneals=da,t_burn=burn) # annealing and burn in settings. 
				
				status <- "Failed Aitken's Convergence Criterion"
				if (nmax > length(model_results_i$logliks)) {
					status <- "Converged according to Aitken's Convergence Criterion"
				}
				# AQUIRE STATUS IF ANY 
				model_results_i$status <- status
				if(is.null(model_results_i$Error)){
					info_loglik <- append(tail(model_results_i$logliks,1),info_loglik) # first one is loglik 
					info_npar <- append(number_of_params,info_npar) # number of paramaters. 
					info_BIC <- append(2*info_loglik[1] - log(n)*info_npar[1],info_BIC) # append BIC value 
					info_model_lexicon <- append(paste("Model:", model_name, "G: ",G_i),info_model_lexicon) # lexicon for summary
					model_results_i$sigs <- convert_matrices(model_results_i,G_i,p) # convert matrices into proper form
					info_model <- append(list(model_results_i),info_model) # store all model objects. 
				}
				else{
					info_loglik <- append(NA,info_loglik) # first one is loglik 
					info_npar <- append(number_of_params,info_npar) # number of paramaters. 
					info_BIC <- append(NA,info_BIC) # append BIC value 
					info_model_lexicon <- append(paste("Model:", model_name, "G: ",G_i),info_model_lexicon) # lexicon for summary
					info_model <- append(list(NA),info_model) # store all model objects. 
				}
			}
		}

	}

	info_matrix <- list(startobject=startobject, # gives starting object information. 
						info_loglik=info_loglik,info_npar=info_npar,info_BIC=info_BIC, # logliks, params, BICs, 
						lexicon=info_model_lexicon,model_objs=info_model,Gs=G,row_tags=(row_tags+1)) # lexicon (tags), and model_objs 
	if(length(info_matrix$model_obj) < 1){                                       # plus 1 for row_tags due to c++
		stop("No models estimated")
	}
	info_matrix$best_model <- get_best_model(info_matrix)
	info_matrix$map <- MAP(info_matrix$best_model$model_obj[[1]]$zigs)
	info_matrix$best_model$map <- info_matrix$map

	# Legacy code adapted by Nik Pocuca to fix dependency issues...
	info_matrix$BIC <- construct_BIC_3D(info_matrix)

    gpar= list()
	for (k in 1:info_matrix$best_model$G ) {
		gpar[[k]] = list()		
		gpar[[k]]$mu       = info_matrix$best_model$model_obj[[1]]$mus[[k]]
		gpar[[k]]$sigma    = info_matrix$best_model$model_obj[[1]]$sigs[[k]]
		gpar[[k]]$invSigma = try(solve(info_matrix$best_model$model_obj[[1]]$sigs[[k]]))
		gpar[[k]]$logdet   = info_matrix$best_model$model_obj[[1]]$log_dets[k]

	}

	gpar$pi = info_matrix$best_model$model_obj[[1]]$pi_gs

	info_matrix$gpar <- gpar
	info_matrix$z <- info_matrix$best_model$model_obj[[1]]$zigs

	class(info_matrix) <- "gpcm"

	return(info_matrix)

}

# PRINT SUMMARY AND PLOT STATEMENTS
print.gpcm <-function(x, ...){
  # split strings and parse
	splitted_strings <- strsplit(x$best_model$model_type," ")[[1]]
  # print to line parsed strings. 
  cat("The model chosen by applying the BIC criteria has", trimws(splitted_strings[5]), "component(s) and the", trimws(splitted_strings[2]), "covariance structure\n using", x$startobject, "\n"  )
}
# just prints the compare BIC matrix. 
summary.gpcm <- function(object, ...){
    cat("BIC for each model, number of components (rows), and covariance structure (columns).\n")
	print(object$BIC[,,3])
}

# plots a line graph of BICs. 
plot.gpcm <- function(x, ...) {
	bicl = x$BIC[,,3]
	# you need to wrap this plot up in a print statement otherwise it doesnt work. 
	print(levelplot(bicl,
          col.regions = colorRampPalette(c("black","brown","red","orange","gold","yellow","white"))(prod(dim(bicl)) + 10),
          xlab = "G",ylab = "Covariance Type"))
	trellis.focus("legend", side="right", clipp.off=TRUE, highlight=FALSE)
	trellis.unfocus()	
}



# MARK :// EVERYTHING BELOW THIS LINE ARE 
# HELPER FUNCTIONS !!  

# check NA function along vector. 
# usually performed with apply() function along margin 1. 
checkNA <- function(vec)
{
	# count the number of NA 
	num_vec_nas = sum(is.na(vec))
	if(num_vec_nas >= length(vec)) stop("Cannot impute missing data along vector, need at least one non missing value value")
}

# Function: model to id
# given a model string name, returns id associated with it.
# input: m_name (model string name) 
model_to_id <- function(in_m_name)
{
	model_id <- switch(in_m_name, "EII" = 0,"VII" = 1,  "EEI" = 2,  "EVI" = 3,  "VEI" = 4,  "VVI" = 5,  "EEE" = 6,  
                        "VEE" = 7,  "EVE" = 8,  "EEV" = 9,  "VVE" = 10,  "EVV" = 11,"VEV" = 12,"VVV" = 13)
	if(is.null(model_id))
	{
		stop("Error, Model name is not one of the 14 models.")
	}
	else{ return(model_id)}
}

# Function: id to model. 
# given a model id, returns a model string name. 
# simple functions
# id to model 
id_to_model <- function(in_id)
{
	model_id <- switch(in_id, "0" = "EII", "1" = "VII",  "2" = "EEI" ,  "3" = "EVI", "4" = "VEI",  "5" = "VVI", "6" = "EEE",  
                        "7" = "VEE", "8" = "EVE", "9" = "EEV", "10" = "VVE", "11" = "EVV", "12" = "VEV", "13" = "VVV")

	if( is.null(model_id))
	{ stop("Error, not one of 14 ids (starts at 0).")
	}
	else{
		return(model_id)
	}
}

	# z <- matrix(0,nrow = n, ncol = g)
	# for(i in 1:g){
	# 	z[,i] <- runif(n)	
	# }
	# z <- t(apply(z,1,function(x) { x/sum(x) }))
	# z <- t(apply(z,1,function(x) { x[length(x)] <- 1 -  sum(x[1:(length(x)-1)]); x  }  ))
	# z

# Function: constructs random soft matrix 
z_ig_random_soft <- function(n,g)
{
	z <- matrix( rexp(n*g), nrow = n, ncol = g)
	z <- z/rowSums(z)
	z[,g] = 1 - rowSums(as.matrix(z[,1:(g-1)]))
	z
}

# Function: constructs random hard matrix. 
z_ig_random_hard <- function(n,g)
{
	z <- matrix( rexp(n*g), nrow = n, ncol = g)
	z <- z/rowSums(z)
	z[,g] = 1 - rowSums(as.matrix(z[,1:(g-1)]))
	z <- t(apply(z,1,function(x) { x/sum(x) }))
	z <- t(apply(z,1,function(x) { x[length(x)] <- 1 -  sum(x[1:(length(x)-1)]); x  }  ))
	z <- t(apply(z,1,function(x) { v <- rep(0,length(x))
								ind <- match(max(x),x)
								v[ind ] <- 1
								v 
	}))
	z
}


#Function: kmeans intialization construct 
z_ig_kmeans <- function(X,g)
{
	n <- dim(X)[1]
	z <- matrix(0,nrow = n, ncol = g)
	km <- kmeans(X,g)
	row_count = 0
	for(i in km$cluster){
		row_count = row_count + 1
		z[row_count,i] <- 1
	}
	return (z)
}


# calculates the number of parameters for a given model. this is a leftover function from 1.1  
npar.model <- function(modelname=NULL, p=NULL, G=NULL) {
	val = numeric(3)
	val[1] = G-1 
	val[2] = G*p
	val[3] = ncovpar(modelname= modelname, p=p, G=G)
	val = sum(val)
	return(val)
}


# calculates the number of parameters for a given model. this is a leftover function from 1.1  
npar.model.skew <- function(modelname=NULL, p=NULL, G=NULL, family_name=NULL) {
	val = numeric(4)
	val[1] = G-1 
	val[2] = G*p*2 # multiply by 2 because mu and alpha
	val[3] = ncovpar(modelname= modelname, p=p, G=G) # this remains the same 
	val[4] = npar.gamma(family_name = family_name, G=G) # number of parameters of gamma terms  
	val = sum(val)
	return(val)
}

npar.gamma <- function(family_name = NULL, G=NULL){
	if (is.null(G)) stop("G is null")
	if (is.null(family_name)) stop("family name is null")

	if (family_name =="VG") npar = G
	else if (family_name == "SAL") npar = 0
	else if (family_name == "GH") npar = G*2
	else if (family_name == "ST") npar = G 
	else if (family_name == "T") npar = G
	else stop("family name is not properly defined")

	return (npar)
}




# calculates number of parameters. This is a leftover function from version 1.1, 
# this should be done with a switch statement but its really not worth it to re-code if it works. 
ncovpar <- function(modelname=NULL, p=NULL, G=NULL) {
	if (is.null(p)) stop("p is null")
	if (is.null(G)) stop("G is null")
	if (is.null(modelname)) stop("modelname is null")

	     if (modelname == "EII") npar = 1
	else if (modelname == "VII") npar = G
	else if (modelname == "EEI") npar = p
	else if (modelname == "VEI") npar = p + G -1	
	else if (modelname == "EVI") npar = p*G - G +1
	else if (modelname == "VVI") npar = p*G
	else if (modelname == "EEE") npar = p*(p+1)/2
	else if (modelname == "EEV") npar = G*p*(p+1)/2 - (G-1)*p	
	else if (modelname == "VEV") npar = G*p*(p+1)/2 - (G-1)*(p-1)
	else if (modelname == "VVV") npar = G*p*(p+1)/2
	else if (modelname == "EVE") npar = p*(p+1)/2 + (G-1)*(p-1)
	else if (modelname == "VVE") npar = p*(p+1)/2 + (G-1)*p
	else if (modelname == "VEE") npar = p*(p+1)/2 + (G-1)
	else if (modelname == "EVV") npar = G*p*(p+1)/2 - (G-1)
	else stop("modelname is not correctly defined")
	
	return(npar)		
}


# Print output for the gpcm_best class. 
print.gpcm_best <-function(x, ...){

	splitted_strings  <- strsplit(x$model_type," ")[[1]]
	cov_string <- splitted_strings[2]
	component_string <- splitted_strings[5]

	cat("============================\n")
    cat("Best Model According To BIC \n")
	cat("============================\n")
	cat("Status: ", strsplit(x$status," ")[[1]][1], "\n")
	cat("Covariance Model Type: ",cov_string,"\n")
	cat("Number of Components: ",component_string,"\n")
	cat("Initalization: ",x$startobject,"\n")
	cat("BIC: ", x$BIC, "\n")
	cat("============================\n")
	
}

# Function: calculates the best model from a full list of models and their objects. 
# Get best model according to BIC 
get_best_model <- function(gpcm_model)
{	
	# sanity checks for input 
	if(!is.list(gpcm_model)) { stop("Error: Input is not a gpcm_model") }
	if(!is.list(gpcm_model$model_objs)) { stop("Error: model_objs missing... input is not a gpcm_model ") }
	
	# replace infinity and NaNs with NA 
	gpcm_model$info_BIC[!is.finite(gpcm_model$info_BIC)] <- NA
		
	if(all(is.na(gpcm_model$info_BIC))){
		stop("error: no models estimated.")
	}

	# get best_model index. 
	bm_index <- try(match(try(max(gpcm_model$info_BIC,na.rm = TRUE)), gpcm_model$info_BIC))

	# get G 
	lexicon_best <- strsplit(gpcm_model$lexicon[bm_index]," ")[[1]]
	G_best <- as.numeric(lexicon_best[5])
	Cov_type <- lexicon_best[2]

	status <- gpcm_model$model_objs[[bm_index]]$status
	# construct best model 
	best_model <- list(model_type=gpcm_model$lexicon[bm_index], # what the call is. 
					   model_obj=gpcm_model$model_objs[bm_index], # return model objects for the best model. 
					   BIC=gpcm_model$info_BIC[bm_index], loglik=gpcm_model$info_loglik[bm_index], # get best BIC and loglik. 
					   nparam=gpcm_model$info_npar[bm_index],startobject=gpcm_model$startobject, G=G_best,cov_type=Cov_type ,# number of parameters and G. 
					   status=status) # convergence status

	
	best_model$map <- MAP(best_model$model_obj[[1]]$zigs)

	class(best_model) <- "gpcm_best"


	return(best_model)

}


# LEGACY CODE HELPER FUNCTION. 
construct_BIC_3D <-  function(gpcm_model)
{
  # get BIC and row col names vector 
  BICs <- gpcm_model$info_BIC 
  LOGLIKs <- gpcm_model$info_loglik
  NPARs <- gpcm_model$info_npar
  lexicon <- gpcm_model$lexicon

  # header names and such/ 
  unique_cov_types <- c() 
  unique_Gs <- c()
  # go through and get unique Gs
  for(i in lexicon)
  {
	lexicon_i <- strsplit(i," ")[[1]]
	G_i <- as.numeric(lexicon_i[5])
	Cov_type_i <- lexicon_i[2]
	unique_cov_types <- unique(append(unique_cov_types,Cov_type_i))
	unique_Gs <- sort(unique(append(unique_Gs,G_i)))
  }

	# construct bic_matrix. 
	bic_matrix <- matrix(NA,nrow=length(unique_Gs),ncol=length(unique_cov_types))
	rownames(bic_matrix) <- as.character(unique_Gs)
	colnames(bic_matrix) <- unique_cov_types
	
	# construct LOGLIK matrix 
	loglik_matrix <- matrix(NA,nrow=length(unique_Gs),ncol=length(unique_cov_types))
	rownames(loglik_matrix) <- as.character(unique_Gs)
	colnames(loglik_matrix) <- unique_cov_types

	# construct NPAR matrix. 
	npar_matrix <- matrix(NA,nrow=length(unique_Gs),ncol=length(unique_cov_types))
	rownames(npar_matrix) <- as.character(unique_Gs)
	colnames(npar_matrix) <- unique_cov_types

	for(i in 1:length(lexicon))
	{
		# current lexicon and BIC.  
		lexicon_i <- lexicon[i]
		BIC_i <- BICs[i]
		LOGLIK_i <- LOGLIKs[i]
		NPAR_i <- NPARs[i]

		# extract lexicon 
		lex_strip <- strsplit(lexicon_i," ")[[1]]
		G_i <- as.numeric(lex_strip[5])
		Cov_type_i <- lex_strip[2]

		index_cov_type_i <- match(Cov_type_i,colnames(bic_matrix))
		index_g_i <- match(G_i,rownames(bic_matrix))

		bic_matrix[index_g_i,index_cov_type_i] <- BIC_i
		loglik_matrix[index_g_i,index_cov_type_i] <- LOGLIK_i
		npar_matrix[index_g_i,index_cov_type_i] <- NPAR_i
	}

	bic_matrix[!is.finite(bic_matrix)] <- NA
	loglik_matrix[!is.finite(loglik_matrix)] <- NA

	# get massive matrix. 
	
	return_var_mat = array(0,c(dim(npar_matrix),3))
	return_var_mat[,,1] = loglik_matrix
	return_var_mat[,,2] = npar_matrix
	return_var_mat[,,3] = bic_matrix
	
	row_column_names = dimnames(bic_matrix)
	row_column_names[[3]] = c("loglik","npar","BIC")
	dimnames(return_var_mat) <- row_column_names

	return(return_var_mat)
}





# Gets full dataset, if NA's were passed in,it was imputed. 
get_data <- function(gpcm_best_model){
	return(gpcm_best_model$model_obj[[1]]$X)
}

# take in zi_gs. 
MAP <- function(z_ig) {
	n <- dim(z_ig)[1]
	g <- dim(z_ig)[2]
	labs <- rep(0,n)
	labs <- t(apply(z_ig,1,function(x) {
								ind <- match(max(x),x)
								return(ind)
	}))
	
	return(as.numeric(labs))	
}


# Matrices are in row vector when they come out of C++, so this one converts them to the proper format. 
convert_matrices <- function(internal_model,G,p){
  
  temp_sigs <- internal_model$sigs
  sigs <- internal_model$sigs
  
  for(g in 1:G){
    sigs[g][[1]] <- matrix(unlist(temp_sigs[g]),nrow=p,ncol=p)
  }
  return(sigs)
}

# Rcpp::List e_step_internal(arma::mat X, // data 
                        #    int G, int model_id, // number of groups and model id (id is for parrallel use)
                        #    int model_type,  // covariance model type
                        #    Rcpp::List in_m_obj, // internal object from output
                        #    arma::mat init_zigs, 
                        #    double in_nu = 1.0)

#
#  expect <- e_step_internal(x2NA,2,13,13,best$model_obj[[1]],init_zigs = init_zi,in_nu=1.0 )
# Function e_step: 
# Imputes and calculates the a posterori of a 
e_step <- function(data, # matrix data, G integer, number of groups 
					model_obj, # intake model parameters. 
					start=0, nu = 1.0) # start is same as gpcm, 0 - soft, 1 hard, 2 kmeans, matrix. own function. , nu is deterministic annealing.  
{
	# check if data is in matrix form. 
	if (is.null(data)) stop('Hey, we need some data, please! data is null')
	if (!is.matrix(data)) stop('The data needs to be in matrix form')
	if (!is.numeric(data)) stop('The data is required to be numeric')
	if (nrow(data) == 1) stop('nrow(data) is equal to 1')
	if (ncol(data) == 1) stop('ncol(data) is equal to 1; This function currently only works with multivariate data p > 1')
	# check for full NAs vector. as mixture version 1.6+ can handle missing data.  
	apply(data,1,checkNA)

	G <- model_obj$G
	# some more sanity checks for integer. 
	if (is.null(G)) stop('G is NULL')
	G = as.integer(ceiling(G))
	if (!is.integer(G)) stop('G is not a integer')
	if ( any(G < 1)) stop('G is not a positive integer')
	n <- dim(data)[1]
	p <- dim(data)[2]

	row_tags <- c()
	# grab row_tags if na 
	if(any(is.na(data))){
		for(i in 1:n)
		{
			if(any(is.na(data[i,])))
			{
				row_tags <- append(i + 1,row_tags)
			}
		}
	}

	# check start value and calculate z_ig. 
	if ( is.matrix(start))
	{
		if(length(G) != 1){
			stop("Initialization z matrix should only be for a single G")
		}
		if(dim(start)[1] != dim(data)[1])
		{
			stop("Initialization z matrix should have the same number of rows as data")
		}
		if(dim(start)[2] != G)
		{
			stop("Initialization z matrix should have G number of columns")
		}
	}
	else if ( start == 0){
		startobject <- "random_soft" # check for validity of start object. 
	} 
	else if ( start == 1){
		startobject <- "random_hard" 
	}
	else if ( start == 2){
		startobject <- "kmeans"
	} 
	else{
		stop("start setting is not valid")
	}
	# check startup. 
	in_zigs <- switch(startobject,
					"random_soft"=z_ig_random_soft(n,G),
					"random_hard"=z_ig_random_hard(n,G),
					"kmeans"=z_ig_kmeans(data,G))

	# convert model_name to id. 
	model_name_id <- model_to_id(model_obj$cov_type)

	# pass into C++ call and return results. , one e_step functions the same for all models. 
	return_val <- switch(class(model_obj),
							"ghpcm_best" = gh_e_step_internal(X=data,G=G,
															model_id=model_name_id,
															model_type=model_name_id,
															in_m_obj=model_obj$model_obj[[1]],
															init_zigs=in_zigs,in_nu=nu),
							"stpcm_best" = st_e_step_internal(X=data,G=G,
															model_id=model_name_id,
															model_type=model_name_id,
															in_m_obj=model_obj$model_obj[[1]],
															init_zigs=in_zigs,in_nu=nu),
							"tpcm_best" = t_e_step_internal(X=data,G=G,
															model_id=model_name_id,
															model_type=model_name_id,
															in_m_obj=model_obj$model_obj[[1]],
															init_zigs=in_zigs,in_nu=nu),
							"vgpcm_best" = vg_e_step_internal(X=data,G=G,
															model_id=model_name_id,
															model_type=model_name_id,
															in_m_obj=model_obj$model_obj[[1]],
															init_zigs=in_zigs,in_nu=nu),														
							"gpcm_best" = e_step_internal(X=data,G=G,
															model_id=model_name_id,
															model_type=model_name_id,
															in_m_obj=model_obj$model_obj[[1]],
															init_zigs=in_zigs,in_nu=nu)
															)


	return_val$map = MAP(return_val$zigs)
	return_val$z <- return_val$zigs
	return_val$zigs <- NULL
	return(return_val)
}



# adjusted rand index similar to mclusts adjustedRandIndex
ARI <- function (x, y) 
{
  
  # attempt to cast as x and y 
  tryCatch({
    x <- as.vector(x)
    y <- as.vector(y)
    both <- c(x,y)
    if(any(is.na(both))){
      stop("There are NAS in the labels you have inputted.")
    }
    if (length(y) != length(x)) 
    {
      stop("labels you have inputted are of different lengths")
    }})


  if( length(x)==1)
  {
    if(x == y){
      return (1)
    }else{
     return(0) 
    }
  }
  
  xy <- table(x, y)
  A <- sum(choose(xy, 2))
  B <- sum(choose(rowSums(xy), 2)) - A
  C <- sum(choose(colSums(xy), 2)) - A
  D <- choose(sum(xy), 2) - sum(A,B,C)
  numerator <- A - sum(A,B) * sum(A,C)/sum(A,B,C,D)
  denominator <- sum(2*A,B,C)/2 - sum(A,B)*sum(A,C)/sum(A,B,C,D)
  ari <- numerator/denominator
  
  return(ari)
}



# Must provide either xs and ys (two partitions), or agree.tab (contingency table)
# If hard2soft = TRUE, then xs is a vector (hard assignment) and ys is a matrix (soft assignment)
# If hard2soft = FALSE, then xs and ys are both matrices (soft assignments)
# The contigency table can be provided with agree.tab instead of the two partitions.

sARI <- function(X = NULL, Y = NULL)
{
  # get type for X as it is pivotal for which calculation of sARI to use.  
  if(is.null(dim(X))){
    # case of vector in X 
    Xn <- length(X)
    if(is.null(Xn)){
      stop("Error: Xn was evaluated to be a vector, and no length was calculated, please put X as either a vector or matrix. ")
    }
    hard2soft <- TRUE; 
  } else {
    # X must be a matrix. 
    Xdd <- dim(X)
    Xn <- Xdd[1]
    Xg <- Xdd[2]
    hard2soft <- FALSE; 
  }
  
  # now check Y 
  if(is.null(dim(Y))){
    stop("Error Y must be a posterori table, i.e. an N times G matrix")
  } else {
    Ydd <- dim(Y)
    Yn <- Ydd[1]
    Yg <- Ydd[2]
  }
  
  if(Xn != Yn) {
    stop("Error: number of observations n for X and Y differ")
  }  
  
  
  xs = X; 
  ys = Y; 
  
  if(hard2soft == TRUE){
    xs = as.vector(xs)
    ys = as.matrix(ys)
    
    # Creating the contingency table
    # Rows are hard, Columns are soft
    sum.by.assn = function(hard, soft){aggregate(soft, list(hard), sum)}
    agree.tab = as.matrix(sum.by.assn(xs, ys))[,-1]
    N = sum(agree.tab)
    if(class(agree.tab) == "numeric") agree.tab = as.matrix(agree.tab, ncol = unique(xs))
  } else {
    xs = as.matrix(xs)
    ys = as.matrix(ys)
    
    agree.tab = matrix(NA, nrow = ncol(xs), ncol = ncol(ys))
    for(i in 1:ncol(xs)){
      temp.mat = xs[,i]*ys
      agree.tab[i,] = apply(temp.mat, 2, sum)
    }
    N = sum(agree.tab)
  }  
  
  
  if(all(dim(agree.tab) == c(1, 1)))
    return(1) 
  
  a <- (sum(agree.tab^2) - N)/2
  b <- (sum(apply(agree.tab, 1, sum)^2) - sum(agree.tab^2))/2
  c <- (sum(apply(agree.tab, 2, sum)^2) - sum(agree.tab^2))/2
  d <- (sum(agree.tab^2) + N^2 - sum(apply(agree.tab, 1, sum)^2) - sum(apply(agree.tab, 2, sum)^2))/2
  
  s_ret <- (choose(N, 2)*(a + d)-(((a + b)*(a + c))+(c + d)*(b + d)))/(choose(N, 2)^2 - (((a + b)*(a + c))+(c + d)*(b + d)))
  return(s_ret)  
  
}




