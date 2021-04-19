
# MAIN GHPCM FUNCTION
ghpcm <- function(data=NULL,  G=1:3, mnames=NULL, # main inputs with mnames being the Model Name 
				start=2, label=NULL, # starting inputs , start = 0: random soft, start = 2, random hard. start = 3 mkmeans. 
				veo=FALSE, da=c(1.0), # veo (variables exceed observations), da is deterministic annealing  
				nmax=1000, atol=1e-8, mtol=1e-8, mmax=10, burn=5, # convergence settings for matrix and loglik
				pprogress=FALSE, pwarning=FALSE, stochastic = FALSE)  # progress settings 
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
		else if(all(label == 0)) { stop("There is no need a vector with all 0!, just don't pass in anything")} # check for stupidity. 
		else if(length(label) != n) { stop("label vector has to be the same size as the number of observtions. ")} # check length. 
		else if(length(G) != 1) { stop("Labels can only be inputted in for a single G") } # check for single input G. 
	}

	# go through all G's 
	for(G_i in G)
	{
		# go through all models
		for(model_name in mnames)
		{
			if(pprogress) {	cat("Running GHPCM Model:",model_name,"G=",G_i,"\n")}
			# calculate number of parameters once. 
			number_of_params <- npar.model.skew(model_name,G=G_i,p=p,family_name = "GH")

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
								classif_vector[label_i] <- 5 # put one 
								in_zigs[i,] <- classif_vector # replace
							}
							i <- i + 1
						}
					}
				}
				else{
					in_zigs <- as.matrix(rep(1.0,n))
				}

			  # RUN MODEL 
			  model_results_i <- main_loop_gh(X = t(data),
			                                  G = G_i, in_zigs = in_zigs, model_id = model_id_stochastic_check,
			                                  model_type = model_id + stochastic*20, in_nmax = nmax, in_l_tol = atol,
			                                  in_m_iter_max = mmax, in_m_tol = mtol, anneals = da,
			                                  t_burn = burn)
			  status <- "Failed Aitken's Convergence Criterion"
			  if (nmax > length(model_results_i$logliks)) {
			    status <- "Converged according to Aitken's Convergence Criterion"
			  }
				if(pwarning && !is.null(model_results_i$Error) ){
					cat(paste(model_name,"| G =",G_i,":",model_results_i$Error),"\n")
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
			  else {
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
	info_matrix$best_model <- gh_get_best_model(info_matrix)
	info_matrix$BIC <- construct_BIC_3D(info_matrix)
	info_matrix$map <- MAP(info_matrix$best_model$model_obj[[1]]$zigs)
	info_matrix$best_model$map <- info_matrix$map 


    gpar= list()
	for (k in 1:info_matrix$best_model$G ) {
		gpar[[k]] = list()		
		gpar[[k]]$mu       = info_matrix$best_model$model_obj[[1]]$mus[[k]]
		gpar[[k]]$sigma    = info_matrix$best_model$model_obj[[1]]$sigs[[k]]
		gpar[[k]]$alpha    = info_matrix$best_model$model_obj[[1]]$alphas[[k]]
		gpar[[k]]$invSigma = try(solve(info_matrix$best_model$model_obj[[1]]$sigs[[k]]))
		gpar[[k]]$logdet   = info_matrix$best_model$model_obj[[1]]$log_dets[k]

	}

	gpar$pi = info_matrix$best_model$model_obj[[1]]$pi_gs

	info_matrix$gpar <- gpar
	info_matrix$z <- info_matrix$best_model$model_obj[[1]]$zigs




	class(info_matrix) <- "ghpcm"

	return(info_matrix)

}


# PRINT SUMMARY AND PLOT STATEMENTS
print.ghpcm<-function(x, ...){
  # split strings and parse
	splitted_strings <- strsplit(x$best_model$model_type," ")[[1]]
  # print to line parsed strings. 
  cat("The model chosen by applying the BIC criteria has", trimws(splitted_strings[5]), "component(s) and the", trimws(splitted_strings[2]), "covariance structure\n using", x$startobject, "\n"  )
}
# just prints the compare BIC matrix. 
summary.ghpcm<- function(object, ...){
    cat("BIC for each model, number of components (rows), and covariance structure (columns).\n")
	print(object$BIC[,,3])
}

# plots a line graph of BICs. 
plot.ghpcm<- function(x, ...) {
	bicl = x$BIC[,,3]
	# you need to wrap this plot up in a print statement otherwise it doesnt work. 
	print(levelplot(bicl,
          col.regions = colorRampPalette(c("black","brown","red","orange","gold","yellow","white"))(prod(dim(bicl)) + 10),
          xlab = "G",ylab = "Covariance Type"))
	trellis.focus("legend", side="right", clipp.off=TRUE, highlight=FALSE)
	trellis.unfocus()	
}



# Print output for the gpcm_best class. 
print.ghpcm_best <-function(x, ...){

	splitted_strings  <- strsplit(x$model_type," ")[[1]]
	cov_string <- splitted_strings[2]
	component_string <- splitted_strings[5]

	cat("============================\n")
    cat("Best GHPCM Model According To BIC \n")
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
gh_get_best_model <- function(gpcm_model)
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
  
  class(best_model) <- "ghpcm_best"
  
  return(best_model)
  
}
