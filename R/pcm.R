

# global wrapper function for running all possible models across all famalies. 


pcm <- function(data=NULL,  G=1:3, pcmfamily=c(gpcm,vgpcm,tpcm), # just pass in the functions you want to run, default is to run them all. 
                mnames=NULL, # main inputs with mnames being the Model Name 
                start=2, label=NULL, # starting inputs , start = 0: random soft, start = 2, random hard. start = 3 mkmeans. 
                veo=FALSE, da=c(1.0), # veo (variables exceed observations), da is deterministic annealing  
                nmax=1000, atol=1e-8, mtol=1e-8, mmax=10, burn=5, # convergence settings for matrix and loglik
                pprogress=FALSE, pwarning=FALSE) 
  
{
  # the functions already have sanity checks so all I have to do is figure out 
  # a way to return the best model 
  all_results = list()  
  best_bic = -Inf
  best_model = list()
  # iterate through all family functions
  for(pc_function in pcmfamily)
  {
      results = pc_function(data=data,G=G,mnames=mnames,start=start,label=label,veo=veo,
                            da=da,nmax=nmax,atol=atol,mtol=mtol,mmax=mmax,burn=burn,pprogress=pprogress,
                            pwarning=pwarning)
      family_string <- class(results)
      if(family_string == "gpcm"){
        all_results <- append(all_results, list("gpcm" = results))
      }
      if(family_string == "stpcm"){
        all_results <- append(all_results, list("stpcm" = results))
      }
      if(family_string == "ghpcm"){
        all_results <- append(all_results, list("ghpcm" = results))
      }
      if(family_string =="vgpcm"){
        all_results <- append(all_results, list("vgpcm" = results))
      }
      if(family_string =="tpcm"){
        all_results <- append(all_results, list("tpcm" = results))
      }
  }
  
  for(models in all_results)
  {
    if(models$best_model$BIC > best_bic){
      best_model <- models
      best_bic <- best_model$best_model$BIC
    }
  }

  all_results$best_model = best_model

  class(all_results) <- "pcm"  
  return(all_results)
} 



# PRINT AND SUMMARY STATEMENTS
print.pcm <-function(x, ...){
  print(x$best_model$best_model)
}
# just prints the compare BIC matrix. 
summary.pcm <- function(object, ...){
  family_names = names(object)
  for(i in 1:(length(family_names)-1)){
    cat("==============================================================================================\n")
    cat(paste("FAMILY:",family_names[i],"\n"))
    cat("==============================================================================================\n")
    summary(object[[i]])
  }
  cat("==============================================================================================\n")
}



