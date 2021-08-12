# ==============================================================================
# title:      compPAC.R
#
# summary:    function to compute Probability of Alternate Classification (PAC) 
#             of objects in a dataset.
#
#             PAC is defined in Raymaekers, Rousseeuw and Hubert (2021)
#             <doi:10.1080/00401706.2021.1927849>
# 
# ==============================================================================

compPAC <- function(model, X, y){
  
  #
  # Arguments:
  #   model         : fitted model using caret::train
  #   X             : data, either the data used to fit the model or a test set
  #   y             : labels, either used to fit the model or a test set 
  #
  
  # variables for PAC computation

  n = nrow(X)
  PAC = rep(NA,n)

  yint <- as.integer(y)
  nlab = length(unique(yint)) # number of classes
  
  # get fitted model probabilities
  model_probs <- predict(model,X, type="prob")
  
  # two class case
  if (nlab == 2){
    
    # alternative class is the other class
    altint = 3 - yint # yint will take values 1 or 2
    
    # PAC: probability of the other class
    for(i in 1:n) PAC[i] = model_probs[i,altint[i]]
  }
  else{
    

    ptrue = palt = altint = rep(NA,n)
    probs = model_probs
    
    for(i in 1:n){
      # probability of its true label
      # NA handling
      if(is.na(yint[i])){
        PAC[i] = NA
        next}
      ptrue[i]   = probs[i,yint[i]]

      others     = (1:nlab)[-yint[i]] # indices of the other classes
      palt[i]   = max(probs[i,others]) # most likely alternative class
      altint[i] = others[which.max(probs[i,others])] # the alternative class
      PAC[i]      = palt[i]/(ptrue[i] + palt[i]) # conditional prob: PAC
    }
  }
    
    return(PAC)

}