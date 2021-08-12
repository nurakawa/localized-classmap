# ==============================================================================
# title:      compLocalFarness.R
#
# summary:    function to compute Local Farness of objects in a dataset.
#
#             The Local Farness (LF) of an object is defined as the scaled 
#             difference between the probability that it does not belong to its 
#             own class and the probability that it does belong to its own 
#             class. The probability is measured within a local neighborhood of
#             the object.
#
#             Class probability is measured by an epanechnikov kernel centered 
#             at the object. The kernel assigns weights to the 
#             object's neighbors, with higher weights assigned to nearer 
#             neighbors. The probability is then computed with respect to the
#             class labels of the neighbors.
# 
#             Epsilon, the width of the epanechnikov kernel, determines the size 
#             of the neighborhood taken into account. In a previous version,
#             we used a single epsilon for all points.
#
#             Currently, we defined epsilon_i for each object i to be the 
#             distance to the kth nearest neighbor. To get epsilon_i, we run a
#             k-nearest neighbor search using a k-d tree. For a dataset with a 
#             large number of rows (e.g. 30,000+ rows) it can be slow. However,
#             it is only performed once. It also removes the need for computing
#             a matrix of pairwise distances, since the nearest neighbor search
#             returns a matrix of distance to each nearest neighbor for each
#             object i.
# ==============================================================================

require(FNN) # nearest neighbor search with k-d tree or cover tree
require(rdd) # for function kernelwts, to compute epanechnikov kernel weights

compLocalFarness = function(X, y, k = NULL){

  #
  # Arguments:
  #   X           : numeric matrix or data frame with x_i as rows; 
  #                 training or testing data
  #   y           : labels of each object
  #   k           : number of nearest nns to consider when computing local
  #                 farness. default is min(10, round(sqrt(nrow(X)),0))
  #
  
  # ######
  # Checks
  # ######
  # convery y to integer
  if (class(y) != "integer"){
    yint <- as.integer(y)
  }
  # impute NA values with median
  if (sum(is.na(X)) > 0){
    print('Imputing Missing Values with Median')
    require(tidyverse)
    X <- X %>%
      mutate_all(funs(ifelse(is.na(.),median(., na.rm = TRUE),.)))
  }
  
  
  
  # #################
  # Compute epsilon_i
  # #################
  

  # default value of k
  if(is.null(k)){ 
    k <- min(10, round(sqrt(nrow(X)),0))
    print('Using default value for k')
  }
  else{
    print(paste('Nearest Neighbor Search for ', as.character(k), 
    ' nearest neighbors. This can take some time.'))}
    
    nns <- FNN::get.knn(X,
                   k=k,
                   algorithm ="kd_tree")
    
    # this returns the euclidean distances to the k nearest neighbors
    # and the indices of the k nearest neighbors
    # this is two n*k matrices.
    
    print('Nearest Neighbor Search Complete!')
    

  # epsilon_i is the euclidean distance to the kth nearest neighbor
  epsilon_arr = nns$nn.dist[,k]
  
  # ######################################
  # Epanechnikov Kernel Weighting Function
  # ######################################
  ep_kernel <- function(x, eps){rdd::kernelwts(x, center=0, bw=eps,
                                          kernel = 'epanechnikov')}
  
  
  # #############
  # Local Farness
  # #############
  nlab = length(unique(yint)) # number of labels
  n = nrow(X) # number of data points
  class_probs <- matrix(NA, nrow = n, ncol = nlab) # initialize class probs
  local_farness <- rep(NA, n) # initialize local farness
  
  # class probabilities
  for(i in 1:n){
    # for observation i, the vector of distances
    
    local_dists <- nns$nn.dist[i,]
    
    # epanechnikov kernel weighted distances
    # NOTE had to add +0.001 for stability issue
    wts <- ep_kernel(local_dists, eps=epsilon_arr[i]+0.001) 
    

    for(cl in 1:nlab){
      
      # classmate: someone in the neighborhood of the same class
      n_classmates <- sum(yint[nns$nn.index[i,]]==cl)
      
      # NA handllng
      if(is.na(n_classmates)){
        class_probs[i,cl] = NA
      }
      # case: no classmates, then class prob is 0
      else if(n_classmates == 0){
        class_probs[i,cl] = 0
      }
      # case: homogeneous neighborhood, then class prob is 1
      else if(n_classmates >= k){
        class_probs[i,cl] = 1
      }
      else{
        # for debugging
        #sumwts <- sum(wts[which(yint[nns$nn.index[i,]]==cl)])
        #if(is.na(sumwts)){print(wts[which(yint[nns$nn.index[i,]]==cl)])}
        
        # Pr(i in cl) = sum of weights of neighbors in class cl
        class_probs[i,cl] = sum(wts[which(yint[nns$nn.index[i,]]==cl)])
      }
    }
  }
  
  
  # get localized farness from class probabilities
  for(i in 1:n){
  
    # cl is the label of object i
    cl = yint[i]
    
    # local farness: prob of not own class - prob of own class
    
    # case: multi-class classification
    if(ncol(class_probs)>2){
      local_farness[i] = (sum(class_probs[i,-cl]) - class_probs[i,cl])
      #                  pr of NOT OWN CLASS     - pr OWN class
    }
    # case: binary classification
    else{
      local_farness[i] = (class_probs[i,-cl] - class_probs[i,cl])
      #                  pr of NOT OWN class - pr of OWN class 
    }
  }
  
  
  # rescale to make range 0-1
  local_farness <- (local_farness + 1)/2
  
  
  # uncomment if issue occurs, but so far it does not happen anymore
  # sometimes we get LF that is very very small, e.g. -1.110223e-16
  
  #if(min(local_farness, na.rm = T) < 0){
  #  local_farness <- sapply(local_farness, 
  #                         function(x){if (x < 0){x = 0}else{x = x}})
  #}

  return(local_farness)
}  


