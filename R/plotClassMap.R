# ==============================================================================
# title:      plotClassMap.R
#
# summary:    plots the localized class map from a fitted caret model 
# ==============================================================================

setwd('../R/')
source('compLocalFarness.R')
source('compPAC.R')

# Quantile function: displays localized farnesson as quantiles of N(0,1)

qfunc = function(probs){
  # quantile function of N(0,1) restricted to [0,a]
  if(min(probs) < 0) stop("There are negative probs")
  if(max(probs) > 1) stop("there are probs > 1")
  afarness <- qnorm(probs*(pnorm(4)-0.5) + 0.5)
  return(afarness)
}


plotClassMap <- function(model,X, y,
                         class,
                         cols = NULL, 
                         k = NULL, 
                         main = NULL, 
                         comparison=NULL,
                         identify = FALSE){
  #
  # Arguments:
  #   model         : fitted model using caret::train
  #   X             : train or test data 
  #   y             : labels of train or test data
  #   class         : character, which class to plot
  #   cols          : vector of class colors. If NULL, it will be from rainbow()
  #   k             : # of nearest neighbors for Local Farness#
  
  # predictions from the model, as integer
  preds <- predict(model, X)
  predint <- as.integer(preds)
  
  # accuracy of the model
  acc <- round(mean(preds == y)*100, 2)
  
  # accuracy of the class
  class_acc <- round((mean(preds[which(y == class)] == class))*100,2)

  # get the PAC from the model
  PAC <- compPAC(model, X, y)

  # the number of classes present in the trained model
  class_labels <- levels(model$trainingData$.outcome)
  
  # get the farness from the data
  LF <- compLocalFarness(X = X, y = y, k = k)
  
  # variables for class map
  nlab = length(unique(y))
  if(is.null(cols)){cols = rainbow(nlab)}
  
  # scale farness
  afarness = qfunc(LF[y == class])
  
  aLF = afarness

  # plot parameters
  par(pty = "s")
  plot.new()
  
  # gray rectangle
  rect(0-1,
       -2,
       1*2,
       0.5,col="lightgray", 
       border=FALSE)
  
  # title
  if (is.null(main)){
    main = c(paste0('localized class map: ', as.character(class), ' ', comparison),
             paste(#'Overall Acc.: ', as.character(acc), "%",
                   'class accuracy: ', as.character(class_acc), "%"))
  }
  
  # scatterplot
  par(new=T)
  par(pty = "s")
  plot(y = PAC[y == class],
       x = aLF,
       cex = 1.3,
       
       main = main,
       
       # colors 
       col = cols[predint[y == class]],
      
       # points
       pch = 19,
       
       # x-axis
       xlim = c(0, 4),
       xaxt = "n",
       cex.axis = 1,
       #xlab = 'localized farness',
       xlab = "",
       
       # y-axis
       #ylab = 'P[alternative class]',
       ylab = "",
       ylim = c(0,1)
       )
  
  # place the axis titles
  title(xlab = "localized farness", mgp = c(2.25, 2.25, 0))            
  title(ylab = "P[alternative class]", mgp = c(2.25, 2.25, 0))         
  
  # localized farness
  probs = c(0, 0.5, 0.75, 0.9, 0.99, 0.999, 1)
  axis(1, 
       at = qfunc(probs), 
       labels = probs,
       cex.axis = 1, las = 1)
  
  # lines
  abline(h = 0)
  abline(h = 1)
  abline(v = qfunc(1))
  abline(v = qfunc(0.99), lty = 'dotted')

  
  if(identify == TRUE) { identify(aLF,PAC[y == class]) }
}











