  # ==============================================================================
  # title:      example_amazon.R
  #
  # summary:    Reproduces the Amazon book reviews example of VCR_example_script,
  #             but with a localized class map. The goal is to show that localized
  #             farness is useful in scenarios where class labels leave room for
  #             ambiguity and nuance.
  #
  # please note:  This example is a bit slow. 
  # ==============================================================================
  
  ########
  # Setup
  ########
  rm(list=ls())
  
  source("plotClassMap.R")
  library(kernlab)
  library(e1071)
  library(caret)
  library(classmap)
  
  
  #########################
  # Amazon book review data
  #########################
  
  load("../data/Amazon_bookReviews.rdata") # not extremely big: 1,343 KB
  
  # The dataset was assembled by Prettenhofer and Stein (2010)
  # https://arxiv.org/abs/1008.0716
  # 
  # The dataset has been used for a variety of things, including 
  # classification using svm.
  # A list of citations:
  # https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&
  #   sciodt=0%2C5&cites=4738650459241064524&scipsc=1&q=svm+&btnG=
  #
  # More on string kernels, and their application to these data:
  # http://personales.upv.es/prosso/resources/GimenezEtAl_NPACB18.pdf 
  
  length(X_train); length(X_test); length(y_train); length(y_test) 
  # all 2000
  yint = y_train + 1
  yintnew = y_test+1
  levels = c("1","2")
  cols = c("deepskyblue3","red")
  
  table(yint, y_train) # yint 1 == NEGATIVE, yint 2 == POSITIVE
  
  # The best kernel for these data was the spectrum kernel 
  # with length 7:
  strdot = kernlab::stringdot(type = "spectrum", length = 7)
  strdot
  
  load("../data/Amazon_bookReviews_Kr_full.rdata")
  
  dim(Kr_full) # 4000 4000 # huge matrix
  
  # The kernel matrix of the training data:
  Kr_train = Kr_full[1:2000,1:2000]
  dim(Kr_train) # 2000 2000
  
  # Compute the corresponding points in feature space:
  ptm = proc.time()
  outFV = classmap::makeFV(Kr_train)
  # The kernel matrix has 15 eigenvalues below precS = 1e-12.
  # Will take this into account.
  (proc.time()-ptm)[3] # took 70 secs
  #
  Xf = outFV$Xf # Xf is in feature space
  # Xf can be seen as the scores of a kernel PCA
  # without centering (mean subtraction).
  dim(Xf) # 2000 1985
  
  # To see how well the feature vectors and Kr_train match:
  inprod = Xf%*%t(Xf) # the inner products of the Xf
  max(abs(as.vector(Kr_train-inprod))) # 2.863312e-10 # OK
  plot(apply(abs(Kr_train - inprod),1,max))
  # Deviation of each row of Kr_train from the inner products
  
  range(rowSums(Xf^2)) # all 1
  # The points of Xf lie on the unit sphere, in 1985
  # dimensions. This resembles the radial basis kernel.
  
  plot(Xf[,1:2], col=yint+1) # nontrivial shape, 2 clusters
  # but not separated by class
  pairs(Xf[,1:5], gap=0, col=yint+1)
  #
  plot(apply(Xf,2,var))
  # By construction, the variables of Xf are uncorrelated.
  # How many variables do we need to explain most of the 
  # variance of Xf?
  explvar = cumsum(apply(Xf,2,var))
  plot(explvar/explvar[ncol(Xf)]); abline(h=0.95)
  min(which(explvar > 0.95*explvar[ncol(Xf)])) # 1803
  # so we are in a truly high dimensional situation.
  
  # Now train the support vector classifier on these data.
  # The combination (length=7,cost=2) was the result of
  # 10-fold cross validation, not shown here.
  Xfdat = data.frame(X = Xf, y = as.factor(yint))
  #
  # On Xf we train the SVM with the _linear_ kernel, since
  # we are in feature space:
  set.seed(1) # we have to fix the seed!
  ptm = proc.time()
  svmfit = svm(y~.,data=Xfdat,scale=F,kernel="linear",cost=2,
               probability=T)
  (proc.time()-ptm)[3] 
  
  plot(svmfit$decision.values) # the two horizontal lines
  # contain the support vectors. There are a lot of them,
  # since the dimension is high.
  
  
  ptm = proc.time()
  vcr.train = vcr.svm.train(Xf,
                            factor(yint),
                            levels,
                            svfit=svmfit)
  (proc.time()-ptm)[3] # 13 sec
  
  confmat.vcr(vcr.train, showOutliers = F)
  # 
  # Confusion matrix:
  #          predicted
  # observed     1    2
  #        1  1000    0
  #        2     0 1000
  # 
  # The accuracy is 100%.
  
  # This classification is `perfect' due to overfitting!
  
  stackedplot(vcr.train, classCols = cols)
  
  # Class map of the negative reviews in the training data, 
  #
  par(mar = c(3.5, 3.5, 2.0, 0.2))
  classmap(vcr.train, 1, classCols = cols,
           main = "predictions of the negative book reviews",
           cex = 1.5, cex.lab = 1.5, cex.axis = 1.5,
           cex.main = 1.5, identify = FALSE)
  
  # Class map of the positive reviews in the training data, 
  #
  classmap(vcr.train, 2, classCols = cols,
           main = "predictions of the positive book reviews",
           cex = 1.5, cex.lab = 1.5, cex.axis = 1.5,
           cex.main = 1.5, identify = FALSE)
  
  ###################################
  # What do we find in the test set?
  ###################################
  
  Kr_test = Kr_full[-c(1:2000), 1:2000] 
  
  # Compute feature vectors of the test set:
  ptm = proc.time()
  FVtest = makeFV(kmat=Kr_test, transfmat=outFV$transfmat)
  (proc.time()-ptm)[3] # 7 seconds
  Zf = FVtest$Xf # Zf lives in the same space as the
  # Xf of the training data
  dim(Zf) # 2000 1985
  
  # Compare the inner products with the kernel matrix:
  inprod = Zf%*%t(Xf) # products of Zf with Xf
  max(abs(as.vector(Kr_test-inprod))) # 1.451722e-10 # OK
  plot(apply(abs(Kr_test - inprod),1,max))
  # = deviation of each row of Kr_test from inner products
  
  # Predict with the SVM fit from the training data:
  ptm = proc.time()
  pred_test = predict(svmfit,newdata=data.frame(X=Zf),
                      probability=T)
  (proc.time()-ptm)[3] # 11 seconds
  
  
  ptm = proc.time()
  vcr.test = vcr.svm.newdata(Zf,
                             factor(yintnew),
                             vcr.train)
  (proc.time()-ptm)[3] # 21 sec
  
  confmat.vcr(vcr.test, showOutliers=F) # Now it is realistic:
  # 
  # Confusion matrix:
  #         predicted
  # observed   1   2
  #        1 811 189
  #        2 171 829
  # 
  # The accuracy is 82%.
  
  # 82% correct classification, very close to the numbers (table 1) in
  # http://personales.upv.es/prosso/resources/GimenezEtAl_NPACB18.pdf 
  
  confmat.vcr(vcr.test) # 2 outliers
  #        predicted
  # observed   1   2 outl
  #        1 810 188    2
  #        2 171 829    0
  
  stackedplot(vcr.test,classCols=cols,separSize=1.5,minSize=1.5)
  
  # Class map of the negative reviews in the test set,
  par(mar = c(3.5, 3.5, 2.0, 0.2))
  classmap(vcr.test, 1, classCols = cols,
           main = "predictions of the negative book reviews",
           cex = 1.5, cex.lab = 1.5, cex.axis = 1.5,
           cex.main = 1.5)
  
  # Class map of the positive reviews in the test set,
  par(mar = c(3.5, 3.5, 2.0, 0.2))
  classmap(vcr.test, 2, classCols = cols,
           main = "predictions of the positive book reviews",
           cex = 1.5, cex.lab = 1.5, cex.axis = 1.5, cex.main = 1.5)
  
  
  # -------------------
  # localized class map
  # -------------------
  
  # caret supports string kernels, including the spectrum kernel
  # an example of using string kernels in caret
  # https://github.com/topepo/caret/blob/master/RegressionTests/Code/svmSpectrumString.R
  
  # however, localized farness needs a numeric matrix for nearest neighbor search.
  # therefore, the best way to do this is to compute the kernel matrix from 
  # X using kernlab, get the feature vectors from VCR_code.R,
  # and then fitting a linear SVM using caret.
  
  # convert y to factor with non-numeric levels: required for caret model
  #y = factor(sapply(y_train,function(x){if(x==0){"positive"}else{"negative"}}))
  y = factor(sapply(yint,function(x){if(x==2){"positive"}else{"negative"}}))
  
  
  # check if it's right
  table(data.frame(y,y_train)) # good
  table(data.frame(y,yint)) # good
  
  # caret model with the train data
  # method = "svmSpectrumString" uses a spectrum kernel
  # we use length 7 and cost 2 as in the svfit model
  
  # I set the grid to be the parameters that are already used in svfit
  # I input the kernel matrix. 
  # 
  
  # required for caret
  colnames(Kr_train) <- paste0("X",1:ncol(Kr_train))
  colnames(Xf) <- paste0("X",1:ncol(Xf))
  
  
  
  ptm <- proc.time()
  set.seed(1)
  model <- caret::train(x = Xf,
                        y = y,
                        method = 'svmLinear2', # this uses e0171, like svm()
                        scale = F,
                        trControl = trainControl(method = "none",
                                                 classProbs = TRUE), 
                        tuneGrid = data.frame("cost" = 2))
  proc.time() - ptm
  
  # not too slow, since we already use tuned parameters
  
  model$finalModel # training error is 0, so it was overfit
  
  # > model$finalModel$cost
  # [1] 2
  # > model$finalModel$gamma
  # [1] 0.0005037783
  # > model$finalModel$sigma
  # [1] 0
  
  # exactly the same as svmfit:
  svmfit$cost
  svmfit$gamma
  svmfit$sigma
  
  # same number of support vectors as well
  svmfit
  model$finalModel
  
  
  par(mfrow=c(1,1))
  # the classifier overfit on the training data, so it makes sense that
  # the class map is really bad
  
  #############################################
  # localized class maps on the training data #
  #############################################
  
  # note: for model with type "svmLinear2" it is slow.
  # but for type "svmLinear" it is faster
  
  source('plotClassMap.R')
  
  #pdf('../pdf/amazon-localized-classmap-train-positive.pdf', width = 6, height = 6.25)
  par(mar = c(3.5, 3.5, 2.0, 0.2))
  plotClassMap(model, 
               Xf, 
               y,  
               class = "positive",
               k = 7, 
               cols = cols)
  #dev.off()
  
  #pdf('../pdf/amazon-localized-classmap-train-negative.pdf', width = 6, height = 6.25)
  plotClassMap(model, 
               Xf, 
               y, 
               class = "negative",
               k = 7, 
               cols = cols)
  #dev.off()
  
  # add colnames to Zf, must be the same as colnames of Xf
  colnames(Zf) <- paste0("X",1:ncol(Zf))
  
  # convert y_test
  y_t= factor(sapply(yintnew,function(x){if(x==2){"positive"}else{"negative"}}))
  
  #########################################
  # localized class maps on the test data #
  #########################################
  #pdf('../pdf/amazon-localized-classmap-test-positive.pdf', width = 6, height = 6.25)
  par(mfrow=c(1,1))
  # positive
  par(mar = c(3.5, 3.5, 2.0, 0.2))
  plotClassMap(model, 
               Zf, 
               y_t, 
               class = "positive",
               k = 7, 
               cols = cols)
  legend("topright", fill = cols,
         legend = c("negative", "positive"), cex = 1,
         ncol = 1, bg = "white")
  # we can compute localized farness and PAC outside of the localized class map
  LF <- compLocalFarness(Zf, y_t, k = 7)
  PAC <- compPAC(model, Zf, y_t)
  # to identify points
  # identify(qfunc(LF), PAC)
  
  # (a) 115, (b) 902, (c) 975 (d) 184 (e) 268
  #xvals = qfunc(LF[c(115, 902, 975)])-0.15 #, 1332, 1708)])-0.1
  #yvals = PAC[c(115, 902, 975)] #, 1332, 1708)]
  xvals = c(2.242829, 1.508483, 2.109286, 1.018368 ,1.452520)
  yvals = c(0.92866527, 0.58665700, 0.08889521, 0.9795014 ,0.1349954)
  text(x = xvals, y = yvals, labels = letters[1:length(xvals)], cex = 1.2)
  #dev.off()
  #
  #
  # class map of the negative reviews, test set
  #pdf('../pdf/amazon-localized-classmap-test-negative.pdf', width = 6, height = 6.25)
  par(mar = c(3.5, 3.5, 2.0, 0.2))
  plotClassMap(model, 
               Zf, 
               y_t, 
               class = "negative",
               k = 7, 
               cols = cols)
  legend("bottom", fill = cols,
         legend = c("negative", "positive"), cex = 1,
         ncol = 1, bg = "white")
  # to identify points
  # identify(qfunc(LF), PAC)
  
  # (a) 740 (b) 1251 (c) 1515 (d) 1898
  #xvals = qfunc(LF[c(740, 1251, 1515, 1898)])-0.15
  #yvals = PAC[c(740, 1251, 1515, 1898)]
  xvals = c(3.850000, 2.399333, 3.850000, 3.850000)
  yvals = c((0.5769461-0.05), 0.8844529, 0.9362500, 0.5878015)
  text(x = xvals, y = yvals, labels = letters[1:length(xvals)], cex = 1.2)
  #dev.off()
  
  # ------------
  # sanity check
  # ------------
  # compare the PAC from the original class map to that
  # of the localized class map
  
  sanity_check <- data.frame('vcrPAC' = vcr.test$PAC,
             'myPAC' = PAC)
  
  sanity_check$vcrPAC - sanity_check$myPAC
  # ok, they are almost identical
  
  
  # Table for POSITIVE class map
  
  # (a): 115 highest LF, but lowest PAC
  X_test[115] # it's locally far from other positive reviews.
  # why so? it's quite nuanced
  
  # "if you are the kind of person who want to quickly assimilate and regurgitate 
  # the matter for cissp, then dont even bother."
  #  "and you'd expect this kind of book to live on your shelf for a long long 
  # time than 'teach yourself crap in 24 hours' books, but the quality of paper 
  # will make that unlikely."
  #"hence i am giving 4 stars to a book which otherwise would deserve 6 star"
  # 
  
  # (b) 902 similar to a - it is nuanced
  X_test[902]
  # "the book's main drawback comes in its unappealing characters. most readers like either sympathetic characters or disgusting villains. this book lacked both."
  # "but any long-time fan will enjoy the book. new fans should go back to the beginning of the series and delay this book. "
  
  # (c) very short
  X_test[975] # I like the book very muc
  nchar(X_test[975]) # the shortest review in the test set
  
  summary(nchar(X_test)) 
  # Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  # 22.0   323.8   555.5   864.7  1011.2 11372.0 
  
  
  # (d)
  X_test[184]
  # 184: I would classify this as negative
  # ``the ending just wasn't believable."
  # ``wait for it in the bargain bin"
  
  # (e)
  X_test[268]
  # 268: I would also say it is negative
  # ``this book lacked substance and depth"
  # ``so many missing pieces that could have explained more clearly what led up to the revolution"
  
  
  # Table for NEGATIVE review
  # (a) 740 (b) 1251 (c) 1515 (d) 1898
  
  # (a)
  X_test[740] # a personal opinion, with only one negative phrase
  # very long
  nchar(X_test[740]) # 2023
  
  # "rather than wooldridge's fear and resentment-mongering, 
  # i'd recommend the upcoming boook by free speech radio news host..." 
  # 
  
  # (b)
  X_test[1251] # possibly mislabeled! 
  # [1] "this book is the best i ever read.it tells about all the animals that 
  # live there.a short man wacks tree,all the animals are sad that their family
  # will live without tree's"
  
  # (c)
  X_test[1515] # mislabeled
  # [1] "this book is an awesome reference tool. it arrived in perfect condition !"
  y_test[1515] # 0
  y_t[1515] # neg
  # so really this is labeled as negative, but the text is not at all negative!
  
  # (d)
  X_test[1385] # also a person's political opinion
  nchar(X_test[1385]) # 2045
  
  # "this book tries to explain how and why the media ..."
  # " however, the whole matter is subject to the particular
  # perspective preference of the readers personal comprehension."
  # "rushkoff, although being quite informative nevertheless from time to time 
  # with the book only seem to mainly dwell on a surface level, when what we need 
  # is to see more of the primal motivating factors behind the dissemination of 
  # counter-culture trends and ideas which rushkoff attempted to disseminate 
  # within the length of his work."
  
  
  
