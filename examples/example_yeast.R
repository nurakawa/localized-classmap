# ==============================================================================
# title:      example_yeast.R
#
# summary:    Using the yeast data from the UCI repository, I relabel it to an
#             imbalanced binary classification problem. I show an application of
#             the localized class map: to compare base classifiers. I also show
#             how it can give good direction as to what pre-processing methods
#             we can use to improve the classification.
# ==============================================================================

rm(list = ls())
#dev.off()

require(caret)
require(grDevices)
require(Rtsne)

source("../R/plotClassMap.R")

##############
# Yeast Data #
##############

# source: http://archive.ics.uci.edu/ml/datasets/yeast
yeast_data <- read.table("../data/yeast.data")
yeast_names <- c("sequence_name", 'mcg','gvh', 'alm', 'mit', 'erl',
                 'pox', 'vac', 'nuc', 'site')
names(yeast_data) <- yeast_names

str(yeast_data)

table(yeast_data$site) # in Napierala the minority class was "ME2"

# Change to binary classification
binary_site <- sapply(yeast_data$site, function(x){if(x != 'ME2'){x = "OTHER"}else{x="ME2"}})
binary_site <- factor(binary_site)
table(binary_site)/length(binary_site)*100 #imbalance ratio matches the paper
yeast_data[,'binary_site'] <- binary_site
yeast_data$sequence_name <- NULL
yeast_data$site <- NULL

X <- yeast_data[,-ncol(yeast_data)]
y <- yeast_data$binary_site


##############
# t-sne plot #
##############
#cols = c("gold3","deepskyblue")
cols = c('deepskyblue', 'gold3')
cols <- adjustcolor(cols, alpha = 0.7)

labels <- c("ME2", "OTHER")

set.seed(1234)
tsne_view <- Rtsne(X, dims = 2, check_duplicates = FALSE, 
                   perplexity = 30, pca = FALSE)
#pdf('../pdf/yeast-tsne-plot.pdf', width = 8, height = 8)
par(mfrow=c(1,1))
par(pty = 's')
plot(tsne_view$Y,
     cex = 1.4,
     pch = 19,
     col = cols[y],
     main = 't-SNE Embeddings of Yeast Data',
     xlab = '', ylab = '')
legend('topright',
       fill = cols[1:2],
       legend = labels, cex = 0.9,
       ncol = 1, bg = "white")
#dev.off()


###########################################
# Application: Comparing Base Classifiers #
###########################################

# J48: This is the open-source implementation of C4.5 Classification tree
# MLP: Multi-layer perceptron
# KNN: K-nearest neighbors classifier

methods <- c("J48", "mlp", "knn")

# Tune control
ctrl1 <- trainControl(method = "repeatedcv", repeats = 5,
                      classProbs = TRUE,
                      summaryFunction = twoClassSummary)

model_lst <- list("J48" = list(),
                  "mlp" = list(),
                  "knn" = list())

# Train/Test Split
set.seed(1234)
trainIndex <- createDataPartition(y, p = 0.7,
                                  list = FALSE, 
                                  times = 1)

X_train <- X[trainIndex,]; X_test <- X[-trainIndex,]
y_train <- y[trainIndex]; y_test <- y[-trainIndex]


# Train caret models, caret will tune parameters
for (m in methods){
  set.seed(1234)
  model_lst[[m]] <- caret::train(X_train,
                               y_train,
                               method = m,
                               metric = 'ROC',
                               trControl = ctrl1,
                               preProcess = c('center', 'scale'))}

# Confusion Matrix
# For each model we can generate a confusion matrix using caret:
for (i in 1:length(model_lst)){
  print("###################################")
  print(paste('Model: ', names(model_lst)[i]))
  print("###################################")
  # confusion matrix
  preds <- predict(model_lst[[i]], X_train)
  print(caret::confusionMatrix(reference = y_train, data = preds))
}

# Confusion Matrix
# For each model we can generate a confusion matrix using caret:
for (i in 1:length(model_lst)){
  print("###################################")
  print(paste('Model: ', names(model_lst)[i]))
  print("###################################")
  # confusion matrix
  preds <- predict(model_lst[[i]], X_test)
  print(caret::confusionMatrix(reference = y_test, data = preds))
}
##########################
# Class Maps: Train Data #
##########################

#pdf('../pdf/yeast-comparing-base-classifiers-train.pdf', width = 9, height = 6.5)

# Class Maps: Train Set
par(mfcol = c(2, 3))
for (i in 1:length(model_lst)){
  par(mar = c(0.5, 4.5, 2.5, 0.75))
 
   plotClassMap(model_lst[[i]], 
               X_train,
               y_train, 
               class = "OTHER",
               comparison = names(model_lst)[i],
               k = 10,
               cols = cols)

    
  plotClassMap(model_lst[[i]], 
               X_train, y_train, 
               class = "ME2",
               k = 10,
               cols = cols,
               comparison = names(model_lst)[i])
  
  
}
#dev.off()

#########################
# Class Maps: Test Data #
#########################


#pdf('../pdf/yeast-comparing-base-classifiers-test.pdf', width = 9, height = 6.5)

# Class Maps: Train Set
par(mfcol = c(2, 3))
for (i in 1:length(model_lst)){
  par(mar = c(0.5, 4.5, 2.5, 0.75))
  
  plotClassMap(model_lst[[i]], 
               X_test,
               y_test, 
               class = "OTHER",
               comparison = names(model_lst)[i],
               k = 10,
               cols = cols)
  
  
  plotClassMap(model_lst[[i]], 
               X_test, y_test, 
               class = "ME2",
               k = 10,
               cols = cols,
               comparison = names(model_lst)[i])
  
  
}
#dev.off()

