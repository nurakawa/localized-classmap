rm(list = ls())
dev.off()
require(caret)
source("../R/plotClassMap.R")

# data
data("iris")
head(iris)

X <- iris[,-ncol(iris)]
y <- factor(iris$Species)

# train / test split
set.seed(1234)
trainIndex <- createDataPartition(y, p = 0.7,
                                  list = FALSE, 
                                  times = 1)

X_train <- X[trainIndex,]; y_train <- y[trainIndex]
X_test <- X[-trainIndex,]; y_test <- y[-trainIndex]

# fit model: Support Vector Machine
model <- caret::train(x = X_train,
                      y = y_train,
                      method = "svmLinear",
                      trControl = trainControl(
                        classProbs = TRUE # required for PAC!
                      ))

# view model
model # 96% accuracy

######################################
# Localized Class Map: Training Data #
######################################

# class versicolor
plotClassMap(model = model,
             X = X_train,
             y = y_train,
             class = "versicolor",
             cols = c('blue', 'red', 'orange'))

# class setosa
plotClassMap(model = model,
             X = X_train,
             y = y_train,
             class = "setosa",
             cols = c('blue', 'red', 'orange'))

# class virginica
plotClassMap(model = model,
             X = X_train,
             y = y_train,
             class = "virginica",
             cols = c('blue', 'red', 'orange'))


##################################
# Localized Class Map: Test Data #
##################################

# class versicolor
plotClassMap(model = model,
             X = X_test, # input the test set instead of the training set
             y = y_test, # test labels
             class = "versicolor",
             cols = c('blue', 'red', 'orange'))

# class setosa
plotClassMap(model = model,
             X = X_test,
             y = y_test,
             class = "setosa",
             cols = c('blue', 'red', 'orange'))

# class virginica
plotClassMap(model = model,
             X = X_test,
             y = y_test,
             class = "virginica",
             cols = c('blue', 'red', 'orange'))

#########################
# Different Values of k #
#########################

# k = 3
plotClassMap(model = model,
             X = X_test, 
             y = y_test, 
             class = "versicolor",
             k = 3,
             cols = c('blue', 'red', 'orange'))


# k = 5
plotClassMap(model = model,
             X = X_test, 
             y = y_test, 
             class = "versicolor",
             k = 5,
             cols = c('blue', 'red', 'orange'))


# k = 10
plotClassMap(model = model,
             X = X_test, 
             y = y_test, 
             class = "versicolor",
             k = 10,
             cols = c('blue', 'red', 'orange'))

# k = 20
plotClassMap(model = model,
             X = X_test, 
             y = y_test, 
             class = "versicolor",
             k = 20,
             cols = c('blue', 'red', 'orange'))
