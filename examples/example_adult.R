# ==============================================================================
# title:      example_adult.R
#
# summary:    Using the Adult data from the UCI repository, I show how the 
#             class map can be used to identify algorithmic bias.
#
# data:       http://archive.ics.uci.edu/ml/support/Adult
# resource:   https://ocw.mit.edu/resources/res-ec-001-exploring-fairness-in-machine-learning-for-international-development-spring-2020/module-four-case-studies/case-study-mitigating-gender-bias/MITRES_EC001S19_video7.pdf 
# ==============================================================================


#########
# Setup #
#########
rm(list=ls())
#dev.off()
require(caret)
require(tidyverse)
require(ggplot2)
require(grDevices)
source('../R/plotClassMap.R')


#################
# Adult Dataset #
#################
adult_train <- read.csv('../data/adult.data', header = FALSE, stringsAsFactors = TRUE)
adult_test <- read.csv('../data/adult.test', header = FALSE, stringsAsFactors = TRUE,
                       skip = 1)

adult_names <- c('age', 'workclass', 'fnlwgt', 'education',
                 'education_num', 'marital_status', 'occupation',
                 'relationship', 'race', 'sex', 'capital_gain',
                 'capital_loss', 'hours_per_week', 'native_country',
                 'income')
names(adult_train) <- adult_names
names(adult_test) <- adult_names

#################
# Preprocessing #
#################

# ========= #
# Train Set #
# ========= #

# replace ? with NA
adult_train %>% 
  # age, fnlwgt, capital_gain, capital_loss, hours_per_week is numeric
  mutate(age = as.numeric(age),
         fnlwgt = NULL,
         capital_gain = NULL,
         capital_loss = NULL,
         relationship = NULL,
         hours_per_week = as.numeric(hours_per_week)) %>%
  # make native country binary
  mutate(native_USA = (native_country == " United-States")+0,
         native_country = NULL) %>%
  # make sex binary
  mutate(female = (sex == " Female" ) + 0,
         sex = NULL) %>%
  # make race white vs nonwhite
  mutate(raceWhite = (race == " White" ) + 0,
         race = NULL) %>%
  
  # remove education number
  mutate(education_num = NULL) -> adult2
   
# one-hot encoding 

workclass <- model.matrix( object = ~ workclass - 1,
                           data = adult2)
education <- model.matrix( object = ~ education - 1,
                           data = adult2)
marital_status <- model.matrix( object = ~ marital_status - 1,
                           data = adult2)
occupation <- model.matrix( object = ~ occupation - 1,
                           data = adult2)

tmp <- cbind(workclass, education, marital_status, occupation)


adult2 %>% 
  cbind(tmp) %>%
  apply(MARGIN=2,FUN = function(x){na_if(x," ?")}) %>%
  as.data.frame %>%
  select(-contains("?")) %>%
  mutate(workclass = NULL, 
           education = NULL, 
           marital_status = NULL,
           occupation = NULL ,
           race = NULL)   -> adult_train

# change the levels of income to high and low
levels(adult_train$income)


levels(adult_train$income) <- c('lt50k', 'ge50k')
head(adult_train)

# imbalance:
table(adult_train$income)/nrow(adult_train)

# fix the names
names(adult_train) %>%
  stringr::str_replace_all(" ", "_") %>%
  stringr::str_remove_all("-") -> names(adult_train)

# ======== #
# Test Set #
# ======== #

# same pre-processing as the train set

# replace ? with NA
adult_test %>% 
  # age, fnlwgt, capital_gain, capital_loss, hours_per_week is numeric
  mutate(age = as.numeric(age),
         fnlwgt = NULL,
         capital_gain = NULL,
         capital_loss = NULL,
         relationship = NULL,
         hours_per_week = as.numeric(hours_per_week)) %>%
  # make native country binary
  mutate(native_USA = (native_country == " United-States")+0,
         native_country = NULL) %>%
  # make sex binary
  mutate(female = (sex == " Female" ) + 0,
         sex = NULL) %>%
  # make race white vs nonwhite
  mutate(raceWhite = (race == " White" ) + 0,
         race = NULL) %>%
  # remove education number
  mutate(education_num = NULL) -> adult2

# one-hot encoding  

workclass <- model.matrix( object = ~ workclass - 1,
                           data = adult2)
education <- model.matrix( object = ~ education - 1,
                           data = adult2)
marital_status <- model.matrix( object = ~ marital_status - 1,
                                data = adult2)
occupation <- model.matrix( object = ~ occupation - 1,
                            data = adult2)

tmp <- cbind(workclass, education, marital_status, occupation)

adult2 %>% 
  cbind(tmp) %>%
  apply(MARGIN=2,FUN = function(x){na_if(x," ?")}) %>%
  as.data.frame %>%
  select(-contains("?")) %>%
  mutate(workclass = NULL, 
         education = NULL, 
         marital_status = NULL,
         occupation = NULL ,
         race = NULL)   -> adult_test

# change the levels of income to high and low
levels(adult_test$income)
levels(adult_test$income) <- c('lt50k', 'ge50k')

head(adult_test)

# imbalance:
table(adult_test$income)/nrow(adult_test)

# fix the names
names(adult_test) %>%
  stringr::str_replace_all(" ", "_") %>%
  stringr::str_remove_all("-") -> names(adult_test)



########################################
# model: GLM with LASSO regularization #
########################################

# Training Data
X_train <- adult_train %>% select(-income)

X_train <- X_train %>% 
  apply(MARGIN = 2, FUN = as.numeric) %>%
  as.data.frame
y_train <- adult_train$income
table(y_train) # imbalance

# Testing Data
X_test <- adult_test %>% select(-income)

X_test <- X_test %>% 
  apply(MARGIN = 2, FUN = as.numeric) %>%
  as.data.frame
y_test <- adult_test$income
table(y_test)

# Training Settings
ctrl1 <- trainControl(method = "cv",
                      n = 5,
                      classProbs = TRUE,
                      summaryFunction = twoClassSummary)

# Fit Caret Model
set.seed(1234)
model <- caret::train(x = X_train,
                      y = y_train,
                      method = "glmnet",
                      family = "binomial",
                      metric = 'ROC',
                      trControl = ctrl1)

# See the results
model

# Model Accuracy on train set
train_acc <- caret::confusionMatrix(data = predict(model, X_train),
                                    reference = y_train)
train_acc

# Confusion Matrix and Statistics

#              Reference
# Prediction lt50k ge50k
#      lt50k 22871  3573
#     ge50k  1849  4268
#
# Accuracy : 0.8335 
# 
# Sensitivity : 0.9252          
# Specificity : 0.5443    

test_acc <- caret::confusionMatrix(data = predict(model, X_test),
                                    reference = y_test)

test_acc

# Accuracy : 0.8353
# Sensitivity : 0.9248          
# Specificity : 0.5458

########################
# class maps: train set #
########################

# Colors
cols = c('darkgoldenrod', 'blueviolet')
cols <- adjustcolor(cols, alpha = 0.7)

par(mfrow=c(1,1))
#pdf('../pdf/adult-localized-classmap-train-ge50k.pdf', width = 8, height = 8)
# Target class: ge50k (>= 50K)
plotClassMap(model,
             X = X_train,
             y = y_train,
             class = "ge50k",
             k = 10,
             cols = cols)
legend("bottomright",
       legend = c('lt50k', "ge50k"),
       fill = cols[1:2],
       bg = 'white',
       cex = 1)
#dev.off()
#
# Majority class: lt50k (< 50K)
plotClassMap(model,
             X = X_train,
             y = y_train,
             class = "lt50k",
             k = 10,
             cols = cols)
legend("bottomright",
       legend = c('lt50k', "ge50k"),
       fill = cols[1:2],
       bg = 'white',
       cex = 1)

# The classifier performs poorly on the target class, in part due to class
# imbalance. There is indeed bias in the algorithm (and it is LASSO-penalized)
# so that adds to it as well. But, let's see whether there is systematic bias
# towards protected classes.



########################
# class maps: test set #
########################

# Colors
cols = c('darkgoldenrod', 'blueviolet')
cols <- adjustcolor(cols, alpha = 0.7)

par(mfrow=c(1,1))
#pdf('../pdf/adult-localized-classmap-test-ge50k.pdf', width = 8, height = 8)
# Target class: ge50k (>= 50K)
plotClassMap(model,
             X = X_test,
             y = y_test,
             class = "ge50k",
             k = 10,
             cols = cols)
legend("bottomright",
       legend = c('lt50k', "ge50k"),
       fill = cols[1:2],
       bg = 'white',
       cex = 1)
#dev.off()
#
# Majority class: lt50k (< 50K)
plotClassMap(model,
             X = X_test,
             y = y_test,
             class = "lt50k",
             k = 10,
             cols = cols)
legend("bottomright",
       legend = c('lt50k', "ge50k"),
       fill = cols[1:2],
       bg = 'white',
       cex = 1)

# The classifier performs poorly on the target class, in part due to class
# imbalance. There is indeed bias in the algorithm (and it is LASSO-penalized)
# so that adds to it as well. But, let's see whether there is systematic bias
# towards protected classes.


########################
# class maps: gender   #
########################  

#pdf('../pdf/adult-localized-classmaps-gender.pdf', width = 9, height = 5)
par(mfrow=c(1,2))
par(mar = c(3, 4.5, 3, 0.5))
plotClassMap(model,
             X = X_test[X_test$female==0,],
             y = y_test[X_test$female==0],
             class = "ge50k",
             comparison = 'MALE',
             k = 10,
             cols = cols)
legend("bottom",
       legend = c('lt50k', "ge50k"),
       fill = cols[1:2],
       bg = 'white',
       cex = 1)
#
par(mar = c(3, 4.5, 3, 0.5))
plotClassMap(model,
             X = X_test[X_test$female==1,],
             y = y_test[X_test$female==1],
             class = "ge50k",
             comparison = 'FEMALE',
             k = 10,
             cols = cols)
legend("bottom",
       legend = c('lt50k', "ge50k"),
       fill = cols[1:2],
       bg = 'white',
       cex = 1)
#dev.off()

# This is in line with what we see in slide 25 
# the model has a harder time predicting ge50k for females.
# why? it seems that in this case, it's not because of unfair algorithmic 
# design, as there is bias for both genders. It seems that, simply, there
# might not be as many females who have ge50k in the train set.


#############################
# Checking the training set #
#############################

#pdf('../pdf/adult-train-barchart.pdf', width = 8, height = 8)
par(mfrow=c(1,1))
ggplot(data = adult_train,
       aes(fill = income, x = female)) +
  geom_bar(stat = "count") +
  ggtitle('Income by Gender: Training Data') +
  scale_fill_manual(values=cols) +
  theme_classic() 
#dev.off()


#########################
# class maps: Doctorate #
#########################

#pdf('../pdf/adult-localized-classmaps-doctorate.pdf', width = 9, height = 5)
par(mfrow=c(1,2))
par(mar = c(3, 4.5, 3, 0.5))
plotClassMap(model,
             X = X_test[X_test$education_Doctorate==0,],
             y = y_test[X_test$education_Doctorate==0],
             class = "ge50k",
             comparison = 'Without Doctorate',
             main = 'Class Map: ge50k, without Doctorate',
             k = 10,
             cols = cols)
legend("bottomright",
       legend = c('lt50k', "ge50k"),
       fill = cols[1:2],
       bg = 'white',
       cex = 1)
#
par(mar = c(3, 4.5, 3, 0.5))
plotClassMap(model,
             X = X_test[X_test$education_Doctorate==1,],
             y = y_test[X_test$education_Doctorate==1],
             class = "ge50k",
             comparison = 'With Doctorate',
             k = 10,
             cols = cols,
             main = "Class Map: ge50k, Doctorate")

legend("bottomright",
       legend = c('lt50k', "ge50k"),
       fill = cols[1:2],
       bg = 'white',
       cex = 1)
#dev.off()

# Here, it's different. Indeed, the model predicts better for those with
# doctorate degrees. But, the population of Doctorates have a different
# dsn of local farness. They are more similar group!
