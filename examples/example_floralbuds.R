# ==============================================================================
# title:      example_floralbuds.R
#
# summary:    Using the floral buds data from the VCR paper, I show a use case
#             of a localized class map. The challenge of the floral buds data
#             is not the imbalance of "branch" and "support" class, but rather
#             the structure of these classes, shown in the t-sne embeddings 
#             plot. A localized class map gives the user an idea of where an 
#             algorithm's bias might occur, and why some classes are harder to 
#             identify than others.
#
# ==============================================================================


#######
# Setup
#######

rm(list = ls())
#dev.off()

source("plotClassMap.R")
require(classmap)
require(grDevices)
require(Rtsne)



load("../data/Floralbuds_data.rdata")
dim(X) # 550  6
length(y) # 550
round(table(y)/length(y),2)
#  branch     bud  scales support 
#  0.09    0.66    0.17    0.08 
# imbalanced


# Pairs plot
cols = c("saddlebrown","orange","olivedrab4","royalblue3")
cols <- adjustcolor(cols, alpha = 0.7)
pairs(X, col = cols[y]) # hard to separate visually

# labels
labels = c("branch","bud","scale","support")

# stacked barchart (from the VCR paper)
# Quadratic discriminant analysis:
set.seed(1234)
vcr.obj = vcr.da.train(X,y)

# Stackplot in paper:
#pdf("../pdf/floralbuds-stackedplot.pdf", width=5,height=4.3)
labs = c("branch","bud","scale","support")
stackedplot(vcr.obj, classCols = cols, separSize=0.6,
            minSize=1.5, classLabels=labs ,showOutliers=F,
            htitle = "given class", vtitle = "predicted class")
#dev.off()

# example of class map for scales
#pdf("../pdf/floralbuds-classmap-scales.pdf", width=7,height=7)
par(mar = c(3.3, 3.2, 2.7, 1.0))
classmap(vcr.obj, 3, classCols = cols,
         main = "classmap: scales",
         identify = F, cex = 1.2)
legend("topleft", fill = cols, legend = labels,
       cex = 0.75, ncol = 1, bg = "white")
#dev.off()



##############
# t-sne plot #
##############
set.seed(1234)
tsne_view <- Rtsne(X, dims = 2, perplexity = 30, pca = FALSE)
par(mfrow=c(1,1))
par(pty = 's')
plot(tsne_view$Y,
     cex = 1.2,
     pch = 19,
     col = cols[y],
     main = 't-SNE Embeddings of Floral Buds Data',
     xlab = '', ylab = '')
legend('bottomleft',
       fill = cols[1:4],
       legend = labels, cex = 0.9,
       ncol = 1, bg = "white")

# This plot reveals the structure of the data.
# First, we see an imbalance: the majority of the data is bud, and the second-
# largest group is scale. Support and branch are few. Furthermore, the two 
# largest groups form a cloud, making them relatively easy to identify.
# the smaller groups are dispersed, particularly the supports. 
# Focusing on supports, we can identify some data difficulty factors.
# This is a good case for a localized class map.

##############
# class maps #
##############

# Quadratic discriminant analysis:
set.seed(1234)
vcr.obj = vcr.da.train(X,y)

confmat = confmat.vcr(vcr.obj, showOutliers = F)
#
# Confusion matrix:
#                   predicted
# observed  branch bud scales support
#   branch      46   0      0       3
#   bud          0 355      2       6
#   scales       2   0     90       2
#   support      6   1      0      37
# 
# The accuracy is 96%.


#######################
# localized class map #
#######################

# step 1: create model in caret
model <- caret::train(x = X, y = y, method = "qda")

# step 2: localized class map
# directly input the model, and it computes PAC and LF
par(mfrow=c(1,1))
plotClassMap(model, X, y,
             class = 'support', 
             cols = cols, 
             k = 10)

# check if the model predictions match that of the vcr object
model_preds <- predict(model, X)
mean(model_preds == vcr.obj$pred) # same!

# Localized Class Maps: All Classes
# using the caret model
#
#pdf(file="../pdf/floralbuds-localized-classmap-all.pdf",
    width=8,height=8)
par(mfrow=c(2,2))
par(mar = c(4.3, 3.2, 2.7, 1.0))
# class map: branch
plotClassMap(model = model,
             X = X,
             y = y,
             class = "branch",
             cols = cols,
             k = 5)
#
# class map: bud
par(mar = c(4.3, 3.2, 2.7, 1.0))
plotClassMap(model = model,
             X = X,
             y = y,
             class = "bud",
             cols = cols,
             k = 5)
#
# class map: scales
par(mar = c(4.3, 3.2, 2.7, 1.0))
plotClassMap(model = model,
             X = X,
             y = y,
             class = "scales",
             cols = cols,
             k = 5)
#
# class map: support
par(mar = c(4.3, 3.2, 2.7, 1.0))
plotClassMap(model = model,
             X = X,
             y = y,
             class = "support",
             cols = cols,
             k = 5)
#dev.off()
# we immediately see that support is the class where performance
# is relatively poor. The rest are doing well. Let's take a closer 
# look at support

#######################
# comparison: support #
#######################
#pdf('../pdf/floralbuds-classmap-comparison-support.pdf', width = 8, height = 8)
par(mfrow=c(1,2))
#
# original class map
par(mar = c(3.3, 3.2, 2.7, 1.0))
classmap(vcr.obj, 4, classCols = cols,
         main = "classmap: supports",
         identify = F, cex = 1.2)
legend("topright", fill = cols, legend = labels,
       cex = 0.75, ncol = 1, bg = "white")
#identify(qfunc(vcr.obj$farness), vcr.obj$PAC)
# [1]   8 221 299 502
pt_locations <- data.frame('farness' = qfunc(vcr.obj$farness),
                           'pac' = vcr.obj$PAC)
# xvals = pt_locations[c(8,221,299,502),][,1] + 0.15
# yvals = pt_locations[c(8,221,299,502),][,2]
points(pt_locations[c(8,221,299,502),][,1],
       pt_locations[c(8,221,299,502),][,2],
       pch = 3, cex = 1.3)
labs  = c(  "a",   "b",  "c",   "d")
xvals = c( 1.7537916, 1.1165383, 1.6454840, 0.8857177)+0.15
yvals = c( 0.9960313, 0.8067124, 0.6455838, 0.5680612)
text(x = xvals, y = yvals, labels = labs, cex = 1)

# 
# localized class map
par(mar = c(3.3, 3.2, 2.7, 1.0))
plotClassMap(model, X, y,
             class = 'support', 
             cols = cols, 
             k = 10,
             main = 'localized classmap: supports')
legend("bottomright", fill = cols, legend = labels,
       cex = 0.75, ncol = 1, bg = "white")
pt_locations <- data.frame('farness' = qfunc(compLocalFarness(X,y,k=10)),
                           'pac' = compPAC(model, X, y))
#xvals = pt_locations[c(8,221,299,502),][,1]
#yvals = pt_locations[c(8,221,299,502),][,2]
points(pt_locations[c(8,221,299,502),][,1],
       pt_locations[c(8,221,299,502),][,2],
       pch = 3, cex = 1.3)
labs  = c(  "a",   "b",  "c",   "d")
xvals = c( 4.00000, 1.56660, 2.53195, 1.30591)-0.15
yvals = c( 0.9960313, 0.8067124, 0.6455838, 0.5680612)
text(x = xvals, y = yvals, labels = labs, cex = 1)
 #dev.off()

######################
# comparison: bud    #
######################
#pdf('../pdf/floralbuds-classmap-comparison-bud.pdf', width = 8, height = 8)
par(mfrow=c(1,2))
#
# original class map
par(mar = c(3.3, 3.2, 2.7, 1.0))
classmap(vcr.obj, 2, classCols = cols,
         cex = 1.4,
         main = "classmap: bud")
legend("right", fill = cols, legend = labels,
       cex = 0.7, ncol = 1, bg = "white")
#identify(qfunc(vcr.obj$farness), vcr.obj$PAC)
# bud: [1] 105 117 238 268 466
pt_locations <- data.frame('farness' = qfunc(vcr.obj$farness),
                           'pac' = vcr.obj$PAC)
points(pt_locations[c(105,117,238,268,466),1],
       pt_locations[c(105,117,238,268,466),2],
       pch = 3, cex = 1.3, col = 'black')
labs  = c(  "e",   "f",  "g",   "h", 'i')
xvals = pt_locations[c(105,117,238,268,466),1] - 0.13
yvals = pt_locations[c(105,117,238,268,466),2]
xvals[1] <- xvals[1] + 0.05 # point e
yvals[1] <- yvals[1] + 0.05 # point e

text(x = xvals, y = yvals, labels = labs, cex = 1)
# 
# localized class map
# localized class map
par(mar = c(3.3, 3.2, 2.7, 1.0))
plotClassMap(model, X, y,
             class = 'bud', 
             cols = cols, 
             k = 10,
             main = 'localized classmap: supports')
legend("bottomright", fill = cols, legend = labels,
       cex = 0.75, ncol = 1, bg = "white")
pt_locations <- data.frame('farness' = qfunc(compLocalFarness(X,y,k=10)),
                           'pac' = compPAC(model, X, y))
labs  = c(  "e",   "f",  "g",   "h", 'i')
points(pt_locations[c(105,117,238,268,466),1],
       pt_locations[c(105,117,238,268,466),2],
       pch = 3, cex = 1.3, col = 'black')
xvals = pt_locations[c(105,117,238,268,466),1] - 0.13
yvals = pt_locations[c(105,117,238,268,466),2]
xvals[1] <- xvals[1] + 0.06
yvals[1] <- yvals[1] + 0.04

#xvals = c( 1.7537916, 1.1165383, 1.6454840, 0.8857177)+0.15
#yvals = c(0.9670309, 0.7859255, 0.1755200, 0.5363596)
text(x = xvals, y = yvals, labels = labs, cex = 1)
 #dev.off()
#



##############
# t-SNE plot #
##############
#pdf('../pdf/floralbuds-tsne-plot-labeled.pdf', width = 8, height = 8)
labs = c('a','b','c','d','e',
         'f','g','h','i')
xvals = tsne_view$Y[c(8,221,299,502,105,
                      117,238,268,466),1] 
yvals = tsne_view$Y[c(8,221,299,502,105,
                       117,238,268,466),2] 

par(mfrow=c(1,1))
par(pty = 's')
plot(tsne_view$Y,
     cex = 1.4,
     pch = 19,
     col = cols[y],
     main = 't-SNE Embeddings of Floral Buds Data',
     xlab = '', ylab = '')
legend('bottomleft',
       fill = cols[1:4],
       legend = labels, cex = 0.9,
       ncol = 1, bg = "white")
points(xvals,yvals,pch = 3, cex = 1.4, lwd = 1.3)
text(x = xvals+0.6, y = yvals+1, labels = labs, cex = 1.2)
#dev.off()

###############
# Choice of k # 
###############

#pdf('../pdf/floralbuds-choice-of-k.pdf', width = 4, height = 7)


par(mfcol=c(3,2))
par(mar = c(4.3, 3.2, 2.7, 1.0))
# k = 5
plotClassMap(model = model,
             X = X,
             y = y,
             main = c('localized class map: ',
                      'supports, k = 3'),
             class = "support",
             cols = cols,
              
             k = 3)
#
# k = 10
plotClassMap(model = model,
             X = X,
             y = y,
             main = c('localized class map: ',
                      'supports, k = 10'),
             class = "support",
             cols = cols,
             k = 10)
# k = 30
plotClassMap(model = model,
             X = X,
             y = y,
             main = c('localized class map: ',
                      'supports, k = 40'),
             class = "support",
             comparison = "k = 40",
             cols = cols,
              
             k = 40)


par(mar = c(4.3, 3.2, 2.7, 1.0))
# k = 5
plotClassMap(model = model,
             X = X,
             y = y,
             main = c('localized class map: ',
                      'bud, k = 3'),
             class = "bud",
             comparison = "k = 3",
             cols = cols,
             k = 3)
#
# k = 10
plotClassMap(model = model,
             X = X,
             y = y,
             main = c('localized class map: ',
                      'bud, k = 10'),
             class = "bud",
             comparison = "k = 10",
             cols = cols,
              
             k = 10)
# k = 30
plotClassMap(model = model,
             X = X,
             y = y,
             main = c('localized class map: ',
                      'bud, k = 40'),
             class = "bud",
             comparison = "k = 40",
             cols = cols,
             k = 40)
 #dev.off()
