# Localized Class Map

This repository contains the implementation of the localized class map, a tool
for visualizing classification results. It also includes some examples using benchmark datasets, found in the folder "examples."

The localized class map is an extension of the class map of [Raymaekers, Rousseeuw and Hubert (2021)][1], which is available on CRAN as R 
package `classmap`. The class map is a visualization tool for local explanations of classification algorithms. It explains individual predictions of a classifier by plotting the classifier's view of the predictions. The localized  class map modifies one of the axes of the class map, resulting in a local explanation method that is model-agnostic; it can be used for almost any classifier.

The implementation of the localized class map is made to be compatible with `caret`. A user first trains a model in `caret`, then inputs it directly into `plotClassMap()` to visualize the results. See `examples/example-iris.R` for a simple example.  

## Example: Classification of the Iris Dataset

In the example `examples/example-iris.R`, we use a Support Vector Machine with 
a linear kernel to classify the iris dataset. The iris dataset is a popular benchmark dataset that contains measurements describing three types of iris flowers: virginica, setosa and versicolor. The classification goal is to identify the species of teh flower. After training a classifier, we can get information about its performance, such as the accuracy (here it was 96%) and a confusion matrix, which
shows where the classifier has success and challenges. 

However, we do not have an idea as to _why_ the classifier has difficulty in 
some cases. Is it because of issues with the data? Or, is it because the classifier is not well-suited for the data?

The class map of [Raymaekers, Rousseeuw and Hubert (2021)][1], implemented in R package `classmap` gives insight into the behavior of the classifier by visualizing the data _from the perspective of the classifier_. The class map shows a view of the probabilities assigned to each object by the classifier, and of how far an object lies from its class. 

The __localized class map__ adapts the class map to consider both the perspective of the classifier and the data, separately. We keep the y-axis of the class map, which shows a classifier's idea of where an object lies in its class. On the x-axis, we display _localized farness_, which uses local qualities of the data to assess where an object lies in the data space. This shows a picture of how the classifier performs with respect to the structure of the data.

To get an idea of the structure of the iris dataset, let's look at a t-SNE embeddings plot:

![](img/iris-tsne.png?raw=true)

We notice that the class setosa is well-separated from the other two classes. Classes virginica and versicolor have a bit of overlap. 

We use a Support Vector Machine with a linear kernel to classify the data, obtaining a 96% overall accuracy. Let's use localized class maps to visualize the results:

![](img/iris-localized-class-maps.png?raw=true)


The plots display the localized class map of points whose given labels are versicolor, setosa and virginica. The points are colored by the prediction of the classifier. 

In the first localized class map, all points are colored red, which corresponds to the ground-truth class versicolor, with one misclassified point colored orange, the color that corresponds to class virginica. In the second localized class map, all points are colored blue, which corresponds to the correct class, setosa. In the final map, all points but one are colored orange.

In all the localized class maps, the y-axis is __Probability of Alternative Classification__, or PAC. It gives the conditional probability that an object belongs to a different class, from the perspective of the classifier. The classifier sees the majority of data points belong in their class, although a few points in class versicolor and virginica have a high PAC. The x-axis is __localized farness__, the probability that an item belongs to another class, in light of the structure of the data. __(Please note: details on how to compute these values will be added).__

The localized class maps show a story that corresponds with what we see in the t-SNE embeddings plot. The setosa flowers are a distinct, well-separated group. Structurally, all points lie close to their class. The versicolor and virginica classes have a larger range of localized farness, as some of these points lie on the boundary between classes.

The localized class maps reveal that the classifier has difficulty discerning points that are at the boundary between classes versicolor and virginica. The misclassified versicolor point lies quite far from its class, making it generally harder to classify correctly. The misclassified virginica point has a localized farness of 0.5, meaning that it lies in between two classes; this also indicates challenge for a classifier.
Overall, the classifier seems well-suited to the data, since it only misclassifies challening points.


[1]: https://arxiv.org/abs/2007.14495
