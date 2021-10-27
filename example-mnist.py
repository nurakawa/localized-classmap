# =============================================================================
# title:    example_mnist_full.py
# author:   Nura Kawa
# summary:  Classification of MNIST dataset with MLP classifier.
# #         Explanations of classifier presented with localized classmap.
# =============================================================================

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

# source localized classmap
exec(open('localized_classmap.py').read())

# load and preprocess images
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

# load data
data_path = ""
train_data = np.loadtxt(data_path + "mnist_train.csv",delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv",delimiter=",")

train_imgs = np.asfarray(train_data[:, 1:])
test_imgs = np.asfarray(test_data[:, 1:])

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

# reshape the labels
train_labels = train_labels.reshape(-1)
test_labels = test_labels.reshape(-1)

# make labels int
train_labels = train_labels.astype(int)
test_labels = test_labels.astype(int)

# flatten the images
train_data = train_imgs.reshape((len(train_imgs), -1))
test_data = test_imgs.reshape(len(test_imgs), -1)

# Create a classifier: MLP
clf = MLPClassifier(random_state=42, max_iter=150)

# train the classifier
clf.fit(train_data, train_labels)

# make predictions
predicted = clf.predict(X=test_data)

# evaluate the accuracy
disp = metrics.ConfusionMatrixDisplay.from_predictions(test_labels, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
plt.close()

acc = metrics.accuracy_score(y_true=test_labels, y_pred=predicted)
print(acc)

# plot explanations
# Since this is quite slow, let's do it for class 8

# class 3
plotExplanations(model=clf,
                 X=test_imgs,
                 y=test_labels,
                 k=40,
                 cl=3,
                 annotate=True)

# class 8
plotExplanations(model=clf,
                 X=test_imgs,
                 y=test_labels,
                 k=40,
                 cl=8,
                 annotate=True)
# Note: to get the indices of some interesting points,
# set annotate = True.
# However for now the annotations are not all clear - will fix this

# images to visualize
high_PAC = [4112, 4737, 5495, 3833]
high_LF = [542, 6555, 3757, 4639]

#for i in high_PAC:
#    img = test_imgs[i].reshape((28,28))
#    plt.imshow(img, cmap="Greys")
#    plt.show()

#for i in high_LF:
#    img = test_imgs[i].reshape((28,28))
#    plt.imshow(img, cmap="Greys")
#    plt.show()


num_row = 2
num_col = 2

# plot images
fig, axes = plt.subplots(num_row, num_col,
                         figsize=(2.2*num_col,2*num_row))
for i in range(len(high_PAC)):
    ax = axes[i//num_col, i%num_col]
    img = test_imgs[high_PAC[i]].reshape((28,28))
    ax.imshow(img, cmap='Greys')
    ax.set_title('Predicted: {}'.format(predicted[high_PAC[i]]))
    #ax.set_title('Example: {}'.format(high_PAC[i]))
fig.suptitle('Test Set Examples with high PAC')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(num_row, num_col,
                         figsize=(2.2*num_col,2*num_row))
for i in range(len(high_LF)):
    ax = axes[i//num_col, i%num_col]
    img = test_imgs[high_LF[i]].reshape((28,28))
    ax.imshow(img, cmap='Greys')
    ax.set_title('Predicted: {}'.format(predicted[high_LF[i]]))
    #ax.set_title('Example: {}'.format(high_LF[i]))
fig.suptitle('Test Set Examples with high localized farness')
plt.tight_layout()
plt.show()
