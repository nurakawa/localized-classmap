# source 
exec(open('localized_classmap.py').read())

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# Create a classifier: Linear Discriminant Analysis
clf = LinearDiscriminantAnalysis()

# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False, random_state=1234)

# train the classifier
clf.fit(X_train, y_train)

PAC = compPAC(model=clf, X=X_test, y=y_test)
LF = compLocalFarness(X=X_test, y=y_test, k=20, metric='euclidean')

plotExplanations(model=clf, X=X_test, y=y_test, k=10, cl=8, annotate=True)

predicted = clf.predict(X=X_test)
misclassified_1 = np.array([6, 54])
misclassified_2 = np.array([116, 134])

# Visualize the misclassified points
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(3, 2))
for ax, image, prediction in zip(axes, X_test[misclassified_1],
                                 predicted[misclassified_1]):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title(f'Prediction: {prediction}')

plt.show()

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(3, 2))
for ax, image, prediction in zip(axes, X_test[misclassified_2],
                                 predicted[misclassified_2]):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title(f'Prediction: {prediction}')

plt.show()
