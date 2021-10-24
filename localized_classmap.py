import numpy as np
from sklearn.neighbors import KDTree

def compPAC(model, X, y):
    """
    :param model: sklearn model fitted to training data.
                  model must have "probability=True" when initialized.
    :param X:     dataset for prediction, usually a held-out test set
    :param y:     labels corresponding to X
    :return:      Probability of Alternative Classification (PAC) from the trained classifier
    """

    # parameters
    n = X.shape[0] # number of data points in X
    PAC = np.array([0.0]*n) # initialize PAC array
    nlab = len(np.unique(y)) # number of classes

    # get fitted model probabilities
    model_probs = model.predict_proba(X)

    # case: two classes
    if nlab == 2:
        altint = 1 - yint # yint will take values 0 or 1
        for i in range(n):
            PAC[i] = model_probs[i, altint[i]]

    # case: more than two classes
    ptrue = np.array([0.0]*n) # array containing probability an item belongs to its true class
    palt = np.array([0.0]*n) # array containing probability an item belongs to an alternative class
    for i in range(n):
        ptrue[i] = model_probs[i][y[i]] # prob of the true class
        others = list(range(nlab))
        others.pop(y[i]) # indices of the other classes
        palt[i] = np.max(model_probs[i][others]) # most likely alternative class
        PAC[i] = (palt[i]) / (palt[i] + ptrue[i]) # PAC: conditional prob of alternative class

    return PAC


def compLocalFarness(X, y, k, metric='euclidean'):
    """
    :param X:       dataset for prediction, should be the same as what was used for PAC
    :param y:       corresponding labels of X
    :param k:       number of nearest neighbors to consider for localized farness computation
    :param metric:  distance metric for nearest neighbor search.
    :return:        localized farness computed from the data, independent of classifier
    """

    # find nearest neighbors with KD Tree
    kdt = KDTree(X, metric=metric)
    dist, ind = kdt.query(X, k=k) # get the nearest neighbor distances and indices

    # array of epsilon_i (widths of epanechnikov kernels)
    epislon_arr = [dist[i][(k-1)] for i in range(len(dist))]

    # epanechnikov kernel weighting function
    ep_kernel = lambda x: (3/4)*(1 - (x*x))*(int(abs(x) <= 1))
    kernel_wt = lambda ep, d: (1/ep) * ep_kernel(x=(d/ep))

    # compute localized farness
    n = X.shape[0] # number of rows in the data
    local_farness = np.array([0.0]*n) # initialize local farness

    for i in range(n):
        local_dists = dist[i] # distances from point i to its neighbors
        wts = [kernel_wt(ep=epislon_arr[i], d=local_dists[ii]) for ii in range(len(local_dists))]
        wts = wts / sum(wts) # weight the local distances. wts should sum to 1.
        class_prob = sum(wts[y[ind[i]] == y[i]]) # Pr(i \in g_i)
        local_farness[i] = 1.0 - class_prob # LF(i) = 1 - Pr(i \in g_i)

    # round to 4 decimal places, for simplicity
    # NOTE: np.abs() is used because sometimes we get -0.0. LF is always positive
    local_farness = np.abs(np.round(local_farness, 4))
    return local_farness

def plotExplanations(model, X, y, k):
    pass

