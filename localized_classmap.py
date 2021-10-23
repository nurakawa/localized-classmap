import numpy as np
from sklearn.neighbors import KDTree

def compPAC(model, X, y):

    n = X.shape[0]
    PAC = np.array([0.0]*n)
    nlab = len(np.unique(y)) # number of classes

    # get fitted model probabilities
    model_probs = model.predict_proba(X)

    # case: two classes
    if nlab == 2:
        altint = 1 - yint # yint will take values 0 or 1
        for i in range(n):
            PAC[i] = model_probs[i, altint[i]]

    # case: more than two classes
    ptrue = np.array([0.0]*n)
    palt = np.array([0.0]*n)
    for i in range(n):
        ptrue[i] = model_probs[i][y[i]] # prob of the true class
        others = list(range(nlab))
        others.pop(y[i]) # indices of the other classes
        palt[i] = np.max(model_probs[i][others]) # most likely alternative class

        PAC[i] = (palt[i]) / (palt[i] + ptrue[i])

    return PAC


def compLocalFarness(model, X, y, k, metric='euclidean'):

    nlab = len(np.unique(y)) # number of classes

    # find nearest neighbors
    kdt = KDTree(X, metric=metric)
    dist, ind = kdt.query(X, k=k)

    # array of epsilon_i
    epislon_arr = [dist[i][(k-1)] for i in range(len(dist))]

    # epanechnikov kernel weighting function
    ep_kernel = lambda x: (3/4)*(1 - (x*x))*(int(abs(x) <= 1))
    kernel_wt = lambda ep, d: (1/ep) * ep_kernel(x=(d/ep))

    # compute localized farness
    n = X.shape[0]
    local_farness = np.array([0.0]*n)
    class_probs = [[0.0] * nlab] * n

    for i in range(n):
        local_dists = dist[i]
        wts = [kernel_wt(ep=epislon_arr[i], d=local_dists[ii]) for ii in range(len(local_dists))]
        wts = wts / sum(wts)
        # wts should sum to 1

        for cl in range(nlab):

            n_classmates = sum(y[ind[i]]==cl)

            if n_classmates == 0:
                class_probs[i][cl] = 0
            elif n_classmates >= k:
                class_probs[i][cl] = 1
            else:
                class_probs[i][cl] = sum(wts[y[ind[i]]==cl])

    print(class_probs)

    for i in range(n):
        cl = y[i]
        others = list(range(nlab))
        others.pop(cl)
        a = 0
        for j in others:
            a += class_probs[i][j]
        local_farness[i] = a - class_probs[i][cl]

    # rescale
    local_farness = (local_farness + 1) / 2
    return local_farness

def plotExplanations(model, X, y, k):
    pass

