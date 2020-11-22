import numpy as np
import random
from sklearn.metrics import log_loss
import copy
# return the indices of selected features
def featureSelection(Mbar, X, y, D=3, r=[0.8, 0.7, 0.6]):
    # previous ensemble model: Mbar
    # training data: (X, y)  X=(N,F), y is an ndarray(N,1)
    # number of bins: D
    # sample ratio for each bin: r
    N,F = X.shape
    y_predict0 = Mbar.predict_proba(X) # model class' structure is assume to be of the same form as our homework
    L = log_loss(y, y_predict0) # assume there is a loss function, L=(N,1) ndarray
    g = np.zeros(F)
    # loop through each feature
    for f in range(F):
        Xf = copy.deepcopy(X)
        # shuffle the value of the f-th column in X
        np.random.shuffle(Xf[:,f])
        y_predict = Mbar.predict_proba(Xf)
        Lf = log_loss(y, y_predict)
        g[f] = np.absolute(L-Lf)
        #g[f] = np.absolute(np.mean(np.subtract(L, Lf)) / np.std(np.subtract(L, Lf)))
    if np.linalg.norm(x=g) < 1E-10:
        return [i for i in range(F)]
    sorted_idx = np.argsort(g)[::-1]
    print(g)
    print(sorted_idx)
    bin_size = round(F/D)
    feature = []
    for d in range(D-1):
        pool = sorted_idx[d*bin_size:(d+1)*bin_size]
        print(pool)
        choice = np.random.choice(pool, round(r[d]*len(pool)),replace=False)
        print(choice)
        feature.extend(choice)
    pool = sorted_idx[(D-1)*bin_size:F]
    print(pool)
    choice = np.random.choice(pool, round(r[D-1]*len(pool)),replace=False)
    print(choice)
    feature.extend(choice)
    # sort the indices in the selected features
    feature.sort()
    print(feature)
    return feature
