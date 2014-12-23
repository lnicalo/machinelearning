__author__ = 'Luis Fernando'

import numpy as np
from numpy import seterr
import sys


def mutualInformation(x, y):
    """ Mutual information - Parzen method """

    classes = np.unique(y)

    # Priori probability
    p = np.mean(np.array([y == i for i in classes]), axis=1)

    # Check priori probability
    if len(p) < 2:
        print >> sys.stderr, "Number of classes must be at least 2"
        return -1

    mi = np.empty(x.shape[1], dtype=float)
    for j in xrange(0, x.shape[1]):
        x_j = x[:, j]
        # P(x|y)
        p_x_y = np.array([parzen(x_j, x_j[y == i]) for i in classes])

        # P(y|x): Bayes' rule
        den = np.dot(p, p_x_y)
        p_c_fij = np.array([p[i] * p_x_y[i, ] / den for i in xrange(0, p_x_y.shape[0])])

        # Entropy: H(y|x)
        seterr(divide='ignore')
        a = np.where(p_c_fij > 0, np.log2(p_c_fij), 0)
        seterr(divide='warn')
        a = p_c_fij * a
        H_c_x = -np.nansum(a) / x.shape[0]

        # Entropy: H(y)
        H_w = -np.dot(p, np.log2(p))
        mi[j, ] = H_w - H_c_x

    return mi


def parzen(x0, x):
    """ Probability density function estimation (univariate) """
    M = x.shape[0]

    # h: Window width
    sigma = np.var(x)
    h = (4 / (3 * M)) ** (1 / 5) * sigma

    # Parzen window
    dist2 = (x0[..., np.newaxis] - x[np.newaxis, ...]) ** 2
    phi = 1 / (np.sqrt(2 * np.pi) * h * M) * np.exp(-dist2 / (2 * (h ** 2)))
    return np.sum(phi, 1)


if __name__ == '__main__':
    from matplotlib import pyplot

    # Mutual information variation with mean distance
    numFeats = 50
    distance = np.linspace(-0.8, 0.8, numFeats)
    mi = []
    N = 500

    ## Experiments
    # Labels
    y = np.concatenate((np.ones((N,)), 2 * np.ones((N,))))

    # Features
    x = np.empty([N*2, numFeats], dtype=float)
    for i in xrange(0, numFeats):
        mean1 = 0
        var1 = 0.1
        mean2 = mean1 + distance[i]
        var2 = var1

        x[0:N, i] = mean1 + var1 * np.random.randn(N)
        x[N:2*N+1, i] = mean2 + var2 * np.random.randn(N)

    # Compute mutual information
    mi = mutualInformation(x, y)

    pyplot.figure(1)
    pyplot.plot(distance, mi)
    pyplot.xlabel('Distance')
    pyplot.ylabel('Mutual information')
    pyplot.show()
