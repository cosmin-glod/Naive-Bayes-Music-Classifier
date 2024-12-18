import os

import numpy as np
from sklearn.decomposition import PCA

def center(data):
    means = np.mean(data, axis = 0)
    centeredData = data - means
    std = np.std(centeredData, axis = 0)
    final = np.divide(centeredData, std)
    return final

def PCA(variance = 0.95):

    data = np.load("features.npy")
    means = center(data)

    cov = np.cov(means, rowvar = False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    ordered = np.argsort(eigenvalues)[::-1]

    largestValues = eigenvalues[ordered]
    largestVectors = eigenvectors[:, ordered]

    optimalSigma = np.cumsum(largestValues) / np.sum(largestValues)
    sigma = np.where(optimalSigma >= variance)[0][0] + 1 # number of principal components

    T = largestVectors[:, :sigma]
    projection = np.dot(means, T)

    np.save("pca.npy", projection)


PCA()
