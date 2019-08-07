#!/usr/bin/python

from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
import numpy as np

def getKmeanskClusters(k, X):
    km = KMeans(n_clusters=k, random_state=1).fit(X)
    Y_ = km.predict(X)
    return Y_, km

def meanClustering(X, bw):
    ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)
    return ms, n_clusters_, cluster_centers, labels

def getBestGMMUsingBIC(X, n_components_range):
    lowest_bic = np.infty
    bic = []
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                    covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm, bic, cv_types
