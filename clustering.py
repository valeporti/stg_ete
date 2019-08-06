#!/usr/bin/python

from sklearn.cluster import KMeans, MeanShift
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
