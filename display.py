#!/usr/bin/python

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import itertools

def draw_correlation_matrix(df):
    sns.set(style="white")
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(5, 5))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True) #color_palette("GnBu_d")
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, square=True, 
                linewidths=.5, cbar_kws={"shrink": .5})

def printScatter(x, y):
    plt.scatter(x, y, s=0.8, c='midnightblue')
    plt.show()
    plt.clf()
    plt.close()

def print3dScatter(x, y, z):
    fig = plt.figure()	
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=0.8, c='midnightblue')

def getPlotGMM(gmm, X, color_iter, col1, col2, ax, tts, show_gc=True):
    Y_ = gmm.predict(X)
    for i, (mean, cov, color) in enumerate(zip(gmm.means_, gmm.covariances_,
                                           color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        ax.scatter(X[Y_ == i, col1], X[Y_ == i, col2], .8, color=color)
        ax.set_xlabel(tts[col1])
        ax.set_ylabel(tts[col2])

        if not show_gc: continue
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(.5)
        ax.add_artist(ell)

def printThreeKMaeans(X, Y_, titles):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].scatter(X[:, 0], X[:, 1], c=Y_, s=0.8, cmap='Set1')
    ax[0].set_xlabel(titles[0])
    ax[0].set_ylabel(titles[1])
    ax[1].scatter(X[:, 1], X[:, 2], c=Y_, s=0.8, cmap='Set1')
    ax[1].set_xlabel(titles[1])
    ax[1].set_ylabel(titles[2])
    ax[2].scatter(X[:, 0], X[:, 2], c=Y_, s=0.8, cmap='Set1')
    ax[2].set_xlabel(titles[0])
    ax[2].set_ylabel(titles[2])
    plt.show()
    plt.close()

def plotGM(X, Y, means, covariances, ax, col1, col2, color_iter):
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        ax.scatter(X[Y == i, col1], X[Y == i, col2], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

def plotMSh(X, ms, n_clusters_, cluster_centers, labels):
    colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.', markersize=1)
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=10)
    #plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    plt.clf()
    plt.close()
