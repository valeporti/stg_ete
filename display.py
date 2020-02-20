#!/usr/bin/python

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import itertools

def plotOneGMMClusterAxe(ax, X, predicted, cluster, col1, col2, color, tts):
    ax.scatter(X[predicted == cluster, col1], X[predicted == cluster, col2], 0.8, color)
    ax.scatter(X[predicted != cluster, col1], X[predicted != cluster, col2], 0.8, 'navy')
    ax.set_xlabel(tts[col1])
    ax.set_ylabel(tts[col2])

def plotAll2DGMMs(X, predicted, cluster, color, tts):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    plotOneGMMClusterAxe(ax[0, 0], X, predicted, cluster, 0, 1, color, tts )
    plotOneGMMClusterAxe(ax[0, 1], X, predicted, cluster, 0, 2, color, tts )
    plotOneGMMClusterAxe(ax[0, 2], X, predicted, cluster, 0, 3, color, tts )
    plotOneGMMClusterAxe(ax[1, 0], X, predicted, cluster, 1, 2, color, tts )
    plotOneGMMClusterAxe(ax[1, 1], X, predicted, cluster, 1, 3, color, tts )
    plotOneGMMClusterAxe(ax[1, 2], X, predicted, cluster, 2, 3, color, tts )
    plt.show(); fig.clf(); plt.close();

def plotOneGMMCluster(X, predicted, cluster, col1, col2, color, tts):
    plt.scatter(X[predicted == cluster, col1], X[predicted == cluster, col2], 0.8, color)
    plt.scatter(X[predicted != cluster, col1], X[predicted != cluster, col2], 0.8, 'navy')
    plt.xlabel(tts[col1])
    plt.ylabel(tts[col2])

def plotOneGMMCluster3D(ax, X, predicted, cluster, col1, col2, col3, color, tts):
    ax.scatter(xs=X[predicted != cluster, col1], ys=X[predicted != cluster, col2], zs=X[predicted != cluster, col3], s=0.8, c='navy')
    ax.scatter(xs=X[predicted == cluster, col1], ys=X[predicted == cluster, col2], zs=X[predicted == cluster, col3], s=0.8, c=color)
    ax.set_xlabel(tts[col1])
    ax.set_ylabel(tts[col2])
    ax.set_zlabel(tts[col3])

def printPCAScatter(df, titles):
    if len(titles) == 3:
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        printScatterV2(ax[0], df[ titles[0] ], df[ titles[1] ], titles[0], titles[1] )
        printScatterV2(ax[1], df[ titles[0] ], df[ titles[2] ], titles[0], titles[2] )
        printScatterV2(ax[2], df[ titles[1] ], df[ titles[2] ], titles[1], titles[2] )
    elif len(titles) == 4:
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        printScatterV2(ax[0, 0], df[ titles[0] ], df[ titles[1] ], titles[0], titles[1] )
        printScatterV2(ax[0, 1], df[ titles[0] ], df[ titles[2] ], titles[0], titles[2] )
        printScatterV2(ax[0, 2], df[ titles[0] ], df[ titles[3] ], titles[0], titles[3] )
        printScatterV2(ax[1, 0], df[ titles[1] ], df[ titles[2] ], titles[1], titles[2] )
        printScatterV2(ax[1, 1], df[ titles[1] ], df[ titles[3] ], titles[1], titles[3] )
        printScatterV2(ax[1, 2], df[ titles[2] ], df[ titles[3] ], titles[2], titles[3] )
    plt.show(); fig.clf(); plt.close();


def printScatterV2(ax, x, y, titx, tity):
    ax.scatter(x, y, s=0.8, c='midnightblue')
    ax.set_xlabel(titx)
    ax.set_ylabel(tity)

def boxPlotTheFour(df):
    fig, ax = plt.subplots(1, 4, figsize=(20, 6))
    df.boxplot(column='vectorCorrKLD', ax=ax[0])
    df.boxplot(column='vectorFAmpKLD', ax=ax[1])
    df.boxplot(column='vectorUFAmpKLD', ax=ax[2])
    df.boxplot(column='vectorRRKLD', ax=ax[3])
    plt.show(); fig.clf(); plt.close();

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

def printPCAGMM(gmm, X, titles, color_iter):
    if len(titles) == 3:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        getPlotGMM(gmm, X, color_iter, 0, 1, ax[0], titles)
        getPlotGMM(gmm, X, color_iter, 0, 2, ax[1], titles)
        getPlotGMM(gmm, X, color_iter, 1, 2, ax[2], titles)
    elif len(titles) == 4:
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
        getPlotGMM(gmm, X, color_iter, 0, 1, ax[0, 0], titles)
        getPlotGMM(gmm, X, color_iter, 0, 2, ax[0, 1], titles)
        getPlotGMM(gmm, X, color_iter, 0, 3, ax[0, 2], titles)
        getPlotGMM(gmm, X, color_iter, 1, 2, ax[1, 0], titles)
        getPlotGMM(gmm, X, color_iter, 1, 3, ax[1, 1], titles)
        getPlotGMM(gmm, X, color_iter, 2, 3, ax[1, 2], titles)
    plt.show(); fig.clf(); plt.close();

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
    plt.clf()
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

def plotMShFromFig(ax, col1, col2, X, ms, n_clusters_, cluster_centers, labels, titles):
    colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        ax.plot(X[my_members, col1], X[my_members, col2], col + '.', markersize=1)
        ax.plot(cluster_center[col1], cluster_center[col2], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)
    ax.set_xlabel(titles[col1])
    ax.set_ylabel(titles[col2])

def printMultiDimensionMSh(X, ms, n_clusters_, cluster_centers, labels, titles):
    if len(titles) == 3:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        plotMShFromFig(ax[0], 0, 1, X, ms, n_clusters_, cluster_centers, labels, titles)
        plotMShFromFig(ax[1], 0, 2, X, ms, n_clusters_, cluster_centers, labels, titles)
        plotMShFromFig(ax[2], 1, 2, X, ms, n_clusters_, cluster_centers, labels, titles)
    elif len(titles) == 4:
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
        plotMShFromFig(ax[0, 0], 0, 1, X, ms, n_clusters_, cluster_centers, labels, titles)
        plotMShFromFig(ax[0, 1], 0, 2, X, ms, n_clusters_, cluster_centers, labels, titles)
        plotMShFromFig(ax[0, 2], 0, 3, X, ms, n_clusters_, cluster_centers, labels, titles)
        plotMShFromFig(ax[1, 0], 1, 2, X, ms, n_clusters_, cluster_centers, labels, titles)
        plotMShFromFig(ax[1, 1], 1, 3, X, ms, n_clusters_, cluster_centers, labels, titles)
        plotMShFromFig(ax[1, 2], 2, 3, X, ms, n_clusters_, cluster_centers, labels, titles)
    plt.show(); fig.clf(); plt.close();

def plotMSh(X, ms, n_clusters_, cluster_centers, labels):
    colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.', markersize=1)
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)
    #plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    plt.clf()
    plt.close()

def plotBICScores(bic, cv_types, color_iter, n_components_range):
    bars = []
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    #xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    #    .2 * np.floor(bic.argmin() / len(n_components_range))
    #plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    plt.show()
    plt.clf()
    plt.close()

# Baysian mixture
def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(means[n], eig_vals[0], eig_vals[1],
                                  180 + angle, edgecolor='black')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor('#56B4E9')
        ax.add_artist(ell)


def plot_results(ax1, ax2, estimator, X, y, title, plot_title=False):
    ax1.set_title(title)
    ax1.scatter(X[:, 0], X[:, 1], s=5, marker='o', color=colors[y], alpha=0.8)
    ax1.set_xlim(-2., 2.)
    ax1.set_ylim(-3., 3.)
    ax1.set_xticks(())
    ax1.set_yticks(())
    plot_ellipses(ax1, estimator.weights_, estimator.means_,
                  estimator.covariances_)

    ax2.get_xaxis().set_tick_params(direction='out')
    ax2.yaxis.grid(True, alpha=0.7)
    for k, w in enumerate(estimator.weights_):
        ax2.bar(k, w, width=0.9, color='#56B4E9', zorder=3,
                align='center', edgecolor='black')
        ax2.text(k, w + 0.007, "%.1f%%" % (w * 100.),
                 horizontalalignment='center')
    ax2.set_xlim(-.6, 2 * n_components - .4)
    ax2.set_ylim(0., 1.1)
    ax2.tick_params(axis='y', which='both', left=False,
                    right=False, labelleft=False)
    ax2.tick_params(axis='x', which='both', top=False)

    if plot_title:
        ax1.set_ylabel('Estimated Mixtures')
        ax2.set_ylabel('Weight of each component')
