#!/usr/bin/python

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import scipy.stats as st
import gc
import math
import helpers as hp

### From a transformed matrix, inverse the process to get the initial equivalent
# X - is the transformed matrix
# std - the scikit learn standardScaler class
# pca - the scikit learn PCA class
def getInitialForm(X, std, pca = None):
    XI = pca.inverse_transform(X) if pca is not None else X[:]
    return std.inverse_transform(XI)

### Impute missing values in the DataFrame using a SimpleImputer strategy
def cleanData(df_ALL, strategy):
    X = df_ALL.to_numpy()
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imp.fit(X)
    X = imp.transform(X)
    df = hp.convertDictInDF(hp.putDataInDict(
            [X[:,0], X[:,1], X[:,2], X[:,3]],
            ['vectorRRKLD', 'vectorFAmpKLD', 'vectorUFAmpKLD', 'vectorCorrKLD']
        ))
    del X; gc.collect();
    return df;

### Get the PCA 
# n_components - number of components as the total percentage of the desired variance or as an integer
# X - the matrix
def getPCA(n_components, X):
    pca = PCA(n_components)
    pca.fit(X)  
    print(f'variance ratio: {pca.explained_variance_ratio_}') 
    #print(pca.singular_values_)
    #print(pca.components_)
    return pca

### Transforming the matrix to the desired data type and returning the DataFrame of the passed matrix
def getXandDf(pca, X, columns):
    # issue transforming to float64 on transform https://github.com/scikit-learn/scikit-learn/issues/11000
    Xpca = pca.transform(X).astype(np.float32) # maintain float32
    dfPca = pd.DataFrame(data=Xpca, columns=columns)
    del pca; gc.collect();
    return Xpca, dfPca

### First approach to remove the outliers by manually imposing the maximum values for the problematic features (vectorUFAmpKLD, vectorRRKLD)
# df - the data frame to treat
# threshold - threshold to apply to the zscore obtained for vectorCorrKLD && vectorFAmpKLD in order to remove its outliers
# UFAMP_limit - manual threshold "to the right only" (not absolute value) to remove the values bigger than that value for vectorUFAmpKLD
# RRKLD_limit - manual threshold "to the right only" (not absolute value) to remove the values bigger than that value for vectorRRKLD
def removeOutliers(df, threshold = 2, UFAMP_limit = 2e+5, RRKLD_limit = 0.5e+5):
    initial_shape = df.shape
    df_2 = df.copy()
    zCorrKLD = np.abs(st.zscore(df['vectorCorrKLD']))
    zFAmpKLD = np.abs(st.zscore(df['vectorFAmpKLD']))
    toMaintain_CorrKLD = zCorrKLD <= threshold
    toMaintain_FAmpKLD = zFAmpKLD <= threshold
    toMaintain = np.logical_and(toMaintain_CorrKLD, toMaintain_FAmpKLD) # boolean array
    df_2 = df_2[toMaintain]
    df_3 = df_2.copy()
    df_3 = df_2[(df_2['vectorUFAmpKLD'] < UFAMP_limit)]
    df_3 = df_3[(df_3['vectorRRKLD'] < RRKLD_limit)]
    df_nout = df_3.copy()
    print(f'after soft removal (vectorCorrKLD && vectorFAmpKLD) shape : {df_2.shape} && { str( round(100 * df_2.shape[0]/initial_shape[0], 2) ) }')
    print(f'after hard removal (vectorUFAmpKLD && vectorRRKLD) shape : {df_3.shape} && { str( round(100 * df_3.shape[0]/initial_shape[0], 2) ) }')
    del toMaintain_CorrKLD, toMaintain_FAmpKLD, toMaintain, zCorrKLD, zFAmpKLD, df_2, df_3
    gc.collect()
    return df_nout

### Simulate importance of a MeanShift center by usng its bandwidth as perimeter
def getImportance(col, center, bandwidth):
    return np.where(np.logical_and(col>=center - bandwidth, col<=center + bandwidth))[0].shape[0]

### First version to remove outliers using meanshift (Second proposal after the first "manual" approach: removeOutliers
# cols - pass the columns for which ths approach will be done (normally [vectorUFAmpKLD, vectorRRKLD])
# X - Matrix
# threshold - threshold over which "important" values will outstand, and thus, not be removed
# cluster_centers - obtained from MeanShift calculation
# bandwidth - estimated to do the MeanShift calculation
def removeOuliersFromMeanShift(cols, X, threshold, cluster_centers, bandwidth):
    total = X.shape[0]
    indexes = []
    for col in cols:
        centers = cluster_centers[:, col]
        centers.sort()
        importance = [ getImportance(X[:, col], c, bandwidth) for c in centers ]
        #z = np.abs(st.zscore(importance))
        z = np.array([ v / total for i, v in enumerate(importance) ])
        print('to maintain', np.where(z > threshold))
        to_remove_centers = centers[(z <= threshold)]
        extremes = [ to_remove_centers[0] - bandwidth, to_remove_centers[-1] + bandwidth ]
        print('extremes', extremes) 
        removing_indexes = np.where(np.logical_and(X[:, col]>=extremes[0], X[:, col]<=extremes[1]))[0]
        print('to remove', removing_indexes)
        indexes = np.concatenate((indexes, removing_indexes))
    to_remove = np.unique(indexes)
    return np.delete(X, to_remove, 0)

### grabs the value in an array just before the value passed 
# main - ordered array 
# n - the value from which we want to obtain the before value
def getBeforeCenter(main, n):
    t = (main < n)
    i = [ j for j, b in enumerate(t) if b ]
    v = main[t]
    return v[-1]

### Remove Outliers by usng the Meanshift approach Version2, due to a desire to get back the remaining indexes besides the remaining data
# cols - pass the columns for which ths approach will be done (normally [vectorUFAmpKLD, vectorRRKLD])
# X - Matrix
# threshold - threshold over which "important" values will outstand, and thus, not be removed
# cluster_centers - obtained from MeanShift calculation
# bandwidth - estimated to do the MeanShift calculation
def removeOuliersFromMeanShiftV2(cols, X, idx, threshold, cluster_centers, bandwidth):
    print(f'shapes: { idx.shape }, {X.shape}')    
    total = X.shape[0]
    indexes = []
    for col in cols:
        centers = cluster_centers[:, col]
        centers.sort()
        importance = [ getImportance(X[:, col], c, bandwidth) for c in centers ]
        z = np.array([ v / total for i, v in enumerate(importance) ])
        to_remove_centers = centers[(z <= threshold)] # they are always the last centers
        before_first_center = getBeforeCenter(centers, to_remove_centers[0])
        to_remove_centers = np.insert(to_remove_centers, 0, before_first_center) 
        extremes = [ to_remove_centers[0], to_remove_centers[-1] + bandwidth ]
        removing_indexes = np.where(np.logical_and(extremes[0]<=X[:, col], X[:, col]<=extremes[1]))[0]
        indexes = np.concatenate((indexes, removing_indexes))
    to_remove = np.unique(indexes).astype(np.int32)
    return np.delete(X, to_remove, 0), np.delete(idx, to_remove, 0)


### RemoveOutliers Verision 2, motivated in order to add a more "generic" way of removing the oultiers
def removeOutliersV2(df, indexes, threshold = 2, threshold_hard = 0.01, cols_hard = [0, 2], samples_bandwidth = 50000):
    initial_shape = df.shape
    X = df.to_numpy().astype(np.float64)
    bandwidth = estimate_bandwidth(X, n_samples=samples_bandwidth, quantile=0.5)
    print(f'bandwidth: { bandwidth }')
    ms, n_clusters_, cluster_centers, labels = meanClustering(X, bandwidth)
    X, indexes = removeOuliersFromMeanShiftV2(cols_hard, X, indexes, threshold_hard, cluster_centers, bandwidth)
    print(f'shapes: { indexes.shape }, {X.shape}')
    df_h = hp.convertDictInDF(hp.putDataInDict(
        [X[:,0], X[:,1], X[:,2], X[:,3]],
        ['vectorRRKLD', 'vectorFAmpKLD', 'vectorUFAmpKLD', 'vectorCorrKLD']))
    df_2 = df_h.copy()
    zCorrKLD = np.abs(st.zscore(df_h['vectorCorrKLD']))
    zFAmpKLD = np.abs(st.zscore(df_h['vectorFAmpKLD']))
    toMaintain_CorrKLD = zCorrKLD <= threshold
    toMaintain_FAmpKLD = zFAmpKLD <= threshold
    toMaintain = np.logical_and(toMaintain_CorrKLD, toMaintain_FAmpKLD)
    print('tomain',toMaintain)
    indexes = indexes[toMaintain]
    df_2 = df_2[toMaintain]
    df_nout = df_2.copy()
    print(f'after hard removal (vectorUFAmpKLD && vectorRRKLD) shape : {df_2.shape} && { str( round(100 * df_2.shape[0]/initial_shape[0], 2) ) }')
    print(f'after soft removal (vectorCorrKLD && vectorFAmpKLD) shape : {df_h.shape} && { str( round(100 * df_h.shape[0]/initial_shape[0], 2) ) }')
    del toMaintain_CorrKLD, toMaintain_FAmpKLD, toMaintain, zCorrKLD, zFAmpKLD, df_2, df_h, X, ms, n_clusters_, cluster_centers, labels
    gc.collect()
    return df_nout, indexes

### RemoveOuliers V3, since one of the two "hard to remove" features is not well removed with the "more generic" way, a hard removal is taken into account
def removeOutliersV3(df, indexes, threshold = 2, threshold_hard = 0.01, cols_hard = [0], samples_bandwidth = 50000, UFAMP_limit = 2e+5):
    initial_shape = df.shape
    X = df.to_numpy().astype(np.float64)
    bandwidth = estimate_bandwidth(X, n_samples=samples_bandwidth, quantile=0.5)
    print(f'bandwidth: { bandwidth }')
    ms, n_clusters_, cluster_centers, labels = meanClustering(X, bandwidth)
    X, indexes = removeOuliersFromMeanShiftV2(cols_hard, X, indexes, threshold_hard, cluster_centers, bandwidth)
    print(f'shapes: { indexes.shape }, {X.shape}')
    df_h = hp.convertDictInDF(hp.putDataInDict(
        [X[:,0], X[:,1], X[:,2], X[:,3]],
        ['vectorRRKLD', 'vectorFAmpKLD', 'vectorUFAmpKLD', 'vectorCorrKLD']))
    df_2 = df_h.copy()
    zCorrKLD = np.abs(st.zscore(df_2['vectorCorrKLD']))
    zFAmpKLD = np.abs(st.zscore(df_2['vectorFAmpKLD']))
    toMaintain_CorrKLD = zCorrKLD <= threshold
    toMaintain_FAmpKLD = zFAmpKLD <= threshold
    toMaintain = np.logical_and(toMaintain_CorrKLD, toMaintain_FAmpKLD)
    print('tomain',toMaintain)
    indexes = indexes[toMaintain]
    df_2 = df_2[toMaintain]
    df_2 = df_2[(df_2['vectorUFAmpKLD'] < UFAMP_limit)]  #not removable other way 
    df_nout = df_2.copy()
    print(f'after hard removal (vectorUFAmpKLD && vectorRRKLD) shape : {df_2.shape} && { str( round(100 * df_2.shape[0]/initial_shape[0], 2) ) }')
    print(f'after soft removal (vectorCorrKLD && vectorFAmpKLD) shape : {df_h.shape} && { str( round(100 * df_h.shape[0]/initial_shape[0], 2) ) }')
    del toMaintain_CorrKLD, toMaintain_FAmpKLD, toMaintain, zCorrKLD, zFAmpKLD, df_2, df_h, X, ms, n_clusters_, cluster_centers, labels
    gc.collect()
    return df_nout

def normalize(df):
    num_attribs = list(df)
    pipeline = Pipeline([
            ('selector', hp.DataFrameSelector(num_attribs)), # to handle pandas data frame
            ('std_scaler', StandardScaler(copy=True))
        ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("main", pipeline),
        ])
    del pipeline, num_attribs
    gc.collect()
    return full_pipeline.fit_transform(df), full_pipeline

def normalizeV2(df):
    X = df.to_numpy()
    std = StandardScaler(copy=True)
    X = std.fit(X).transform(X)
    return X, std

def runOutNormPCA(df_ALL, threshold = 20, UFAMP_limit = 1e+6, RRKLD_limit = 2e+6):
    df_nout = removeOutliers(df_ALL, threshold, UFAMP_limit, RRKLD_limit)
    Xnorm = normalize(df_nout)
    pca = getPCA(0.95, Xnorm)
    titPca = [ 'pc' + str(i) for i, v in enumerate(pca.explained_variance_ratio_) ]
    Xpca, dfPca = getXandDf(pca, Xnorm, titPca)
    return df_nout, Xnorm, Xpca, dfPca, titPca

def runOutNormPCAV2(df_ALL, indexes, threshold = 20, threshold_hard = 0.01, cols_hard = [0, 2], samples_bandwidth = 50000):
    df_nout, indexes = removeOutliersV2(df_ALL, indexes, threshold, threshold_hard, cols_hard, samples_bandwidth)
    Xnorm, std = normalizeV2(df_nout)
    pca = getPCA(0.95, Xnorm)
    titPca = [ 'pc' + str(i) for i, v in enumerate(pca.explained_variance_ratio_) ]
    Xpca, dfPca = getXandDf(pca, Xnorm, titPca)
    return df_nout, Xnorm, Xpca, dfPca, titPca, pca, std, indexes

def runOutNormV2(df_ALL, indexes, threshold = 20, threshold_hard = 0.01, cols_hard = [0, 2], samples_bandwidth = 50000, v3 = True, UFAMP_limit=2e+5):
    df_nout = None
    if v3 is True:
        df_nout = removeOutliersV3(df_ALL, indexes, UFAMP_limit=UFAMP_limit)
    else:
        df_nout, indexes = removeOutliersV2(df_ALL, indexes, threshold, threshold_hard, cols_hard, samples_bandwidth)
    Xnorm, std = normalizeV2(df_nout)
    return df_nout, Xnorm, std, indexes

def runOutNorm(df_ALL, indexes, threshold = 20, threshold_hard = 0.01, cols_hard = [0, 2], samples_bandwidth = 50000, v2 = True):
    df_nout = None
    if v2 is True:
        df_nout, indexes = removeOutliersV2(df_ALL, indexes, threshold, threshold_hard, cols_hard, samples_bandwidth)
    else:
        df_nout = removeOutliers(df_ALL, threshold)
    Xnorm, std = normalizeV2(df_nout)
    return df_nout, Xnorm, std, indexes

"""def runClusering(df_ALL, indexes, withpca=False, threshold = 20, threshold_hard = 0.01, cols_hard = [0, 2], samples_bandwidth = 50000, UFAMP_limit = 1e+6, RRKLD_limit = 2e+6):
    if withpca is True:
        df_nout, indexes = removeOutliersV2(df_ALL, indexes, threshold, threshold_hard, cols_hard, samples_bandwidth)
"""
# ---------------------------

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

def getBestGMMUsingBIC(X, n_components_range, cv_types=['spherical', 'tied', 'diag', 'full'], reg_covar=1e-6):
    lowest_bic = np.infty
    bic = []
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type, reg_covar=reg_covar).fit(X)
            print(f'gmm weights: { gmm.weights_ }')
            bicval = gmm.bic(X)
            print(f'bic value: { bicval }')
            bic.append(bicval)
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm, bic, cv_types

def getBestGMMUsingAIC(X, n_components_range, cv_types=['spherical', 'tied', 'diag', 'full'], reg_covar=1e-6):
    lowest_aic = np.infty
    aic = []
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type, reg_covar=reg_covar).fit(X)
            print(f'gmm weights: { gmm.weights_ }')
            aicval = gmm.aic(X)
            print(f'bic value: { aicval }')
            aic.append(aicval)
            if aic[-1] < lowest_aic:
                lowest_aic = aic[-1]
                best_gmm = gmm
    return best_gmm, aic, cv_types
