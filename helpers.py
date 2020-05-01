#!/usr/bin/python

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import math
import gc
import os
import re
import shutil
from functools import reduce
import clustering as cl

def distancesFromVToAllMatrixElements(v, m):
    return np.linalg.norm(m - v, axis=-1)

def getNNearestToPoint(v, m, n):
    """Return the n nearest points (distances, indexes in matrix(m))"""
    index = list(range(m.shape[0]))
    distances = distancesFromVToAllMatrixElements(v, m)
    s = zip(*sorted(zip( distances, index ))) # sort according to distances
    distances, index = zip(*sorted(zip( distances, index ))) # sort according to distances
    return np.array(distances[:n]), np.array(index[:n])

def getFromNearestInfo(v, X, qty, indexes, df_info, ordered_titles, X_is_normalized=True, std=None):
    v = np.array(v)
    distances, ix = getNNearestToPoint(v, X, qty)
    return getDfInfo(X, indexes, ix, qty, df_info, ordered_titles, X_is_normalized=X_is_normalized, std=std)

def getDfInfo(X, indexes, sub_indexes, qty, df_info, ordered_titles, X_is_normalized=True, std=None):
    matched_indexes = indexes[ sub_indexes ]
    X_to_use = X if X_is_normalized and not std else cl.getValuesBeforeNormalization(X, std)
    matched_X = X_to_use[ sub_indexes ]
    info = df_info.loc[ matched_indexes ]
    info = info.iloc[ 0 : qty ]
    for i, title in enumerate(ordered_titles):
        info[ title ] = matched_X[:qty, i]
    return info

def getObjOfRepresentativeness(r):
    def repToObj(o, e):
        o[e['group']] = e
        return o
    return reduce(repToObj, r, {})

def getFromClusterInfo(X, predicted, qty, indexes, cluster, df_info, ordered_titles, X_is_normalized=True, std=None):
    """
    Get some rows complete row, including its info
    X is the matrix
    qty is the number of rows wanted to be retrieved
    """
    matches = ( predicted == cluster )
    return getDfInfo(X, indexes, matches, qty, df_info, ordered_titles, X_is_normalized=X_is_normalized, std=std)

def fileAtPathExists(path):
    return os.path.isfile(path)

def copyAndRename(original, new):
    shutil.copyfile(original, new)

def readFileToPandas(file):
    if re.match(r'.+\.feather', file):
        print('reading feather')
        return pd.read_feather(file)
    elif re.match(r'.+\.h5', file):
        print('reading h5')
        return pd.read_hdf(file)
    else:
        raise TypeError('Not supported file type')

def cleanDF(df, to_drop):
    dropped = df[to_drop]
    new = df.drop(columns=to_drop)
    return new, dropped

### Create a Folder
# d -- string of path
def createDir(d):
    if not os.path.exists(d):
        os.makedirs(d)

### Flatten a list depth 1
# l -- list ( maximum flattening "[[]] -> []" )
def flattenList(l):
    return [item for sublist in l for item in sublist]

### Flatten a list of type NumPy
# l -- numpy list
def flattenNPList(l):
    return l.flatten()

# -------

# Transform a matrix like data to a dict like data
# data -- matrix
# labels -- the labels that will contain the different columns of data
def putDataInDict(data, labels):
    if len(labels) != len(data): raise NameError("different sizes in labels and data")
    return { labels[i]: d for i, d in enumerate(data) }

### Transform a dictionary into a dataframe
# d -- dictionary of type { "column_title": [] ... }
def convertDictInDF(d):
    return pd.DataFrame(d)

### Concatenate two Numpy Lists
# info -- existing numpy list
# nez_info -- listo to be added to the existing one
def addTwoNPLists(info, new_info):
    return np.concatenate((info, new_info))

### Scikit-like transformer for a Pandas DataFrame
# (There is nothing in Scikit-Learn to handle Pandas DataFrames)
# To use inside a pipeline on preprocessing
# (BaseEstimator, TransformerMixin) -- from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

### A way to calculate the most meaningful groups
# Model should contain model.means_, model.covariances_ as attributes
# Models that work: GMM, BGM
# X -- the data matrix
# _Y -- the predicted values (using model.predict(X) for example) 
def getRepresentativeness(model, X, _Y):
    r = []
    for i, (mean, cov) in enumerate(zip(model.means_, model.covariances_)):
        r.append({ 'group': i, 'qty': len(X[_Y == i, 0]), 'representativeness': len(X[_Y == i, 0])/len(X) })
    tot = { '>50': 0, '>30': 0, '>15': 0, '>05': 0, '<05': 0 }
    for i, e in enumerate(r):
        if e['representativeness'] >= 0.5: tot['>50'] += 1
        elif e['representativeness'] >= 0.3: tot['>30'] += 1
        elif e['representativeness'] >= 0.15: tot['>15'] += 1
        elif e['representativeness'] >= 0.05: tot['>05'] += 1
        else : tot['<05'] += 1
    print(f'totals: {tot}')
    return r

### A way to calculate the most meaningful groups
# Model should contain model.cluster_centers_
# Models that work: KMeans, MeanShift ( clustering methods )
# X -- the data matrix
# _Y -- the predicted values (using model.predict(X) for example)
def getRepresentativenessKM(km, X, _Y):
    r = []
    for i, (center) in enumerate(zip(km.cluster_centers_)):
        r.append({ 'group': i, 'qty': len(X[_Y == i, 0]), 'representativeness': len(X[_Y == i, 0])/len(X) })
    tot = { '>50': 0, '>30': 0, '>15': 0, '>05': 0, '<05': 0 }
    for i, e in enumerate(r):
        if e['representativeness'] >= 0.5: tot['>50'] += 1
        elif e['representativeness'] >= 0.3: tot['>30'] += 1
        elif e['representativeness'] >= 0.15: tot['>15'] += 1
        elif e['representativeness'] >= 0.05: tot['>05'] += 1
        else : tot['<05'] += 1
    print(tot)
    return r

### Get Rows from DataFrame
# df -- data frame
# part -- part of the total percentage (maximum value = 1/percent, minimum value = 1)
# percent -- percentage of the divided total ( value between 1 and 0 )
# example: part = 2, percent=0.25 -> will return the second quarter of the information 
def getRows(df, part, percent):
    total = df.shape[0]
    indexes = np.arange(0, total, step=1, dtype=np.int32)
    qty = math.floor( total * percent )
    start = qty * (part - 1)
    end = qty * part if qty * part < total else total - 1
    grabbed_indexes = indexes[ start:end ]
    newdf = df.iloc[ grabbed_indexes, : ]
    del qty, indexes, total; gc.collect();
    return newdf, grabbed_indexes

### Get Rows Randomly from DataFrame
# df -- data frame
# percent -- percentage of the data that wants to be returned ( value between 1 and 0 )
def getRandomRows(df, percent):
    total = df.shape[0]
    indexes = np.arange(0, total, step=1, dtype=np.int32)
    np.random.shuffle(indexes)
    qty = math.floor( total * percent )
    grabbed_indexes, ignored_indexes = indexes[0:qty], indexes[qty:total]
    newdf, ignored_df = df.iloc[grabbed_indexes, :], df.iloc[ignored_indexes, :]
    del qty, indexes, total; gc.collect()
    return newdf, grabbed_indexes, ignored_df, ignored_indexes

