#!/usr/bin/python

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def createDir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
def flattenList(l):
    return [item for sublist in l for item in sublist]

def flattenNPList(l):
    return l.flatten()

# -------

def putDataInDict(data, labels):
    if len(labels) != len(data): raise NameError("different sizes in labels and data")
    return { labels[i]: d for i, d in enumerate(data) }

def convertDictInDF(d):
    return pd.DataFrame(d)

def addTwoNPLists(info, new_info):
    return np.concatenate((info, new_info))

# There is nothing in Scikit-Learn to handle Pandas DataFrames
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

def getRepresentativeness(model, X, _Y):
    r = []
    for i, (mean, cov) in enumerate(zip(model.means_, model.covariances_)):
        r.append({ 'group': i, 'qty': len(X[_Y == i, 0]), 'representativeness': len(X[_Y == i, 0])/len(X) })
    tot = { '>50': 0, '>30': 0, '>15': 0, '<15': 0 }
    for i, e in enumerate(r):
        if e['representativeness'] >= 0.5: tot['>50'] += 1
        elif e['representativeness'] >= 0.3: tot['>30'] += 1
        elif e['representativeness'] >= 0.15: tot['>15'] += 1
        else : tot['<15'] += 1
    print(tot)
    return r

def getRepresentativenessKM(km, X, _Y):
    r = []
    for i, (center) in enumerate(zip(km.cluster_centers_)):
        r.append({ 'group': i, 'qty': len(X[_Y == i, 0]), 'representativeness': len(X[_Y == i, 0])/len(X) })
    tot = { '>50': 0, '>30': 0, '>15': 0, '<15': 0 }
    for i, e in enumerate(r):
        if e['representativeness'] >= 0.5: tot['>50'] += 1
        elif e['representativeness'] >= 0.3: tot['>30'] += 1
        elif e['representativeness'] >= 0.15: tot['>15'] += 1
        else : tot['<15'] += 1
    print(tot)
    return r