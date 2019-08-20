#!/usr/bin/python

import scipy.io as sio
import pandas as pd
import numpy as np
import os
import re
import gc
import helpers as hp
import sys

# HELPERS
def getPatientStruct(mat_file):
    return mat_file['patient']

def getVoie(patient):
    return patient[0, 0]['voie']

# voie will have a shape of (1, n), where n # of info inside range(0,n-1)
def getVoieInfo(voie, n):
    voie_n = voie[0,n]
    if (voie_n.shape[1] == 0): return np.asarray([], dtype=np.float32)
    return voie_n[0, 0]

def getLabels(voie, n):
    voie_n = voie[0,n]
    return (voie_n.dtype).names

def isStartStringMatched(s, m):
    return re.match(m + '.+', s)

def getFeatureInfo(vo, n, f):
    fi = getVoieInfo(vo, n)
    return fi[f] if len(fi) != 0 else np.asarray([], dtype=np.float32)

# -------

# general info to grab for further study
# return info: labels
def getFirstOneInfo(file_path):
    mat = sio.loadmat(file_path)
    ps = getPatientStruct(mat)
    vo = getVoie(ps)
    n = (vo.shape)[1]
    labels = getLabels(vo, 0)
    del ps, vo, n, mat
    gc.collect()
    return labels

# Accumulate one patient info
# acumulate from same file and different structs
def getPatientVars(patient_dir):
    mat = sio.loadmat(patient_dir)
    ps = getPatientStruct(mat)
    vo = getVoie(ps)
    n = (vo.shape)[1]
    del ps, mat
    gc.collect()
    return vo, n

# -------

def addPatientFeatureInfoV2(patient_dir, features, feat_dict):
    for file_path in os.listdir(patient_dir):
        if file_path.endswith(".mat"):
            n = 0
            try:
                vo, n = getPatientVars(patient_dir + '/' + file_path)
            except KeyboardInterrupt:
                raise # get out
            except TypeError as e :
                print("TypeError error:")
                print(e)
            except ValueError as e : 
                print("Value error: ")
                print(e)
            except:
                print("Unexpected error:", sys.exc_info()[0])
            for f, feature in enumerate(features):
                if feature not in feat_dict : feat_dict[feature] = np.asarray([], dtype=np.float32)
                for i in range(n):
                    new = hp.flattenNPList(getFeatureInfo(vo, i, feature))
                    feat_dict[feature] = hp.addTwoNPLists(feat_dict[feature], new) 

def addAllPatientsInfoV2(mdir, features, total = None):
    feat_dict = {}
    cnt = 0
    for dir_path in os.listdir(mdir):
        if isStartStringMatched(dir_path, 'RS'):    
            print(f'Working on {dir_path}')
            addPatientFeatureInfoV2(mdir + dir_path, features, feat_dict)
            #print(feat_dict[features[0]].dtype, feat_dict[features[1]].dtype, feat_dict[features[2]].dtype, feat_dict[features[3]].dtype)
        cnt += 1
        if cnt >= total and total is not None: break
    return feat_dict

def addAllPatientsInfoV3(mdir, features, total = None, start = 0):
    feat_dict = {}
    cnt = 0
    stt = 0
    for dir_path in os.listdir(mdir):
        if isStartStringMatched(dir_path, 'RS') and stt > start:
            print(f'Working on {dir_path}')
            addPatientFeatureInfoV2(mdir + dir_path, features, feat_dict)
            print(feat_dict[features[0]].dtype, feat_dict[features[1]].dtype, feat_dict[features[2]].dtype, feat_dict[features[3]].dtype)
        if stt > start: cnt += 1
        stt += 1
        if cnt >= total and total is not None: break
    return feat_dict

def addPatientFeatureInfo(patient_dir, feature):
    info = []
    for file_path in os.listdir(patient_dir):
        if file_path.endswith(".mat"):
            n = 0
            try:
                vo, n = getPatientVars(patient_dir + '/' + file_path)
            except KeyboardInterrupt:
                raise # get out
            except TypeError as e :
                print("TypeError error:")
                print(e)
            except ValueError as e : 
                print("Value error: ")
                print(e)
            except:
                print("Unexpected error:", sys.exc_info()[0])
            for i in range(n):
                new = hp.flattenNPList(getFeatureInfo(vo, i, feature))
                info = hp.addTwoNPLists(info, new)       
    return info

def addAllPatientsInfo(mdir, feature, total = None):
    info = []
    cnt = 0
    for dir_path in os.listdir(mdir):
        if isStartStringMatched(dir_path, 'RS'):
            print(f'Working on {feature} - {dir_path}')
            new = addPatientFeatureInfo(mdir + dir_path, feature)
            info = hp.addTwoNPLists(info, new)
        cnt += 1
        if cnt >= total and total is not None: break
    return info
