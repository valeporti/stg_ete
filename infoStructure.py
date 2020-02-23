#!/usr/bin/python

import scipy.io as sio
import pandas as pd
import numpy as np
import os
import re
import gc
import helpers as hp
import sys
import random

### Get Patient Data inside MatFile CHU-like data (HDF5)
# In the data passed, the structure in a mat file needed to access "patient" as a first step
def getPatientStruct(mat_file):
    return mat_file['patient']

### Get Voie Data inside Patient CHU-like data (HDF5)
# Data should be reached accessing the "voie" in first row/col in patient matrix
def getVoie(patient):
    return patient[0, 0]['voie']

### Get info inside "Voie"
# voie will have a shape of (1, n), where n # of info inside range(0,n-1)
# voie 
# n -- number of voie
def getVoieInfo(voie, n):
    voie_n = voie[0,n]
    if (voie_n.shape[1] == 0): return np.asarray([], dtype=np.float32) # data come in float32 format, maintain it
    return voie_n[0, 0]

### Get Labels of Data
# In order to organize data, and ask for the correct data to the structure, we need the labels
# voie 
# n - number indicating the voie number
def getLabels(voie, n):
    voie_n = voie[0,n]
    return (voie_n.dtype).names

### Check if "s" string is right after "m" string
# Used to avoid non desired folders
#
def isStartStringMatched(s, m):
    return re.match(m + '.+', s)

### Get data corresponding to the voie ("vo"), number ("n") passed, and feature/label ("f")
# vo -- voie
# n -- number of voie
# f -- feature/label name  (string)
def getFeatureInfo(vo, n, f):
    fi = getVoieInfo(vo, n)
    return fi[f] if len(fi) != 0 else np.asarray([], dtype=np.float32) # data come in float32 format, maintain it

# -------

### General info to grab for further study
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

### Accumulate one patient info
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
# For each .mat file found, obtain the features
def addPatientFeatureInfoV2(mdir, patient_dir, features, feat_dict):
    for file_path in os.listdir(mdir + patient_dir):
        if file_path.endswith(".mat"):
            n = 0; temp_size = 0; size = 0; got_one = False
            try:
                vo, n = getPatientVars(mdir + patient_dir + '/' + file_path)
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
            # collect the features
            for f, feature in enumerate(features):
                if feature not in feat_dict : feat_dict[feature] = np.asarray([], dtype=np.float32)
                if 'voie_num' not in feat_dict : feat_dict['voie_num'] = np.asarray([], dtype=np.uint8)
                for i in range(n):
                    new = hp.flattenNPList(getFeatureInfo(vo, i, feature))
                    feat_dict[feature] = hp.addTwoNPLists(feat_dict[feature], new)
                    temp_size += len(new)
                    # add general voie, to add info of where to find the data
                    if not got_one: 
                        voie_num = np.full(new.shape, fill_value=i, dtype=np.uint8)
                        feat_dict['voie_num'] = hp.addTwoNPLists(feat_dict['voie_num'], voie_num)
                if not got_one: 
                    got_one = True
                    size += temp_size
                    temp_size = 0
            # add file path 
            if 'paths' not in feat_dict : feat_dict['paths'] = np.asarray([])
            paths = np.full((size,), fill_value=f'{ patient_dir }/{ file_path }', dtype='>U64')
            feat_dict['paths'] = hp.addTwoNPLists(feat_dict['paths'], paths) 

def addAllPatientsInfoV2(mdir, features, total = None):
    feat_dict = {}
    cnt = 0
    for dir_path in os.listdir(mdir):
        if isStartStringMatched(dir_path, 'RS'):    
            print(f'Working on {dir_path}')
            addPatientFeatureInfoV2(mdir, dir_path, features, feat_dict)
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
            addPatientFeatureInfoV2(mdir, dir_path, features, feat_dict)
            print(feat_dict[features[0]].dtype, feat_dict[features[1]].dtype, feat_dict[features[2]].dtype, feat_dict[features[3]].dtype)
        if stt > start: cnt += 1
        stt += 1
        if cnt >= total and total is not None: break
    return feat_dict

def saveToFeather(data, columns, name):
    df_ALL = hp.convertDictInDF(data)
    df_ALL.to_feather(name)
    del df_ALL; gc.collect()

def saveToHDF(data, columns, name):
    df_ALL = hp.convertDictInDF(data)
    df_ALL.to_hdf(name, key='df', mode='a', append=True, format='table', data_columns=True)
    del df_ALL; gc.collect()

def addAllPatientsInfoV4(mdir, features, total = [10], dir_to_save = None, to_hdf = False):
    """
    add all patients info to a variable
    (directory, features = titles in files, total is Array with desired quantities to save, dir_to_save is the directory to be saved if None, it won't be saved) 
    """
    feat_dict = {}; counter = 0
    directories = [ v for i, v in enumerate(os.listdir(mdir)) if isStartStringMatched(v, 'RS') ]
    suffled_d = [i for i in range(len(directories))]
    random.shuffle(suffled_d)
    for t, sub_total in enumerate(total): 
        copied = False        
        for i in range(counter, sub_total):
            if len(directories) <= i: break
            print(f'Working on { directories[ suffled_d[i] ] } #{ i }')
            addPatientFeatureInfoV2(mdir, directories[ suffled_d[i] ], features, feat_dict)
            #print(feat_dict[features[0]].dtype, feat_dict[features[1]].dtype, feat_dict[features[2]].dtype, feat_dict[features[3]].dtype)
            if dir_to_save and to_hdf: 
                this_tot = total[ t-1 ] if t > 0 else sub_total
                this_file = f'{ dir_to_save }{ str(this_tot) }_f32.h5'
                new_file = f'{ dir_to_save }{ str(sub_total) }_f32.h5'
                if hp.fileAtPathExists(this_file) and t == 0: os.remove(this_file)
                if hp.fileAtPathExists(this_file) and t > 0 and not copied:
                    hp.copyAndRename(this_file, new_file) # if file existant, it will remove it
                    copied = True
                saveToHDF(feat_dict, features + ['voie_num', 'paths'], f'{ dir_to_save }{ str(sub_total) }_f32.h5')
                feat_dict = {}
                #raise NotImplementedError('not supported method yet (HDF5)')
        if dir_to_save and not to_hdf: saveToFeather(feat_dict, features + ['voie_num', 'paths'], f'{ dir_to_save }{ str(sub_total) }_f32.feather')
        counter = sub_total
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
