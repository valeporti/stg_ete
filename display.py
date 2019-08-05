#!/usr/bin/python

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
