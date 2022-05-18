# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:23:01 2022

@author: Alexander


6 Hypothesentest

6.2 Zugversuche an Bond-Verbindungen

"""



""" Import libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy as sp
import scipy.io
import scipy.stats

import math as m
import statistics

import glob
import os, sys




'''Reading and rearranging the data from csv-file'''

'''normalerweise benutzt man pd.read_csv'''
#data = pd.read_csv('../Data/Glasfaser.csv', header=None)
#X = np.ravel(data.values)

'''hier benutze ich loatdmat'''
mat = scipy.io.loadmat('../Data/Zugversuch.mat')


data = mat['zugfestigkeit']
X = np.ravel(data)


''' rel. Histogramm einer Stichprobe '''
#max_range = ( np.floor(X.min()), np.ceil(X.max()) )     # must be tuple
#max_range = (310,350)

ax = plt.figure(figsize=(6, 4)).subplots(1, 1)
ax.hist(X, bins=10, range=None, density=True, weights=None, cumulative=False, histtype='bar', rwidth=0.8)
ax.grid(True, which= 'both', axis='both', linestyle='--')

#ax.set_ylim(top=0.1)
#ax.set_xlim(right=350, left=310)



