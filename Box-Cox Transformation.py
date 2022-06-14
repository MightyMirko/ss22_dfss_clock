# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:56:40 2022

@author: Alexander
"""

#Bibliotheken
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

from scipy.io import wavfile
#from scipy import signal as sig
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.signal import windows
from scipy.signal import medfilt
from scipy.stats import norm, t, chi2, f, weibull_min, gamma
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder



DATA_FILE_TRAIN = 'gesamtdaten_energien.csv'
df = pd.read_csv(DATA_FILE_TRAIN, header=0)
energie = df.iloc[:, 1]

# Datenübertragen
X = np.array(energie.drop(energie[energie>8].index))
print(X)



h_abs1, x1 = np.histogram(X, bins=50)
# plot histogramm
ax = plt.figure(figsize=(6, 4)).subplots(1, 1)
ax.hist(X, x1 , histtype='bar', weights=np.ones(X.shape)/X.shape, rwidth=0.9, cumulative=False)
ax.grid(True, which= 'both', axis='both', linestyle='--')
ax.set_xlabel('Energie')
ax.set_ylabel('Häufigkeit h(E)')
#ax.set_ylim(top=0.25)
#ax.set_title('Kommulative Häufigkeitsverteilung')
plt.tight_layout()








#%%

fig = plt.figure()
ax1 = fig.add_subplot(211)
x = stats.loggamma.rvs(5, size=500) + 5
prob = stats.probplot(x, dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')

ax2 = fig.add_subplot(212)
xt, _ = stats.boxcox(x)
prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation')
plt.show()







''' -------------------------------------------------------------- '''




h_abs1, x1 = np.histogram(x, bins=50)
h_abs2, x2 = np.histogram(xt, bins=50)

X = x
ax = plt.figure(figsize=(6, 4)).subplots(1, 1)
ax.hist(X, x1 , histtype='bar', weights=np.ones(X.shape)/X.shape, rwidth=0.9, cumulative=False)
ax.grid(True, which= 'both', axis='both', linestyle='--')
ax.set_xlabel('Energie')
ax.set_ylabel('Häufigkeit h(E)')
#ax.set_ylim(top=0.25)
#ax.set_title('Kommulative Häufigkeitsverteilung')
plt.tight_layout()

X = xt
ax = plt.figure(figsize=(6, 4)).subplots(1, 1)
ax.hist(X, x2 , histtype='bar', weights=np.ones(X.shape)/X.shape, rwidth=0.9, cumulative=False)
ax.grid(True, which= 'both', axis='both', linestyle='--')
ax.set_xlabel('Energie')
ax.set_ylabel('Häufigkeit h(E)')
#ax.set_ylim(top=0.25)
#ax.set_title('Kommulative Häufigkeitsverteilung')
plt.tight_layout()