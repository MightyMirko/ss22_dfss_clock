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
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Daten einlesen
DATA_FILE_TRAIN = 'gesamtdaten_energien.csv'
df = pd.read_csv(DATA_FILE_TRAIN, header=0)
# Datenübertragen
energie = df.iloc[:, 1]

''' Wichtig !!! 
    Erkenntnis:             Prognosewert
    0,3 < energie < 5,5 => 3.046427043731887
    0,3 < energie < 8   => 3.1210158442267004
    0,2 < energie < 20  => 3.1200215770101565
    0,3 < energie < 100 => 3.1993541276651527
    0,0 < energie < 800 => 3.2354675179262467 '''
    
# Vorselektieren
upper_lim = 800
lower_lim = 0.0
#X = np.array(energie.drop(energie[((energie<0.2) | (energie>8))].index))
X = np.array(energie[((energie>lower_lim) & (energie<upper_lim))])
print(X)
Xt, lmbda = boxcox(X)
print(Xt)
inv = inv_boxcox(Xt, lmbda)
print(inv)

h_abs1, x1 = np.histogram(X, bins=500)
# plot histogramm
ax = plt.figure(figsize=(6, 4)).subplots(1, 1)
ax.hist(X, x1 , histtype='bar', weights=np.ones(X.shape)/X.shape, rwidth=0.9, cumulative=False)
ax.grid(True, which= 'both', axis='both', linestyle='--')
ax.set_xlabel('Energie')
ax.set_ylabel('Häufigkeit h(E)')
#ax.set_ylim(top=0.25)
ax.set_title('vor Transformation mit ' + str(lower_lim) + ' < energie < ' + str(upper_lim))
plt.tight_layout()


h_abs2, x2 = np.histogram(Xt, bins=50)
# plot histogramm
ax = plt.figure(figsize=(6, 4)).subplots(1, 1)
ax.hist(Xt, x2 , histtype='bar', weights=np.ones(Xt.shape)/Xt.shape, rwidth=0.9, cumulative=False)
ax.grid(True, which= 'both', axis='both', linestyle='--')
ax.set_xlabel('Energie')
ax.set_ylabel('Häufigkeit h(E)')
#ax.set_ylim(top=0.25)
ax.set_title('nach Transformation mit ' + str(lower_lim) + ' < energie < ' + str(upper_lim))
plt.tight_layout()


#%%

# Berechnung des Prognosebereichs
data = Xt
N = np.size(data)
data_mean = np.mean(data)
data_var = np.var(data, ddof = 1)
data_std = np.std(data, ddof = 1)
Einheit_unit = ''

'''
# Unbekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma1 = 0.95# 0.9973
c1 = t.ppf((1 - gamma1) / 2, N - 1)
c2 = t.ppf((1 + gamma1) / 2, N - 1)
x_prog_min = data_mean + c1 * data_std * np.sqrt(1 + 1 / N)
x_prog_max = data_mean + c2 * data_std * np.sqrt(1 + 1 / N)
print('Prognosewert:', round(x_prog_min, 4), Einheit_unit, '< x <=', round(x_prog_max, 4), Einheit_unit)
'''

# Unbekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma1 = 0.95# 0.9973
c = t.ppf((gamma1), N - 1)
x_prog_max = data_mean + c * data_std * np.sqrt(1 + 1 / N)
print('Prognosewert:', x_prog_max)

#%%

x_lim_max = inv_boxcox(x_prog_max, lmbda)
print(x_lim_max)