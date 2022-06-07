# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:42:31 2022

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
from scipy.stats import norm, t, chi2, f

#%%

path_audio = r'data/Neuer Ordner'
file_list = os.listdir(path_audio)

number_of_files = len(file_list)
energie = np.zeros(number_of_files)

# Energie bestimmen absolut
for idx in range(len(file_list)):
    #einlesen
    samplerate, data = wavfile.read(path_audio + '/' + file_list[idx])
    # Berechnung
    signal_2 = data**2
    energie[idx] = signal_2.sum()

# plot Energie der Signale
plt.figure()
plt.plot(energie)
plt.grid(True)
plt.xlabel('Anzahl Aufnahmen')
plt.ylabel('Energie')
plt.title('absolute Energie der Aufnahmen')
plt.show()

#%%

''' weitere Analysen '''

# Berechnung des Prognosebereichs
data = energie
N = np.size(data)
data_mean = np.mean(data)
data_var = np.var(data, ddof = 1)
data_std = np.std(data, ddof = 1)
Einheit_unit = ''

# Unbekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma = 0.9973
c1 = t.ppf((1 - gamma) / 2, N - 1)
c2 = t.ppf((1 + gamma) / 2, N - 1)
x_prog_min = data_mean + c1 * data_std * np.sqrt(1 + 1 / N)
x_prog_max = data_mean + c2 * data_std * np.sqrt(1 + 1 / N)
print('Prognosewert:', round(x_prog_min, 4), Einheit_unit, '< x <=', round(x_prog_max, 4), Einheit_unit)

#%%

''' Plots '''
N = len(energie)
x = np.arange(N)
y = np.ones(N)
# plot Energie der Signale
plt.figure()
plt.plot(x, energie)
plt.grid(True)
plt.xlabel('Anzahl Aufnahmen')
plt.ylabel('Energie')
plt.title('absolute Energie der Aufnahmen')

# plot Prognosebereich
plt.plot(x, y*x_prog_min)
plt.plot(x, y*x_prog_max)
plt.plot(x, y*np.median(energie))
plt.plot(x, y*np.mean(energie))
plt.grid(True)
plt.show()