# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:42:31 2022

@author: Alexander

Energie-Berechnung und
Schwellwert-Abschätzung für Energie-Filter
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

#%%

# Pfad auf das Verzeichniss
path_audio = r'data/Neuer Ordner'
# Liste der Dateien erstellen
file_list = os.listdir(path_audio)

# Manuelle Filterung der Daten
# Schlechte Dateien aussortieren
file_list.remove('20220510_02_27_09.wav')
file_list.remove('20220510_05_30_45.wav')
file_list.remove('20220510_05_48_54.wav')
file_list.remove('20220510_09_28_01.wav')
file_list.remove('20220510_09_28_07.wav')
file_list.remove('audio_for_test.wav')
file_list.remove('ausreißer_mit_tick.wav')

# Anzahl der Dateien ermitteln
number_of_files = len(file_list)
# Ergebnisspeicher allokieren
energie = np.zeros(number_of_files)

# absolute Energie bestimmen
for idx in range(len(file_list)):
    #einlesen
    samplerate, data = wavfile.read(path_audio + '/' + file_list[idx])
    
    # plot
    plt.figure()
    plt.plot(data)
    plt.grid(True)
    plt.title(file_list[idx])
    plt.show()
    
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

''' Histogramm '''

X = energie
h_abs, x = np.histogram(X, bins=50)

ax = plt.figure(figsize=(6, 4)).subplots(1, 1)
ax.hist(X, x , histtype='bar', weights=np.ones(X.shape)/X.shape, rwidth=0.9, cumulative=False)
ax.grid(True, which= 'both', axis='both', linestyle='--')
ax.set_xlabel('Energie')
ax.set_ylabel('Häufigkeit h(E)')
#ax.set_ylim(top=0.25)
#ax.set_title('Kommulative Häufigkeitsverteilung')
plt.tight_layout()



#%%

''' Prognosebereich bestimmen '''

# Berechnung des Prognosebereichs
data = energie
N = np.size(data)
data_mean = np.mean(data)
data_var = np.var(data, ddof = 1)
data_std = np.std(data, ddof = 1)
Einheit_unit = ''

'''
# Unbekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma = 0.95# 0.9973
c1 = t.ppf((1 - gamma) / 2, N - 1)
c2 = t.ppf((1 + gamma) / 2, N - 1)
x_prog_min = data_mean + c1 * data_std * np.sqrt(1 + 1 / N)
x_prog_max = data_mean + c2 * data_std * np.sqrt(1 + 1 / N)
print('Prognosewert:', round(x_prog_min, 4), Einheit_unit, '< x <=', round(x_prog_max, 4), Einheit_unit)
'''

# Unbekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma = 0.95# 0.9973
c = t.ppf((gamma), N - 1)
x_prog_max = data_mean + c * data_std * np.sqrt(1 + 1 / N)
print('Prognosewert:', x_prog_max)


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
#plt.plot(x, y*x_prog_min)
plt.plot(x, y*x_prog_max)
plt.plot(x, y*np.median(energie))
plt.plot(x, y*np.mean(energie))
plt.grid(True)
plt.show()

#%%

''' Approximation als Weibull-Verteilung '''

x = np.arange(0,8,0.1)     # x manuell festlegen
eta = 1.1
beta = 1.1#np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
WB = weibull_min.pdf(x, c=beta, loc=0.25, scale=eta)

fig, ax = plt.subplots(1, 1)
ax.plot(x, WB)
ax.grid(True)
ax.set_xlabel("Energie")
ax.set_ylabel("Häufigkeitsverteilung")

#%%

''' Approximation als Gamma-Verteilung '''

x = np.arange(0,8,0.1)     # x manuell festlegen
b = 1.1
p = 1.1#np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
gam = gamma.pdf(x, a=p, loc=0.25, scale=b)

fig, ax = plt.subplots(1, 1)
ax.plot(x, gam)
ax.grid(True)
ax.set_xlabel("Energie")
ax.set_ylabel("Häufigkeitsverteilung")


#%%

''' Approximation als Chi2-Verteilung '''

