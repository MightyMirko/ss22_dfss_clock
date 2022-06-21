# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:43:18 2022

@author: Alexander
"""

import os

import matplotlib.pyplot as plt
# Bibliotheken
import numpy as np
from scipy.io import wavfile
from scipy.special import inv_boxcox
from scipy.stats import boxcox
# from scipy import signal as sig
from scipy.stats import t

# %%

# Pfad auf das Verzeichniss
path_audio = r'../data/Neuer Ordner'
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
    '''
    # plot
    plt.figure()
    plt.plot(data)
    plt.grid(True)
    plt.title(file_list[idx])
    plt.show()
    '''
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

''' Transformation '''
Xt, lmbda = boxcox(energie)
print(Xt)

#%%

''' Prognosebereich bestimmen '''
# Berechnung des Prognosebereichs
data = Xt
N = np.size(data)
data_mean = np.mean(data)
data_var = np.var(data, ddof = 1)
data_std = np.std(data, ddof = 1)
Einheit_unit = ''

# Unbekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma1 = 0.95# 0.9973
c = t.ppf((gamma1), N - 1)
x_prog_max = data_mean + c * data_std * np.sqrt(1 + 1 / N)
print('Prognosewert:', x_prog_max)

#%%

''' Rücktransformation '''
x_lim_max = inv_boxcox(x_prog_max, lmbda)
print('Beste Prognose', + x_lim_max)

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
plt.plot(x, y*x_lim_max)
plt.plot(x, y*np.median(energie))
plt.plot(x, y*np.mean(energie))
plt.grid(True)
plt.legend(['Energie', 'Prognosewert', 'Median', 'Mittelwert'])
plt.show()
