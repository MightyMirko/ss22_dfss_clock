# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:42:31 2022

@author: Alexander
"""

# Bibliotheken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.io import wavfile
# from scipy import signal as sig
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.signal import windows
from scipy.signal import medfilt
from scipy.stats import norm, t, chi2, f
from tqdm.notebook import tqdm

# %%
if os.path.exists('H:\Messung_BluetoothMikro\Messung 3\Audios'):
    path_audio = r'H:\Messung_BluetoothMikro\Messung 3\Audios'
file_list = os.listdir(path_audio)
file_list = file_list[100:9000]
number_of_files = len(file_list)
#p = tqdm(total=number_of_files, disable=False)

print(number_of_files)
energie = np.zeros(number_of_files)

# Energie bestimmen absolut
for idx in range(len(file_list)):
    # einlesen
    samplerate, data = wavfile.read(path_audio + '/' + file_list[idx])
    # Berechnung
    signal_2 = data ** 2
    energie[idx] = signal_2.sum()
    if idx % 100 == 0:
        print(idx)

print('Fertig:')

# %%
# plot Energie der Signale
plt.rcParams['figure.dpi'] = 300
# plt.rcParams['interactive'] = True
plt.rcParams['figure.figsize'] = (12, 9)

plt.figure()
# plt.scatter(x=,y=energie)
# plt.xlim(xmin=20, xmax=30)
plt.plot(energie)

plt.grid(True)
plt.xlabel('Anzahl Aufnahmen')
plt.ylabel('Energie')
plt.title('absolute Energie der Aufnahmen')
plt.show()

# %%
X = energie
h_abs, x = np.histogram(X, bins=1000)

ax = plt.figure(figsize=(6, 4)).subplots(1, 1)
ax.hist(X, x, histtype='bar', weights=np.ones(X.shape) / X.shape, rwidth=0.9, cumulative=False)
ax.grid(True, which='both', axis='both', linestyle='--')
ax.set_xlabel('Energie')
ax.set_ylabel('Häufigkeit h(d)')
# ax.set_ylim(top=0.001)
# ax.set_xlim(left=10, right=1000)

ax.set_title('Häufigkeitsverteilung')
plt.tight_layout()

plt.show()

# %%

''' weitere Analysen '''

# Berechnung des Prognosebereichs
data = energie
N = np.size(data)
data_mean = np.mean(data)
data_var = np.var(data, ddof=1)
data_std = np.std(data, ddof=1)
Einheit_unit = ''

# Unbekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma = 0.95
c1 = t.ppf((1 - gamma) / 2, N - 1)
c2 = t.ppf((1 + gamma) / 2, N - 1)
x_prog_min = data_mean + c1 * data_std * np.sqrt(1 + 1 / N)
x_prog_max = data_mean + c2 * data_std * np.sqrt(1 + 1 / N)
print('Prognosewert:', round(x_prog_min, 4), Einheit_unit, '< x <=', round(x_prog_max, 4), Einheit_unit)

# %%

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
# plt.plot(x, y*x_prog_min)
plt.plot(x, y * x_prog_max)
plt.plot(x, y * np.median(energie))
plt.plot(x, y * np.mean(energie))
plt.grid(True)
plt.show()

# %%
plt.figure()
plt.scatter(x, energie,color='b')
plt.grid(True)
plt.xlabel('Anzahl Aufnahmen')
plt.ylabel('Energie')
plt.title('absolute Energie der Aufnahmen')
# plt.show()
# %%
birne = np.max(energie)
apfel = np.argmax(energie)
faulerapfel = file_list[int(apfel)]

# %%
#
energie[int(apfel)] = 0
# %%
# plot Prognosebereich
# plt.plot(x, y*x_prog_min)
plt.plot(x, y * x_prog_max, color='green', linewidth=5, label='Max Prognose')
plt.plot(x, y * np.median(energie),color='k',label='Median')
plt.plot(x, y * np.mean(energie), color='r', label='ar.Mittelwert')
plt.grid(True)
plt.legend()
plt.show()
