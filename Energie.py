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

''' weitere Analysen der Ausreiser '''