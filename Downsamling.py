# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:18:48 2022

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

from scipy.signal import decimate


# Eingabewerte für Testumgebung
wave_duration = 3
freq = 3

''' Relevante Eingabewerte / WICHTIG!!!!!! Bei uns
    samplerate = 48000 
    Reduktionsfaktor q = 2 '''
sample_rate = 100
q = 2
''' ------------------------------------ '''

# Testumgebung blabla
samples = wave_duration*sample_rate
samples_decimated = int(samples/q)
x = np.linspace(0, wave_duration, samples, endpoint=False)
y = np.cos(x*np.pi*freq*2)
xnew = np.linspace(0, wave_duration, samples_decimated, endpoint=False)


''' Entscheidende Funktion '''
ydem = decimate(y, q)  # keine Vorfilterung notwendig!


# plot zu Kontrolle
plt.plot(x, y, '.-', xnew, ydem, 'o-')
plt.xlabel('Time, Seconds')
plt.legend(['data', 'decimated'], loc='best')
plt.show()