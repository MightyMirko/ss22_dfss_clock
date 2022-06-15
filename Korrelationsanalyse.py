# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:54:00 2022

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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder



#C:\Users\Alexander\PycharmProjects\ss22_dfss_clock\data\csv

DATA_FILE_TRAIN = 'data/csv/26-29-output.csv'

signal = pd.read_csv(DATA_FILE_TRAIN, header=0)
samplerate = 48000

#Anzahl Sampels erfassen & zugehöriger Zeitvektor erstellen
sig_length = len(signal)
t = np.linspace(0.0, sig_length/samplerate, num=sig_length)
# Abtastzeit
T = 1/samplerate

plt.figure()
plt.plot(t, signal)
plt.grid(True)
plt.xlabel('Zeit in s')
plt.ylabel('Amplitude')
plt.title('original Signal')
plt.show()
'''
print(df.head)
print(df.tail)
print(df.iloc[[0:], [6721]])
'''






