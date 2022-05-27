# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:47:33 2022

@author: Alexander
"""

#Bibliotheken
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from scipy.io import wavfile
#from scipy import signal as sig
from scipy.signal import butter
from scipy.signal import sosfilt

#einlesen & konvertieren
samplerate, data = wavfile.read('20220412_16_19_26.wav')
signal = pd.Series(data) 

#Anzahl Sampels erfassen & zugehöriger zeitvektor erstellen
sig_length = len(signal)
t = np.linspace(0.0, 2.0, num=sig_length)

#plot Signal gesamt
plt.figure()
plt.plot(t, signal)
plt.grid(True)
plt.xlabel('Zeit in signal')
plt.ylabel('Amplitude')
plt.title('original Signal')
plt.show()

#plot Ticken
plt.figure()
#xmin, xmax, ymin, ymax = 1.2, 1.3, -0.005, 0.005
#plt.axis([xmin, xmax, ymin, ymax])
plt.xlim(1.2, 1.3)
plt.plot(t, signal)
plt.grid(True)
plt.xlabel('Zeit in signal')
plt.ylabel('Amplitude')
plt.title('Ticken im original Signal')
plt.show()

#%%

''' Butterworth-Hochpassfilter
    wn = 150 bestes Ergebnis für Signal
    wn = 500 bestes Ergebnis für Rauschen '''

wn = 500 
sos = butter(10, wn, 'hp', fs=samplerate, output='sos')
filtered = sosfilt(sos, signal)

#plot gefiltertes Signal gesamt
plt.figure()
plt.plot(t, filtered)
plt.grid(True)
plt.xlabel('Zeit in signal')
plt.ylabel('Amplitude')
plt.title('HP gefiltert mit wn = %i' %wn)
plt.show()

#plot Ticken
plt.figure()
#xmin, xmax, ymin, ymax = 1.2, 1.3, -0.003, 0.003
#plt.axis([xmin, xmax, ymin, ymax])
plt.xlim(1.2, 1.3)
plt.plot(t, filtered)
plt.grid(True)
plt.xlabel('Zeit in signal')
plt.ylabel('Amplitude')
plt.title('HP gefiltert mit wn = %i' %wn)
plt.show()


#%%

''' Moving Average-Filter
    Achtung!!! Anzahl an Sampels wird um k-1 verringert '''

# Fenstergröße k festlegen (nur ungerade k erlaubt!!!)
window_size = 5
# übergabe des Signals und convertierung in pandas Series
numbers_series = pd.Series(filtered)
# Auschneidern der Beobachtungsreihe entsprechend der Fenstergröße
windows = numbers_series.rolling(window_size)
# Mittelwert jedes Ausschnittes bilden
moving_averages = windows.mean()
# NaN entfernen
moving_averages = moving_averages[window_size - 1:]


#plot gefiltertes Signal gesamt
plt.figure()
plt.plot(t[(window_size//2):-(window_size//2)], moving_averages)
plt.grid(True)
plt.xlabel('Zeit in signal')
plt.ylabel('Amplitude')
plt.title('Moving Average gefiltert')
plt.show()

#plot Ticken
plt.figure()
#xmin, xmax, ymin, ymax = 1.2, 1.3, -0.003, 0.003
#plt.axis([xmin, xmax, ymin, ymax])
plt.xlim(1.2, 1.3)
plt.plot(t[(window_size//2):-(window_size//2)], moving_averages)
plt.grid(True)
plt.xlabel('Zeit in signal')
plt.ylabel('Amplitude')
plt.title('Moving Average gefiltert')
plt.show()


#%%

#plot Ticken genauer
plt.figure()
#xmin, xmax, ymin, ymax = 1.2, 1.3, -0.003, 0.003
#plt.axis([xmin, xmax, ymin, ymax])
plt.xlim(1.23, 1.26)
plt.plot(t[(window_size//2):-(window_size//2)], moving_averages)
plt.grid(True)
plt.xlabel('Zeit in signal')
plt.ylabel('Amplitude')
plt.title('Untersuchung Ticken')
plt.show()
