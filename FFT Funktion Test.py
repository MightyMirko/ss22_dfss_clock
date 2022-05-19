# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:23:37 2022

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
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.signal import windows


''' FFT-Funktion mit Fensterfunktionen und Plot '''
def FFT_function(signal, samplerate, win = 'none', pl=False, log=False):
    #Anzahl Sampels erfassen & zugehöriger zeitvektor erstellen
    sig_length = len(signal)
    # Abtastzeit
    T = 1 / samplerate

    # Auswahl Fensterfunktion
    if win == 'none':   # Rechteck
        w = np.ones(sig_length)
    elif win == 'hann':
        w = windows.hann(sig_length)
    elif win == 'hamming':
        w = windows.hamming(sig_length)
    elif win == 'blackman':
        w = windows.blackman(sig_length)
    else:
        print('wrong input keyword for window')
        return
    
    # Umwandlung in Array
    y = np.array(signal)
    # Transformation
    xf = fftfreq(sig_length, T)[:sig_length//2]
    ywf = fft(y*w)[:sig_length//2]
    
    # Plot
    if pl == True:
        if log == False:    #linear
            plt.plot(xf, 2/sig_length *np.abs(ywf))
            plt.grid()
            plt.show()
        if log == True:    #logarithmisch
            plt.loglog(xf[1:], 2/sig_length *np.abs(ywf[1:]))
            plt.grid()
            plt.show()
    return xf, ywf


''' FFT-Funktion ohne Fensterfunktionen und Plot '''
def FFT_func(signal, samplerate):
    #Anzahl Sampels erfassen
    sig_length = len(signal)
    # Abtastzeit
    T = 1 / samplerate
    # Umwandlung in Array
    y = np.array(signal)
    # Transformation
    xf = fftfreq(sig_length, T)[:sig_length//2]
    yf = fft(y)[:sig_length//2]
    return xf, 2/sig_length *np.abs(yf)


#%%


#einlesen & konvertieren
samplerate, data = wavfile.read('audio_for_test.wav')
signal = pd.Series(data) 

FFT_function(signal, samplerate, pl=True, log=False)

x, y = FFT_func(signal, samplerate)
plt.plot(x, y)
plt.grid()
