# -*- coding: utf-8 -*-
"""
Created on Mon May 30 21:20:06 2022

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
from scipy.signal import medfilt

#Funktionen
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

#Anzahl Sampels erfassen & zugehöriger Zeitvektor erstellen
sig_length = len(signal)
t = np.linspace(0.0, sig_length/samplerate, num=sig_length)
# Abtastzeit
T = 1/samplerate

#plot Signal gesamt
plt.figure()
plt.plot(t, signal)
plt.grid(True)
plt.xlabel('Zeit in s')
plt.ylabel('Amplitude')
plt.title('original Signal')
plt.show()

#plot Ticken
plt.figure()
#xmin, xmax, ymin, ymax = 1.2, 1.3, -0.005, 0.005
#plt.axis([xmin, xmax, ymin, ymax])
plt.xlim(1.34, 1.45)
plt.plot(t, signal)
plt.grid(True)
plt.xlabel('Zeit in s')
plt.ylabel('Amplitude')
plt.title('Ticken im original Signal')
plt.show()

#%%


''' FFT-Analyse'''
''' FFT-Analyse Gesamtsignal '''

# Vorberechnung der Fensterfunktionen
w_hamming = windows.hamming(sig_length)
w_hann = windows.hann(sig_length)
w_blackman = windows.blackman(sig_length)

# Auswahl der Fensterfunktion
w = w_hamming = windows.hamming(sig_length)

# Transformation
x, y = FFT_func(signal, samplerate)

# Plot
'''
plt.loglog(x, y)
plt.grid(True)
plt.xlabel('Frequenz in Hz')
plt.ylabel('Amplitude')
plt.title('FFT des Gesamtsignals')
plt.show()
'''
plt.semilogy(x, y)
plt.grid(True)
plt.xlabel('Frequenz in Hz')
plt.ylabel('Amplitude')
plt.title('FFT des Gesamtsignals')
plt.show()


''' FFT-Analyse Ticken '''

# Position des Tickens zwischen 1.34 bis 1.45 Sekunden
a_idx = 64320 # Anfangsindex
e_idx = 69600 # Endindex

# Länge des Tickens
tick_length = e_idx-a_idx
w = windows.hamming(tick_length)

# Transformation
x, y = FFT_func(signal[a_idx:e_idx]*w, samplerate)

# Plot
'''
plt.loglog(x, y)
plt.grid(True)
plt.xlabel('Frequenz in Hz')
plt.ylabel('Amplitude')
plt.title('FFT des Tickens')
plt.show()
'''
plt.semilogy(x, y)
plt.grid(True)
plt.xlabel('Frequenz in Hz')
plt.ylabel('Amplitude')
plt.title('FFT des Tickens')
plt.show()

""" Ergebnis: 
    Das Ticken enthält Frequenzen von 0 bis ca. 7000 Hz 
    -> Tiefpass benötigt
    -> Hochpass optional """
    
    
#%%

''' Butterworth-Filter '''
''' Tiefpass '''

fg = 7000
#fg = 500
sos = butter(10, fg, 'low', fs=samplerate, output='sos')
filtered = sosfilt(sos, signal)

# nur zur Kontrolle ob sich das Signal verändert

#plot gefiltertes Signal gesamt
plt.figure()
plt.plot(t, filtered)
plt.grid(True)
plt.xlabel('Zeit in s')
plt.ylabel('Amplitude')
plt.title('TP gefiltert mit fg = %i' %fg)
plt.show()

#plot Ticken
plt.figure()
#xmin, xmax, ymin, ymax = 1.2, 1.3, -0.003, 0.003
#plt.axis([xmin, xmax, ymin, ymax])
plt.xlim(1.34, 1.45)
plt.plot(t, filtered)
plt.grid(True)
plt.xlabel('Zeit in s')
plt.ylabel('Amplitude')
plt.title('TP gefiltert mit fg = %i' %fg)
plt.show()



#%%

''' FFT-Analyse des Tickens nach den Filtern '''

tick_length = 69600-64320
w = windows.hamming(tick_length)

# Transformation
x, y = FFT_func(filtered[64320:69600]*w, samplerate)

# Plot
plt.loglog(x, y)
plt.grid(True)
plt.xlabel('Frequenz in Hz')
plt.ylabel('Amplitude')
plt.title('FFT des Tickens')
plt.show()

plt.semilogy(x, y)
plt.grid(True)
plt.xlabel('Frequenz in Hz')
plt.ylabel('Amplitude')
plt.title('FFT des Tickens')
plt.show()