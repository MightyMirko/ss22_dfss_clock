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

