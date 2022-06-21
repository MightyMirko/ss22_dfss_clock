# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:54:00 2022

@author: Alexander
"""

import matplotlib.pyplot as plt
# Bibliotheken
import numpy as np
import pandas as pd

# from scipy import signal as sig


# C:\Users\Alexander\PycharmProjects\ss22_dfss_clock\data\csv

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






