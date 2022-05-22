# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 00:52:33 2022

@author: Alexander
"""

import numpy as np 
import pandas as pd 
from scipy import signal
import matplotlib.pyplot as plt


b, a = signal.butter(4, 100, 'low', analog=True)

w, h = signal.freqs(b, a)

plt.semilogx(w, 20 * np.log10(abs(h)))

plt.title('Butterworth filter frequency response')

plt.xlabel('Frequency [radians / second]')

plt.ylabel('Amplitude [dB]')

plt.margins(0, 0.1)

plt.grid(which='both', axis='both')

plt.axvline(100, color='green') # cutoff frequency

plt.show()


#%%

t = np.linspace(0, 0.2, 1000, False)  # 1 second

sig = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*300*t)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')

ax1.plot(t, sig)

ax1.set_title('50 Hz and 300 Hz sinusoids')

ax1.axis([0, 0.2, -2, 2])


sos = signal.butter(2, 100, 'hp', fs=1000, output='sos')

filtered = signal.sosfilt(sos, sig)

ax2.plot(t, filtered)

ax2.set_title('After 50 Hz high-pass filter')

ax2.axis([0, 0.2, -2, 2])

ax2.set_xlabel('Time [seconds]')

plt.tight_layout()

plt.show()