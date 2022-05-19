# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:36:50 2022

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

# Number of sample points
N = 500

# sample spacing
T = 1 / 800

x = np.linspace(0, N*T, N, endpoint=False)
y = np.sin(50*2*np.pi*x) + 0.5*np.sin(80*2*np.pi*x)

yf = fft(y)[0:N//2]
xf = fftfreq(N, T)[:N//2]

plt.plot(xf, 2.0*np.pi /N * np.abs(yf))
plt.grid()
plt.show()

#%%

from scipy.signal import blackman

# Number of sample points
N = 96000

# sample spacing
T = 1 / 48000

x = np.linspace(0, N*T, N)#, endpoint=False)
y = np.sin(50*2*np.pi*x) + 0.5*np.sin(80*2*np.pi*x)
w = blackman(N)

xf = fftfreq(N, T)[:N//2]
yf = fft(y)[:N//2]
ywf = fft(y*w)[:N//2]

#linear plot
plt.xlim(0, 200)
plt.plot(xf, 2/N *np.abs(yf))
plt.plot(xf, 2/N *np.abs(ywf))
plt.grid()
plt.show()

#Bode plot
plt.xlim(0, 400)
plt.semilogy(xf[1:], 2/N *np.abs(yf[1:]))
plt.semilogy(xf[1:], 2/N *np.abs(ywf[1:]))
plt.grid()
plt.show()


#%%

from scipy.fft import fft, fftfreq

# Number of sample points

N = 96000

# sample spacing

T = 1.0 / 48000.0

x = np.linspace(0.0, N*T, N)#, endpoint=False)

y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

yf = fft(y)

from scipy.signal import blackman

w = blackman(N)

ywf = fft(y*w)

xf = fftfreq(N, T)[:N//2]

import matplotlib.pyplot as plt

plt.xlim(0, 400)
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')

plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')

plt.legend(['FFT', 'FFT w. window'])

plt.grid()

plt.show()




