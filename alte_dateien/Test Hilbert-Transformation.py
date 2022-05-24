# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:44:57 2022

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


dt=0.0001
t = np.arange(0,0.1,dt)
x = (1+np.cos(2*np.pi*50*t))*np.cos(2*np.pi*1000*t)
plt.plot(x)

h=abs(hilbert(x))    
plt.plot(h)