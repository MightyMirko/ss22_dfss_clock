# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:48:13 2022

@author: Alexander
"""

# Bibliotheken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.io import wavfile
#from scipy import signal as sig
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.signal import hilbert
from scipy.signal import windows
from scipy.signal import medfilt
from scipy.fft import fft, fftfreq
from scipy.stats import norm, t, chi2, f, weibull_min, gamma

from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.signal import decimate

from datetime import datetime


''' Funktionen '''


def calc_energie(signal):
    ''' für einzelen Signale
        Berechnet die Energie für ein komplettes Signal unabhängig ob
        Dataframe oder Array (Array ist effizienter)
        und gibt einen einzelnen Wert zurück'''
    return np.sum(signal**2)


def calc_energie_np(data):
    ''' Für numpy Matrizen
        Berechnet die Energie für einen kompletten Datensatz
        Datensatz darf nur aus Signalen bestehen! keine Dateinamen ''' 
    #return np.sum(np.square(data), axis=1)
    # oder
    return np.sum(data**2, axis=1)

def calc_power_seg(signal):
    # Energie Lesitung segmentiert bestimmen
    samplerate = 48000
    window_size = 2000
    matrix = signal.reshape(len(signal)//window_size,window_size)
    power_seg = samplerate * np.mean(matrix**2, axis=1)
    return power_seg


def Klassenzuweisung(sec_wert):
    ''' Hier wird erstmal der Sekundenwert der Datei übergeben
        zur Kompensation des statischen Fehlers werden noch 3 Sekunden
        dazu addiert
        Gibt die Kalassenzuweisung und den korrigierten Sekundenwert zurück '''
        
    # Kompensation des statischen Fehlers
    sec_wert = int(sec_wert) + 3
    # Überlauf verhinden
    sec_wert = sec_wert % 60
    out = ''
    if (sec_wert >= 0) and (sec_wert < 7.5):
        out = 'oben'
    elif (sec_wert > 7.5) and (sec_wert < 22.5):
        out = 'rechts'
    elif (sec_wert > 22.5) and (sec_wert < 37.5):
        out = 'unten'
    elif (sec_wert > 37.5) and (sec_wert < 52.5):
        out = 'links'
    elif (sec_wert > 52.5) and (sec_wert < 60):
        out = 'oben'
    else:
        out = 'Fehler!!!'
    return out, sec_wert





#%%

''' Start der Main '''

path_audio = r'Data/Audios'         # Pfad auf das Verzeichniss
file_list = os.listdir(path_audio)  # Liste der Dateien erstellen
number_of_files = len(file_list)    # Anzahl der Dateien ermitteln


# Test
testgroße = 1000
#X = np.empty((testgroße, 96000))

tick_list = []
reduc_f = 2           # Downsampling-Faktor



#for idx in range(len(file_list)):
for idx in range(0, testgroße):  
    
    ''' Daten einlesen '''
    samplerate, data = wavfile.read(path_audio + '/' + file_list[idx])  # einlesen
    
    # Downsampling
    data_r = decimate(data, reduc_f)
    samplerate_r = round(samplerate / reduc_f)
    
    # Grenzen des Signals
    lower_limit = 0
    upper_limit = round(2*samplerate_r)
    # Ausdehneung des Ticks
    left_expansion = round(0.04*samplerate_r)
    right_expansion = round(0.1*samplerate_r)
    
    
    ''' Energie filter '''
    energie = calc_energie(data)                # Energie berechnen 
    if (0.3 < energie) & (energie < 3.05):      # aussortieren
        
    
        ''' Tick-Detektion 1. Tick '''
        pos_max = np.argmax(data_r**2)               # Tick finden
        
        ''' Intervallgrenzen prüfen '''
        
        if (pos_max - left_expansion) < 0:             # linke Intervallgrenze
            # löschen des nicht schneidbaren Ticks
            data_r[lower_limit:(pos_max + right_expansion)] = 0
            
        elif (pos_max + right_expansion) > (upper_limit-1): # rechte Intervallgrenze
            # löschen des nicht schneidbaren Ticks
            data_r[(pos_max - left_expansion):upper_limit] = 0
            
        else:
            ''' Auschneiden des Ticks und Klassenzuweisung '''
            tick = np.array(data_r[(pos_max - left_expansion):(pos_max + right_expansion)])
            out, sec_wert = Klassenzuweisung(file_list[idx][15:17])
            #tick_list.append([file_list[idx], sec_wert, out, tick])
            tick_list.append([file_list[idx], sec_wert, out] + list(tick))
            data_r[(pos_max - left_expansion):(pos_max + right_expansion)] = 0
            
        
    
        ''' Tick-Detektion 2.Tick '''
        pos_max = np.argmax(data_r**2)               # Tick finden
        
        ''' Intervallgrenzen prüfen '''
        
        if (pos_max - left_expansion) < 0:             # linke Intervallgrenze
            # löschen des nicht schneidbaren Ticks
            data_r[lower_limit:(pos_max + right_expansion)] = 0
            
        elif (pos_max + right_expansion) > (upper_limit-1): # rechte Intervallgrenze
            # löschen des nicht schneidbaren Ticks
            data_r[(pos_max - left_expansion):upper_limit] = 0
            
        else:
            ''' Auschneiden des Ticks und Klassenzuweisung '''
            tick = data_r[(pos_max - left_expansion):(pos_max + right_expansion)]
            out, sec_wert = Klassenzuweisung(file_list[idx][15:17])
            #tick_list.append([file_list[idx], sec_wert, out, tick])
            tick_list.append([file_list[idx], sec_wert, out] + list(tick))
        

#%%
  
# convert to CSV
#df = pd.DataFrame(tick_list, columns =['Dateiname', 'korrZeit', 'Klasse', 'Ticken']
df = pd.DataFrame(tick_list, columns =['Dateiname', 'korrZeit', 'Klasse'] + list(range(3360)))
filepath = './CSV-Files/Out.csv'
df.to_csv(filepath, index=True)


