# @Author Mirko Matosin
# @Date 14.02.2022

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.dpi'] = 150
# plt.rcParams['interactive'] = True
plt.rcParams['figure.figsize'] = (21, 9)
from scipy.io import wavfile
from os.path import join as pjoin
from os import listdir
from scipy.fft import fft, fftfreq
from scipy.signal import butter
from scipy.signal import sosfilt


def plotSounds(ax, data, abszisse, param_dict={}, filename='N/A', samplerate=48000, log=True,
               xlab='Time [s]', ylab='Amplitude', title='Ticken im Original'):
    """
    Plottet eine Spalte oder ein Vektor in Timedomain

    :param data: hier kommt eine Series (Spalte Dataframe) hinein.
    :param filename: Legendenname
    :param samplerate: 96000 sind Standard. Wird für die Zeitachse genommen
    :param xlab: Label auf Abszisse
    :param ylab: Label auf Ordinate
    :return:
    """
    out = ax.plot(abszisse, data, label=filename, **param_dict)
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_xscale('linear')
    if log:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    # plt.xlim(xmax=1.31, xmin=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    return out


def getWavFromFolder(wavdir, do_test=False):
    """
    Sucht alle Wavdateien in einem Ordner und speichert die Rohdaten in einem gemeinsamenem Objekt
    :param wavdir: Pfad wo die wav daten liegen
    :return: relativer Pfad mit Dateiname und das pandas Dataframe für die Rohdaten
    """
    wavfiles = []
    filenames = []
    df = pd.DataFrame()
    for file in listdir(wavdir):
        if file.endswith(".wav"):
            import re
            filename = re.sub(r'.wav$', '', file)
            if do_test and not 'audio' in filename:
                continue
            else:
                pfad = pjoin(wavdir, file)
                wavfiles.append(pfad)
                filenames.append(filename)
                samplerate, data = wavfile.read(filename=pfad)
                df[filename] = data
                #df['samplerate'] = samplerate
                #break
    return wavfiles, df


from playsound import playsound
from multiprocessing import Process

if __name__ == "__main__":

    print('hello World')
    do_plot = True
    do_test_mode = True
    do_play = False

    audio_dir = (r'data/')  # r steht für roh/raw.. Damit lassen sich windows pfade verarbeiten
    files, df = getWavFromFolder(wavdir=audio_dir,do_test=False)  # ich mag explizite Programmierung. Also wavdir=...,
    # damit sehen wir sofort welche variable wie verarbeitet wird. Erleichtert die Lesbarkeit.
    plt.clf()

    # ergibt ein Tupel aus Spaltenname und Serie für jede Spalte im Datenrahmen:
    for (columnName, columnData) in df.iteritems():
        if do_test_mode and not '20220509_17_19_45' in columnName:
            continue
        if do_play:
            filepath = [audio_dir + columnName + '.wav']
            p = Process(target=playsound, args=(filepath))
            p.start()
        ## Normalisierung
        columnData = columnData / (2**32)
        # Braucht man dieses Lesen wirklich jedes Mal?
        samplerate = wavfile.read(filename=audio_dir+columnName+'.wav')[0]

        length = columnData.shape[0] / samplerate
        time = np.linspace(0., length, columnData.shape[0])
        omega_butter = 100

        if do_plot:
            pass
        else:
            print('Keine Plots erwünscht')
        break
        if do_play:
            p.join()
    plt.clf()
    plt.close()
    print('bye world')
