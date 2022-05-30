# @Author Mirko Matosin
# @Date 14.02.2022

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.dpi'] = 150
# plt.rcParams['interactive'] = True
plt.rcParams['figure.figsize'] = (12, 9)
from scipy.io import wavfile
from os.path import join as pjoin
from os import listdir

import tqdm
import os

import pandas
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
import numpy as np

import time
import joblib
from joblib import Parallel, delayed


def plotsounds(ax, data, abszisse, param_dict={}, filename='N/A', samplerate=48000, log=True,
               xlab='Time [signal]', ylab='Amplitude', title='Ticken im Original'):
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


from tqdm.notebook import tqdm


def getwav_fromfolder(wavdir, do_test=False):
    """
    Sucht alle Wavdateien in einem Ordner und speichert die Rohdaten in einem gemeinsamenem Objekt
    :param wavdir: Pfad wo die wav daten liegen
    :return: relativer Pfad mit Dateiname und das pandas Dataframe für die Rohdaten
    """
    wavfiles = []
    filenames = []
    df = pd.DataFrame()

    show_progress = True
    t = tqdm(total=1, unit="file", disable=not show_progress)

    if not os.path.exists(wavdir):
        raise IOError("Cannot find:" + wavdir)

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
                # df['samplerate'] = samplerate
                # break
        t.set_postfix(dir=wavdir)
    t.close()

    return wavfiles, df


def chroma_plot(featvec, names):
    fig300, scat = plt.subplots(12, 1)
    x = 21
    while x <= 32:
        scat[x - 21].plot(featvec[x])
        scat[x - 21].set_ylabel(names[x])
        x += 1
    fig300.show()
    plt.close()
    return


def fn_plot(featvec, names, s):
    fig300, scat = plt.subplots(8, 1)
    x = 0
    while x < 7:
        scat[x].plot(featvec[x])
        scat[x].set_ylabel(names[x])
        x += 1
    scat[7].plot(s)
    fig300.tight_layout()
    fig300.show()
    plt.close()


def plot_energy(f, fn, filename):
    energy = f[fn.index('energy'), :]  # Alle Daten aus der index nummer 1 ziehen..
    d_energy = f[fn.index('delta energy'), :]
    dd_energy = np.diff(d_energy)
    ddd_energy = np.diff(dd_energy)

    fig1, axs = plt.subplots(4, 1, sharex='col')
    axs[0].set_title(filename)
    axs[0].plot(energy)
    axs[0].set_ylabel(fn[1])
    axs[1].plot(d_energy)
    axs[1].set_ylabel(fn[1] + ' 1. Ableitung')
    axs[2].plot(dd_energy)
    axs[2].set_ylabel(fn[1] + ' 2. Ableitung')
    axs[3].plot(ddd_energy)
    axs[3].set_xlabel('Frame Number')
    axs[3].set_ylabel(fn[1] + ' 3. Ableitung')
    for i in axs:
        i.grid()
    fig1.show()


def processFolder(audiofile, audio_dir, win=0.005, step=0.005):
    if not '.wav' in audiofile:
        return 0, 0, 0, 0
    if '16_10_21' in audiofile:
        return 0, 0, 0, 0
    audiofile = os.path.join(audio_dir + "\\" + audiofile)
    fs, signal = aIO.read_audio_file(audiofile)
    ###############
    # Anlegen der Variablen
    ###############
    duration = len(signal) / float(fs)
    time = np.arange(0, duration - step, win)
    # extract short-term features using a 50msec non-overlapping windows
    [f, fn] = aF.feature_extraction(signal, fs, int(fs * win), int(fs * step), True)
    return f, fn, fs, signal


def cut_signal(f, fn, s, fs=48000):
    energy = f[fn.index('energy'), :]  # Alle Daten aus der index nummer 1 ziehen..
    d_energy = f[fn.index('delta energy'), :]
    dd_energy = np.diff(d_energy)
    ddd_energy = np.diff(dd_energy)

    indexofpeak = d_energy.argmax()  # Wo ist das Maximum
    peak_in_ms = win * indexofpeak
    back_in_ms = peak_in_ms - 0.04
    adv_in_ms = peak_in_ms + 0.1
    back_in_sample = int(fs * back_in_ms)
    adv_in_sample = int(fs * adv_in_ms)
    deltasample = adv_in_sample - back_in_sample
    olds = s.copy()
    tmps = s[back_in_sample:adv_in_sample]
    news = s.copy()
    news[back_in_sample:adv_in_sample] = s[back_in_sample:adv_in_sample] * 0

    return olds, tmps, news


def rutar(a, b):
    return a + b, a * b, str(a) + str(b)


if __name__ == "__main__":

    #############
    # Anlegen der Kontrollvariablen
    ############
    print('hello World')
    do_plot = False  # Plotten der Graphen zum Debuggen
    do_test_mode = False  # Diverse Beschleuniger
    do_play = False  # außer Betrieb

    csv_exp = []  # Der Haupt-Dataframe
    zeilennamen = []
    audio_dir = ''
    if os.path.exists('H:\Messung_BluetoothMikro\Messung 3\Audios'):
        audio_dir = r'H:\Messung_BluetoothMikro\Messung 3\Audios'
    audio_dir = r'../data'

    plt.clf()
    iteration_over_file = 0
    anzahl_bearbeitet = 0
    wavfiles = listdir(audio_dir)
    anzahl = len(wavfiles)
    anzahlnochnicht = anzahl
    csvlength = 50  # Achtung es werden die Zeilen 2x gezählt -> 50 dateien = 100 zeilen
    ###############
    # Anlegen der Variablen aus den statistischen Methoden
    ###############
    win, step = 0.005, 0.005

    # r steht für roh/raw.. Damit lassen sich windows pfade verarbeiten
    # damit sehen wir sofort welche variable wie verarbeitet wird. Erleichtert die Lesbarkeit.
    # ergibt ein Tupel aus Spaltenname und Serie für jede Spalte im Datenrahmen:
    #############
    # Untersuchen des Signals und Fenstern
    ############

    number_of_cpu = joblib.cpu_count()
    wavfiles = [x for x in wavfiles if not '16_10_21' in x]
    wavfiles = [x for x in wavfiles if 'wav' in x]

    delayed_func = [delayed(processFolder)(audiofile, audio_dir, win, step) for audiofile in wavfiles]
    # delayed_func = [delayed(rutar)(i,3) for i in range(0,100)]
    parallel_pool = Parallel(n_jobs=number_of_cpu, verbose=10)
    f, fn, fs, signal = {}, {}, {}, {},
    res = parallel_pool(delayed_func)

    try:
        farr = [item[0] for item in res]
        fnarr = [item[1] for item in res]
        fsarr = [item[2] for item in res]
        signalarr = [item[3] for item in res]
    except:
        pass
    i = 0
    zeilennamen = []
    df = pd.DataFrame(signalarr, index=wavfiles)
    fn = fnarr[0]
    fs = fsarr[2]
    fdf = pd.DataFrame()
    for it in farr:
        fserdf = pd.DataFrame(it)
        pd.concat(fdf, fserdf)
        print(it.shape)
    fdf.reindex(wavfiles)
    # delayed_func = [delayed(cut_signal)(feature, fn, signal) for feature in f]

    # pd.DataFrame(result)

    #########################
    # achtung while Schleife ende
    #########################
    print('bye world')
