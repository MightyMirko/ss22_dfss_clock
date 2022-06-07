# @Author Mirko Matosin
# @Date 14.02.2022
#


# fft anschauen. clustern, standardisieren, normalisieren
# isolation forest
# db scan

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

def square(x):
    return x**2

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
        return
    if '16_10_21' in audiofile:
        return
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


def cut_signal(f, fn, s):
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


if __name__ == "__main__":

    #############
    # Anlegen der Kontrollvariablen
    ############
    print('hello World')
    do_plot = False  # Plotten der Graphen zum Debuggen
    do_test_mode = True  # Diverse Beschleuniger
    do_play = False  # außer Betrieb

    csv_exp = []  # Der Haupt-Dataframe
    zeilennamen = []
    audio_dir = ''

    if do_test_mode:
        audio_dir = r'data'
    else:
        if os.path.exists('H:\Messung_BluetoothMikro\Messung 3\Audios'):
            audio_dir = r'H:\Messung_BluetoothMikro\Messung 3\Audios'

    plt.clf()

    duration = 0
    timew = 0
    iteration_over_file = 0
    anzahl_bearbeitet = 0
    wavfiles = listdir(audio_dir)
    wavfiles = wavfiles[:400]
    anzahl = len(wavfiles)
    anzahlnochnicht = anzahl
    csvlength = 3  # Achtung es werden die Zeilen 2x gezählt -> 50 dateien = 100 zeilen
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

    import time
    from joblib import Parallel, delayed

    """
    start_time = time.perf_counter()
    (result) = Parallel(n_jobs=25)(delayed(processFolder)(audiofile, audio_dir,win,step) for audiofile in wavfiles)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
    print(result)
    """

    if not do_test_mode:
        for audiofile in wavfiles:
            if not '.wav' in audiofile:
                continue
            if '16_10_21' in audiofile:
                continue
            anzahl_bearbeitet += 1
            anzahlnochnicht -= 1
            iteration_over_file = 0

            while iteration_over_file < 2:
                txt = ('Dies ist die {}. csvDatei von {} im {}. Durchlauf').format(anzahl_bearbeitet,
                                                                                   anzahl,
                                                                                   iteration_over_file)
                txt1 = ('Bearbeiten von {}.').format(audiofile)
                print(txt, '\n', txt1)
                f, fn, fs, signal = processFolder(audiofile, audio_dir, win, step)
                time = np.linspace(0., len(signal) / float(fs), len(signal))
                duration = len(signal) / float(fs)
                timew = np.arange(0, duration - step, win)

                ###############
                # Nehme den gefunden Index und schneide signal heraus in tmps
                ###############
                olds, tmps, news = cut_signal(f, fn, signal)
                zeile = audiofile.strip(audio_dir).strip('.wav') + 'tick' + str(iteration_over_file)
                zeilennamen.append(zeile)

                try:
                    csv_exp.append(tmps)
                except ValueError:
                    try:
                        angepasst = np.pad(tmps, [csv_exp.shape[0] - tmps.shape[0], 0], 'constant')
                        csv_exp.append(angepasst)
                    except ValueError:
                        pass

                if len(csv_exp) >= csvlength * 2:
                    outn = str(anzahl_bearbeitet - csvlength) + '-' + str(anzahl_bearbeitet) + "-output.csv"
                    df = pd.DataFrame(csv_exp, index=zeilennamen)
                    if not do_test_mode:
                            try:
                                output = os.path.join(audio_dir + '\\' + 'csv' + '\\' + outn)
                                df.to_csv(output, index=True)
                            except PermissionError:
                                outn = 'io_hand' + outn
                                output = os.path.join(audio_dir + '\\' + 'csv' + '\\' + outn)
                                df.to_csv(output, index=True)
                            df = pd.DataFrame()
                            csv_exp, zeilennamen = [], []

                #########################
                # Setzen vor Rekursion Ende
                #########################

                signal = news.copy()
                iteration_over_file += 1

                #########################
                # achtung while Schleife ende
                #########################

            # break

    if do_test_mode:
        try:
            dft = pd.read_csv('test.csv')
        except:
            dft = pd.DataFrame(csv_exp, index=zeilennamen).transpose(copy=True)
        # Pro Tick:
        dft_nrg_proTick = dft.apply(lambda x: x ** 2 / (2 ** 15))
        statsdf_proTick = dft_nrg_proTick.describe()  # tmp = pd.DataFrame(dft.median())
        meddf_proTick = pd.DataFrame(dft_nrg_proTick.median()).transpose()
        # Pro Sample

        dft_nrg_proS= dft_nrg_proTick.T
        statsdf_proS= statsdf_proTick.T
        meddf_proS= meddf_proTick.T
        timew = np.arange(0, 6720, 1)
        #fig,ax = plt.subplots(2,1)
        #ax[0].plot(yaxis = dft.iloc[:,3], xaxis = timew)
        #fig.show()
        #proTick = pd.concat([statsdf_proTick, meddf_proTick], keys=[ 'stats', 'median'], axis=1, join='outer')
        #proSam = pd.concat([statsdf_proS, meddf_proS], keys=[ 'stats', 'median'], axis=1, join='outer')
        #print(proTick,proSam)
        #for col in dft_nrg_proTick.columns:
        #    dft_nrg_proTick[col].plot()
        #    plt.title(col)
        #    plt.show()
        for col in dft_nrg_proTick.columns:
            fig, ax = plt.subplots(3, 1)
            dft[col].plot(subplots=True,ax=ax[0])

            dft_nrg_proTick[col].plot(subplots=True,ax=ax[1])
            dft_nrg_proTick[col].plot.hist(subplots=True, ax=ax[2])


            fig.tight_layout()
            fig.suptitle(col)
            #fig.supxlabel('Time in Samples')
            ax[0].set_ylabel = 'Signal'
            ax[1].set_ylabel = 'genormte Energie'
            ax[2].set_ylabel = 'Häufigkeiten'
            ax[0].set_xlabel = ''
            ax[1].set_xlabel = 'Zeit in Samples'
            ax[2].set_xlabel = ''
            fig.show()
        # standard_deviations = 3
    # df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]

    plt.clf()
    plt.close()

    print('bye world')


'''
The join() method inserts column(s) from another DataFrame, or Series.
'''