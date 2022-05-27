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


from tqdm.notebook import tqdm


def getWavFromFolder(wavdir, do_test=False):
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
        t.update()
        t.set_postfix(dir=wavdir)
    t.close()

    return wavfiles, df


def cut(s, fs, win, step):
    [f, fn] = aF.feature_extraction(s, fs, int(fs * win),
                                    int(fs * step), True)
    pass


if __name__ == "__main__":

    print('hello World')
    do_plot = False
    do_test_mode = True
    do_play = False

    audio_dir = (r'data/')
    # r steht für roh/raw.. Damit lassen sich windows pfade verarbeiten
    # damit sehen wir sofort welche variable wie verarbeitet wird. Erleichtert die Lesbarkeit.
    plt.clf()

    # ergibt ein Tupel aus Spaltenname und Serie für jede Spalte im Datenrahmen:

    window, step = 0.03, 0.03
    wavfiles = listdir(r'data')
    #############
    # Untersuchen des Signals und Fenstern
    ############

    show_progress = True
    t = tqdm(total=1, unit="file", disable=not show_progress)
    win, step = 0.01, 0.01

    for audiofile in wavfiles:
        if not '.wav' in audiofile:
            continue
        if '16_10_21' in audiofile:
            continue

        audiofile = os.path.join(audio_dir + audiofile)
        # print(audiofile)
        fs, s = aIO.read_audio_file(audiofile)
        duration = len(s) / float(fs)
        time = np.arange(0, duration - step, win)
        timew = np.linspace(0., duration, len(s))
        # extract short-term features using a 50msec non-overlapping windows

        [f, fn] = aF.feature_extraction(s, fs, int(fs * win), int(fs * step), True)

        energy = f[fn.index('energy'), :]  # Alle Daten aus der index nummer 1 ziehen..
        (spec) = aF.spectrogram(s, fs, int(fs * win), int(fs * step), plot=False, show_progress=True)
        (chroma) = aF.chromagram(s, fs, int(fs * win), int(fs * step), plot=False, show_progress=True)
        d_energy = f[fn.index('delta energy'), :]
        fig1, axs = plt.subplots(4, 1)
        dd_energy = np.diff(d_energy)
        ddd_energy = np.diff(dd_energy)

        axs[0].plot(energy)
        axs[0].set_xlabel('Frame Number')
        axs[0].set_ylabel(fn[1])

        axs[1].plot(d_energy)
        axs[1].set_xlabel('Frame Number')
        axs[1].set_ylabel(fn[1])

        axs[2].plot(dd_energy)
        axs[2].set_xlabel('Frame Number')
        axs[2].set_ylabel(fn[1])

        axs[3].plot(ddd_energy)
        axs[3].set_xlabel('Frame Number')
        axs[3].set_ylabel(fn[1])



        maxenergy = d_energy.max()
        indexofpeak = d_energy.argmax()  # Wo ist das Maximum
        print(indexofpeak)
        ## Nehme den gefunden Index und schneide signal heraus in tmps
        peak_in_ms = win * indexofpeak
        back_in_ms = peak_in_ms - 0.04
        adv_in_ms = peak_in_ms + 0.2

        zero = np.zeros(96000)
        tmps = s[int(back_in_ms * fs*duration):int(fs * adv_in_ms*duration)]

        fig2, axs1 = plt.subplots(3, 1)
        axs1[0].plot(timew, s)
        axs1[0].set_xlabel('Samples')
        axs1[0].set_ylabel('Original')
        axs1[0].grid()
        axs1[1].plot(tmps)
        axs1[1].set_ylabel('Cutted')

        news = s[tmps.shape[0] + 1:]
        axs1[2].plot(timew[timew.shape[0] - news.shape[0]:], news)
        axs1[2].set_ylabel('Old - cutted')

        plt.show()

        # cut(s=news,fs=fs,win=win,step=step)

        break

    t.set_postfix(dir=audio_dir)
    t.close()

    if do_plot:
        fig1, axs = plt.subplots(2, 1)
        axs[0].plot(f[0, :])
        axs[0].set_xlabel('Frame no')
        axs[0].set_ylabel(fn[0])
        axs[1].plot(f[1, :])
        axs[1].set_xlabel('Frame no')
        axs[1].set_ylabel(fn[1])
        plt.show()

        import plotly.graph_objects as go
        import plotly.io as pio

        # pio.renderers.default = 'png'
        pio.renderers.render_on_display = True
        fig = go.Figure(
            data=[go.Bar(y=[2, 1, 3])],
            layout_title_text="A Figure Displayed with fig.show()"
        )
        fig.show();
    else:
        print('Keine Plots erwünscht')

    plt.clf()
    plt.close()

    print('bye world')
