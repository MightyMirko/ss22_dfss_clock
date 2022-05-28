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


def plot_energy(f,fn,filename):
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


def processFolder(audiofile, df, audio_dir, win = 0.005, step = 0.005):
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



def cut_signal(f,fn, s):
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
    do_plot = False # Plotten der Graphen zum Debuggen
    do_test_mode = False # Diverse Beschleuniger
    do_play = False # außer Betrieb

    csv_exp = pd.DataFrame() # Der Haupt-Dataframe
    audio_dir = ''
    if os.path.exists('H:\Messung_BluetoothMikro\Messung 3\Audios'):
        audio_dir = r'H:\Messung_BluetoothMikro\Messung 3\Audios'


    plt.clf()
    apfel = 0
    wavfiles = listdir(audio_dir)
    win, step = 0.005, 0.005
    p = tqdm(total=len(wavfiles), disable=False)
    birne = 0
    # r steht für roh/raw.. Damit lassen sich windows pfade verarbeiten
    # damit sehen wir sofort welche variable wie verarbeitet wird. Erleichtert die Lesbarkeit.
    # ergibt ein Tupel aus Spaltenname und Serie für jede Spalte im Datenrahmen:
    #############
    # Untersuchen des Signals und Fenstern
    ############

    if do_test_mode:
        csv_exp = pandas.read_csv(audio_dir + "csv.csv")
        csv_exp.set_index('Unnamed: 0')
        csv_exp = csv_exp.drop(columns=['Unnamed: 0'])
        apfel = 3

    else:
        while apfel < 2:
            ###############
            # Anlegen der Variablen aus den statistischen Methoden
            ###############
            for audiofile in wavfiles:
                if not '.wav' in audiofile:
                    continue
                if '16_10_21' in audiofile:
                    continue
                txt = ('Dies ist die {}. csvDatei').format(birne)
                txt1 = ('Bearbeiten von {}.').format(audiofile)
                print(txt,'\n', txt1)
                p.update()
                #audiofile = os.path.join(audio_dir + audiofile)
                f, fn, fs, signal = processFolder(audiofile,csv_exp,audio_dir,win,step)
                #tqdm.pandas(csv_exp)
                time = np.linspace(0., len(signal) / float(fs), len(signal))
                duration = len(signal) / float(fs)
                timew = np.arange(0, duration - step, win)

                ###############
                # Nehme den gefunden Index und schneide signal heraus in tmps
                ###############
                olds, tmps, news = cut_signal(f, fn, signal)
                spaltenname = audiofile.strip(audio_dir).strip('.wav') + 'tick' + str(apfel)
                ## noch besser sortieren??
                try:
                    csv_exp[spaltenname] = tmps
                except ValueError:
                    try:
                        angepasst = np.pad(tmps,[csv_exp.shape[0]-tmps.shape[0],0],'constant')
                        csv_exp[spaltenname] = angepasst
                    except ValueError:
                        pass

                if csv_exp.shape[1] >= 1000:
                    output = audio_dir + "output.csv" + str(birne)
                    csv_exp.to_csv(output, index=False)
                    csv_exp = pandas.DataFrame()
                    birne += 1


            if apfel == 0 or apfel == 2:
                pass
                #print('plots off')
                #fn_plot(f,fn,signal)
                #chroma_plot(f,fn)
                #plot_energy(f,fn)
        # do_plot = True
            if apfel == 0:
                global first_nrg
                first_nrg =  f[fn.index('energy'), :]

            if do_plot and apfel == 4 :
                fig2, axs = plt.subplots(4, 1)
                # plt.title('matplotlib.pyplot.figure() Example\n',fontsize=14, fontweight='bold')
                axs[0].set_title(audiofile)
                axs[0].plot(time, olds)
                axs[0].set_xlabel('Samples')
                axs[0].set_ylabel('Original')
                axs[1].plot(tmps)
                axs[1].set_ylabel('Ausgeschnitten')
                axs[2].plot(time, news)
                axs[2].set_ylabel('Old - cutted')
                axs[3].plot(f[fn.index('energy'), :])
                axs[3].plot(first_nrg)

                for i in axs:
                    i.grid()
                fig2.show()

          #  do_plot = False

            #########################
            # Überprüfen ob das geschnittene Signal mehr als starkes Rauschen ist
            #########################
            #energy_gesamt = np.sum(signal**2)/len(signal)
            #tmps_gesamt = np.sum(tmps ** 2)/len(tmps)
            #diff = tmps_gesamt - energy_gesamt
           # csv_exp.describe()

            #txt = "{}. Durchgang:\n" +  "Energie gesamt:\t {}\n" + "Tickenergie gesamt:\t {}\n" +"Differenz:\t {}\n"
            #print(txt.format(apfel+1, energy_gesamt, tmps_gesamt, diff))

            #########################
            # Setzen vor Rekursion Ende
            #########################

            signal = news.copy()
            apfel += 1

            #########################
            # achtung while Schleife ende
            #########################

        #break


    #standard_deviations = 3
    #df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations).all(axis=1)]

    plt.clf()
    plt.close()
    p.close()

    print('bye world')
