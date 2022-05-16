# @Author Mirko Matosin
# @Date 14.02.2022

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.dpi'] = 150
# plt.rcParams['figure.figsize'] = (12, 7)
from scipy.io import wavfile
from os.path import join as pjoin
from os import listdir
from scipy.fft import fft, fftfreq
from scipy.signal import butter
from scipy.signal import sosfilt


def fft_test():
    sample_rate = 600
    signal_length = 2
    # sample spacing
    abtastzeit = 1.0 / (9 * sample_rate)

    #            x = np.linspace(0.0, sample_rate * abtastzeit, sample_rate, endpoint=False)
    #           y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)

    x = np.linspace(start=0.0, stop=2, num=signal_length * sample_rate, endpoint=False)
    y = columnData.to_numpy()
    yf = fft(y)
    xf = fftfreq(sample_rate, abtastzeit)[:sample_rate // 2]

    apfel = np.abs(yf[0:sample_rate // 2])
    print(apfel)
    plt.plot(xf, 2.0 / sample_rate * np.abs(yf[0:sample_rate // 2]))
    plt.xlim(xmin=0, xmax=200)
    # plt.legend(loc='upper right')
    plt.xlabel('Dateiname:\n' + columnName)
    plt.grid()
    plt.show()

    # Autokorrelation: Verstärkung des Signals?
    #autocorr = signal.convolve(y, y)
    # plt.plot(autocorr)
    #plt.show()


def mov_avg(data_filtered, do_plot=False):
    ''' Moving Average-Filter
        Achtung!!! Anzahl an Sampels wird um k-1 verringert '''
    filtered = data_filtered
    t = np.linspace(0.0, 2.0, num=96000)

    # Fenstergröße k festlegen (nur ungerade k erlaubt!!!)
    window_size = 5
    # übergabe des Signals und convertierung in pandas Series
    numbers_series = pd.Series(filtered)
    # Auschneidern der Beobachtungsreihe entsprechend der Fenstergröße
    windows = numbers_series.rolling(window_size)
    # Mittelwert jedes Ausschnittes bilden
    moving_averages = windows.mean()
    # NaN entfernen
    moving_averages = moving_averages[window_size - 1:]
    if do_plot:
        # plot gefiltertes Signal gesamt
        plt.figure()
        plt.plot(t[(window_size // 2):-(window_size // 2)], moving_averages)
        plt.grid(True)
        plt.xlabel('Zeit in s')
        plt.ylabel('Amplitude')
        plt.title('Moving Average gefiltert')
        plt.show()

        # plot Ticken
        plt.figure()
        # xmin, xmax, ymin, ymax = 1.2, 1.3, -0.003, 0.003
        # plt.axis([xmin, xmax, ymin, ymax])
        plt.xlim(1.2, 1.3)
        plt.plot(t[(window_size // 2):-(window_size // 2)], moving_averages)
        plt.grid(True)
        plt.xlabel('Zeit in s')
        plt.ylabel('Amplitude')
        plt.title('Moving Average gefiltert')
        plt.show()


def buttern(data, timevec,filename='N/A', doplot=False, samplerate=48000, omega=500):
    ''' Butterworth-Hochpassfilter
        wn = 150 rad/s bestes Ergebnis für Signal
        wn = 500 rad/s bestes Ergebnis für Rauschen '''

    wn = omega
    # warum braucht es eine so große Ordnung?
    sos = butter(10, wn, btype='hp', fs=samplerate, output='sos')
    filtered = sosfilt(sos, data)

    if doplot:
        # plot gefiltertes Signal gesamt
        plt.figure()
        plt.plot(timevec, filtered)
        plt.grid(True)
        plt.xlabel('Zeit in s')
        plt.ylabel('Amplitude')
        plt.title('HP gefiltert mit wn = %i' % wn)
        plt.show()

    return filtered


def plotSounds(ax,data, abszisse, filename='N/A',samplerate=48000, log=True,
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


    time_axis = abszisse
    plt.plot(time_axis, data, label=filename)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xscale('linear')
    if log:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    # plt.xlim(xmax=1.31, xmin=1)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()
    out = ax
    return out

def getWavFromFolder(wavdir):
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
            filename = file.strip(".wav")
            pfad = pjoin(wavdir, file)
            wavfiles.append(pfad)
            filenames.append(filename)
            samplerate, data = wavfile.read(filename=pfad)
            df[filename] = data
    return wavfiles, df


if __name__ == "__main__":
    #
    print('hello World')

    audio_dir = (r'rohdaten/')  # r steht für roh/raw.. Damit lassen sich windows pfade verarbeiten
    files, df = getWavFromFolder(wavdir=audio_dir)  # ich mag explizite Programmierung. Also wavdir=...,
    # damit sehen wir sofort welche variable wie verarbeitet wird. Erleichtert die Lesbarkeit.
    samplerate = 48000

    # ergibt ein Tupel aus Spaltenname und Serie für jede Spalte im Datenrahmen:
    for (columnName, columnData) in df.iteritems():
        if not '26' in columnName:
            continue
        else:
            # plotSounds(data=columnData, filename=columnName)

            length = columnData.shape[0] / samplerate
            time = np.linspace(0., length, columnData.shape[0])
            omega_butter = 400
            fig1, axs = plt.subplots(2, 1)#, sharex=True)
            ax1,ax2 = axs
            plotSounds(data=columnData, filename=columnName, abszisse=time,
                       figurehandle=fig1)
            plotSounds(data=columnData, filename=columnName, abszisse=time,
                       figurehandle = fig1, log=False)


            filtered = buttern(data=columnData, filename=columnName,timevec=time, omega=omega_butter)
            fig3, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            plotSounds(data=filtered, filename=columnName, abszisse=time,
                       title='Gebuttert mit $\omega$= %i'%omega_butter,
                       figurehandle = fig3, log=False)




            break

    print('bye world')