# @Author Mirko Matosin
# @Date 14.02.2022

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.dpi'] = 150
# plt.rcParams['interactive'] = True
plt.rcParams['figure.figsize'] = (12, 7)
from scipy.io import wavfile
from os.path import join as pjoin
from os import listdir
from scipy.fft import fft, fftfreq
from scipy.signal import butter
from scipy.signal import sosfilt


def fft_test(data, timevec, sample_rate = 48000, fourier_abtastzeit=1e-3, do_plot=False):
    signal_length = len(timevec)
    # sample spacing
    x = np.linspace(start=0.0, stop=2, num=signal_length * sample_rate, endpoint=False)
    y = columnData.to_numpy()

    yf = fft(y)
    xf = fftfreq(sample_rate, fourier_abtastzeit)[:sample_rate // 2]
    print(xf)

    apfel = np.abs(yf[0:sample_rate // 2])
    # Normalisierung?
    normale = (2.0 / sample_rate * apfel)
    if do_plot:
        plt.plot(xf, normale )
        plt.xlim(xmin=0, xmax=2e4)
        # plt.legend(loc='upper right')
        plt.xlabel('Dateiname:\n' + columnName)
        plt.grid()
        plt.show()

    out = [yf,xf]
    return out

    # Autokorrelation: Verstärkung des Signals?
    # autocorr = signal.convolve(y, y)
    # plt.plot(autocorr)
    # plt.show()


def mov_avg(data_filtered, timevec, do_plot=False):
    ''' Moving Average-Filter
        Achtung!!! Anzahl an Sampels wird um k-1 verringert '''

    t = timevec

    # Fenstergröße k festlegen (nur ungerade k erlaubt!!!)
    window_size = 5
    # übergabe des Signals und Konvertierung in pandas Series
    numbers_series = pd.Series(data_filtered)
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

    return moving_averages


def buttern(data, timevec, filename='N/A', doplot=False, samplerate=48000, omega=500):
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
    time_axis = abszisse
    out = ax.plot(time_axis, data, label=filename, **param_dict)
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
    plt.clf()
    # ergibt ein Tupel aus Spaltenname und Serie für jede Spalte im Datenrahmen:
    for (columnName, columnData) in df.iteritems():
        if not '26' in columnName:
            continue
        else:
            # plotSounds(data=columnData, filename=columnName)
            print(columnData.max())
            # Anlegen der Variablen
            length = columnData.shape[0] / samplerate
            time = np.linspace(0., length, columnData.shape[0])
            omega_butter = 400
            # Erst mal buttern mit Original dan filtern mit original
            filtered = buttern(data=columnData, filename=columnName, timevec=time,
                               omega=omega_butter)

            # Plot der Original Datei
            fig1, axs = plt.subplots(2, 1, figsize=(12, 8))  # , sharex=True)
            ax1, ax2 = axs

            # Logaritmische Darstellung
            plotSounds(ax1, data=columnData, filename=columnName, abszisse=time,
                       log=True, title='Original', ylab='Amplitude in dB')

            plotSounds(ax2, data=columnData, filename=columnName, abszisse=time,
                       log=False, title='')
            fig2, axs = plt.subplots(2, 1, figsize=(12, 8))

            plotSounds(axs[0], data=filtered, filename=columnName, abszisse=time,
                       title='Gebuttert mit $\omega$= %i' % omega_butter, log=False)

            moved_average_data = mov_avg(data_filtered=columnData, timevec=time, do_plot=False)
            moved_average_data = moved_average_data.values

            plotSounds(axs[1], data=moved_average_data, filename=columnName, abszisse=time[0:95996],
                       title='Moving AVG', log=False)

            # Buttern, dann filtern
            fig1.show()
            fig2.show()

            fig3, axs = plt.subplots(2, 1, figsize=(12, 8))

            moved_average_data = mov_avg(data_filtered=columnData, timevec=time, do_plot=False)
            moved_average_data = moved_average_data.values
            filtered = buttern(data=moved_average_data, filename=columnName, timevec=time,
                               omega=omega_butter)
            plotSounds(axs[0], data=filtered, filename=columnName, abszisse=time[0:95996],
                       title='Gebuttert mit $\omega$= %i' % omega_butter, log=False)

            plotSounds(axs[1], data=moved_average_data, filename=columnName, abszisse=time[0:95996],
                       title='Moving AVG', log=False)

            fig3.show()
            plt.close(fig3)
            plt.close(fig2)
            plt.close(fig1)

            fig4, axs = plt.subplots(2, 1, figsize=(12, 8))
            fft_test(data=columnData, timevec=time,fourier_abtastzeit=1/10/48e3, do_plot=True)

            break

    print('bye world')
    plt.clf()

    plt.close()
