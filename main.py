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


def kohaerenz(axs):
    '''
    Testphase... Weiß nicht ob notwendig.. Mir fehlt da noch die Theorie
    https://de.wikipedia.org/wiki/Koh%C3%A4renz_(Signalanalyse)
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/cohere.html#sphx-glr-gallery-lines-bars-and-markers-cohere-py
    :param axs:
    :return:
    '''

    s1 = filtered
    s2 = moved_average_data
    dt = 1 / 96000
    print(type(s1), '\n', type(s2))
    print(s1.shape, '\n', s2.shape)

    axs[2].plot(time, s1, time, s1)
    axs[2].set_xlim(0, 2)
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('s1 and s2')
    axs[2].grid(True)

    cxy, f = axs[3].cohere(s1, s1, 256, 1. / dt)
    # axs[3].set_ylabel('coherence')
    out = axs


def fft_test(ax2d, data, sample_rate=48000, do_plot=False, filename='test'):
    '''

    :param ax2d: Matplotlib artist als mehrdimensionale Variable
    :param data: Die Daten für die FFT
    :param sample_rate: Standard ist 48kHz - kann geändert werden (ungetestet)
    :param do_plot: Wenn die Funktion gleich auch plotten soll? Achtung: Rechner stürzt bei vielen Figures ab
    :param filename: Aktueller Dateiname
    :return: Fourierdaten und das Matplotlibhandle zur weiteren Kontrolle der Plots.
    '''
    signal_length = data.shape[0]/sample_rate
    num = signal_length * float(sample_rate)
    # sample spacing
    x = np.linspace(start=0.0, stop=signal_length, num=int(signal_length * sample_rate), endpoint=False)
    if not 'ndarray' in type(data):
        y = data.to_numpy()
    else:
        print(type(data))

    yf = fft(y)
    fourier_abtastzeit = 1  / sample_rate
    xf = fftfreq(sample_rate, fourier_abtastzeit)[:sample_rate // 2]

    out = ''
    if do_plot:
        ax = ax2d[0]
        ax1 = ax2d[1]
        apfel = np.abs(yf[0:sample_rate // 2])
        # Normalisierung?
        normale = (2.0 / sample_rate * apfel)
        out = ax.plot(xf, normale)
        ax.set_title('Fourier Transformation in linear und log bis 10k')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude linear')
        ax.set_xlim(left=0, right=1e4)
        ax.grid()

        out = ax1.plot(xf, normale)
        # plt.legend(loc='upper right')
        ax1.set_xlabel('Dateiname:\n' + filename)
        ax1.set_xlim(left=0, right=1e4)
        ax1.set_yscale('log')
        ax1.set_ylabel('Amplitude in dB')

        ax1.grid()

    fourier = [yf, xf]
    return fourier, out


def mov_avg(data_filtered, timevec, do_plot=False):
    '''
    Moving Average-Filter.. Achtung!!! Anzahl an Sampels wird um k-1 verringert
    :param data_filtered:
    :param timevec:
    :param do_plot:
    :return:
    '''
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
    '''
       Butterworth-Hochpassfilter
        wn = 150 rad/s bestes Ergebnis für Signal
        wn = 500 rad/s bestes Ergebnis für Rauschen
    :param data: Data zum buttern als pandas.series/numpy.array
    :param timevec: x-achse
    :param filename: Dateiname
    :param do_plot: Wenn die Funktion gleich auch plotten soll? Achtung: Rechner stürzt bei vielen Figures ab
    :param samplerate:
    :param omega: Wie soll gebuttert werden?
    :return:
    '''


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
            filename  = re.sub(r'.wav$', '', file)
            if do_test and not 'test' in filename:
                continue
            else:
                pfad = pjoin(wavdir, file)
                wavfiles.append(pfad)
                filenames.append(filename)
                samplerate, data = wavfile.read(filename=pfad)
                df[filename] = data
                break
    return wavfiles, df


from playsound import playsound
from multiprocessing import Process


if __name__ == "__main__":

    print('hello World')
    do_plot = True
    do_test_mode = True
    do_play = False

    audio_dir = (r'rohdaten/')  # r steht für roh/raw.. Damit lassen sich windows pfade verarbeiten
    files, df = getWavFromFolder(wavdir=audio_dir, do_test=do_test_mode)  # ich mag explizite Programmierung. Also wavdir=...,
    # damit sehen wir sofort welche variable wie verarbeitet wird. Erleichtert die Lesbarkeit.
    samplerate = 48000
    plt.clf()

    # ergibt ein Tupel aus Spaltenname und Serie für jede Spalte im Datenrahmen:
    for (columnName, columnData) in df.iteritems():
        if do_test_mode and not 'test' in columnName:
            continue
        if do_play:
            p = Process(target=playsound, args=([audio_dir + 'a'+columnName + '.wav'])                        )
            p.start()

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
                       log=True, title='Original-Signal', ylab='Amplitude in dB')

            plotSounds(ax2, data=columnData, filename=columnName, abszisse=time,
                       log=False, title='')

            fig2, axs = plt.subplots(4, 1, figsize=(12, 8))

            plotSounds(axs[0], data=filtered, filename=columnName, abszisse=time,
                       title='Gebuttert mit $\omega$= %i' % omega_butter, log=False)

            moved_average_data = mov_avg(data_filtered=columnData, timevec=time, do_plot=False)
            moved_average_data = moved_average_data.values
            tmp_time = columnData.shape[0] - 4
            plotSounds(axs[1], data=moved_average_data, filename=columnName, abszisse=time[0:tmp_time],
                       title='Moving AVG', log=False)

            fig3, axs = plt.subplots(2, 1, figsize=(12, 8))

            moved_average_data = mov_avg(data_filtered=columnData, timevec=time, do_plot=False)
            moved_average_data = moved_average_data.values
            filtered = buttern(data=moved_average_data, filename=columnName, timevec=time,
                               omega=omega_butter)


            plotSounds(axs[0], data=filtered, filename=columnName, abszisse=time[0:tmp_time],
                       title='Gebuttert mit $\omega$= %i' % omega_butter, log=False)

            plotSounds(axs[1], data=moved_average_data, filename=columnName,abszisse=time[0:tmp_time],
                       title='Moving AVG', log=False)

            # fig3.show()

            fig4, axs = plt.subplots(2, 1, figsize=(12, 8))
            fft = fft_test(ax2d=axs, data=columnData, do_plot=True, filename=columnName)
            fig5, axs = plt.subplots(2, 1, figsize=(12, 8))

            fft_butter = fft_test(ax2d=axs, data=filtered, do_plot=True, filename=columnName)
            # print(fft[0])

            if do_plot:
                fig1.tight_layout()
                fig2.tight_layout()
                fig3.tight_layout()
                fig4.tight_layout()
                fig1.show()
                fig2.show()
                fig3.show()
                fig4.show()
                plt.close(fig4)
                plt.close(fig3)
                plt.close(fig2)
                plt.close(fig1)
            else:
                print('Keine Plots erwünscht')
            break
            if do_play:
                p.join()

    print('bye world')
    plt.clf()

    plt.close()
