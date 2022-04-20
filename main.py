# @Author Mirko Matosin
# @Date 14.02.2022
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.dpi'] = 150
# plt.rcParams['figure.figsize'] = (12, 7)
from scipy.io import wavfile
from scipy import signal

from os.path import join as pjoin
from os import listdir

'''
@Source
https://stackoverflow.com/questions/2060628/reading-wav-files-in-python
https://www.programcreek.com/python/example/93227/scipy.io.wavfile.read
https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
https://klyshko.github.io/teaching/2019-02-22-teaching

Digital sound

When you hear a sound your ear’s membrane oscillates because the density and pressure of the air in close proximity 
to the ear oscillate as well. Thus, sound recordings contain the relative signal of these oscilations. Digital audio is 
sound that has been recorded in, or converted into, digital form. In digital audio, the sound wave of the audio signal 
is encoded as numerical samples in continuous sequence. For example, in CD (or WAV) audio, samples are taken 44100 times 
per second each with 16 bit sample depth, i.e. there are 2^16 = 65536 possible values of the signal: from -32768 to 32767. 
For the example below, a sound wave, in red, represented digitally, in blue (after sampling and 4-bit quantization).



2f/2s Signal    ?????

To Read:
https://dewesoft.com/daq/guide-to-fft-analysis

'''


def plotSounds(data, filename, samplerate=48000, xlab='Time [s]', ylab='Amplitude'):
    """
    Plottet eine Spalte oder ein Vektor in Timedomain

    :param data: hier kommt eine Series (Spalte Dataframe) hinein.
    :param filename: Legendenname
    :param samplerate: 96000 sind Standard. Wird für die Zeitachse genommen
    :param xlab: Label auf Abszisse
    :param ylab: Label auf Ordinate
    :return:
    """
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data, label=filename)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xscale('linear')
    plt.yscale('log')
    # plt.xlim(xmax=1.31, xmin=1)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


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
    from scipy.fft import fft, fftfreq

    # ergibt ein Tupel aus Spaltenname und Serie für jede Spalte im Datenrahmen:
    for (columnName, columnData) in df.iteritems():
        if not '26' in columnName:
            continue
        else:
            plotSounds(data=columnData, filename=columnName)
            # yf = fft.fft(columnData)
            # xf = fft.fftfreq(yf)
            # Number of sample points

            N = 44000
            # sample spacing
            T = 1.0 / N / 10
            print(T)
            x = np.linspace(start=0.0, stop=20e3, num=N, endpoint=False)
            y = columnData.to_numpy()
            yf = fft(y)
            print(yf)
            print(np.abs(yf))
            xf = fftfreq(N, T)[:N // 2]

            plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
            plt.xlim(xmax=4e3, xmin=0)
            # plt.legend(loc='upper right')
            plt.xlabel(columnName)
            plt.grid()

            plt.show()

            # Autokorrelation: Verstärkung des Signals?
            autocorr = signal.convolve(y,y)
            plt.plot(autocorr)
            plt.show()
            break

'''
FFT ist nun also die Summe aller sinus Funktionen normalisiert auf die Menge aller Datenpunkte N. 

$A(f_k) = \frac{1}{N} \sum_{n=0}^{N-1} a(t_n) e^{-i \frac{2 \pi kn}{N}}$

'''

'''
Das Ziel ist nun die wesentlichen Signalanteile in der Wav zu finden. Dies erfordert das Nutzen der FFT. Die FFT 
wird das Signal in seine spektralen Bestandteile zerlegen. 

Nullhypothese:
Hypothese ist das ein lautes Grundrauschen bis ca. 800 Hz enthalten ist. Das Ticken nach unten (0-30s) selbst sollte im 
Frequenzbereich um 1k Hz liegen. Dies können wir nutzen um das Ticken zu isolieren und zu verstärken

Alternativhypothese: 
Es gibt keinen Unterschied von hoch/runter.. Die Eigenschwingung \omega_E,R ist stets konstant.

Vorgehen:
Der erste Schritt besteht darin, die Eingangszeitdaten in FFT-Zeitblöcke zu zerlegen. Die Eingangszeitdaten können rohe 
Sensorsignale oder vorverarbeitete (z. B. gefilterte) Signale sein. Jeder Zeitblock hat eine Zeitdauer T, die sich auf 
die spektrale Auflösung der erzeugten Spektren bezieht. Die Zeitblöcke können so konfiguriert werden, dass eine 
Fensterfunktion angewandt wird und ein Überlappungssatz entsteht. 
Anschließend werden die FFT-Zeitblöcke mit Hilfe des FFT-Algorithmus vom Zeitbereich in den Frequenzbereich transformiert. 
Jeder Zeitblock ergibt ein momentanes komplexes FFT-Spektrum. 
Die momentanen komplexen FFT-Spektren werden zur Berechnung der momentanen Leistungsspektren verwendet. Die Leistungsspektren 
werden über eine bestimmte Anzahl von Spektren oder eine bestimmte Zeitdauer gemittelt. Leistungsspektren haben reelle 
Werte und beziehen sich auf ein Eingangssignal. Kreuzleistungsspektren haben komplexe Werte und beziehen sich auf zwei 
Eingangssignale.


'''
