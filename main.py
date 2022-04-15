# @Author Mirko Matosin
# @Date 14.02.2022
import numpy as np
import pandas as pd
import scipy.io
import math
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
#plt.rcParams['figure.figsize'] = (12, 7)
from scipy.io import wavfile
from os.path import dirname, join as pjoin
from os import listdir

'''
@Source
https://stackoverflow.com/questions/2060628/reading-wav-files-in-python
https://www.programcreek.com/python/example/93227/scipy.io.wavfile.read
https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
https://klyshko.github.io/teaching/2019-02-22-teaching

'''

'''
Digital sound

When you hear a sound your ear’s membrane oscillates because the density and pressure of the air in close proximity 
to the ear oscillate as well. Thus, sound recordings contain the relative signal of these oscilations. Digital audio is 
sound that has been recorded in, or converted into, digital form. In digital audio, the sound wave of the audio signal 
is encoded as numerical samples in continuous sequence. For example, in CD (or WAV) audio, samples are taken 44100 times 
per second each with 16 bit sample depth, i.e. there are 2^16 = 65536 possible values of the signal: from -32768 to 32767. 
For the example below, a sound wave, in red, represented digitally, in blue (after sampling and 4-bit quantization).

'''

def plotSounds(data, filename, samplerate=96000, xlab = 'Time [s]', ylab='Amplitude'):
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

    audio_dir = (r'rohdaten/') # r steht für roh/raw.. Damit lassen sich windows pfade verarbeiten
    files, df = getWavFromFolder(wavdir=audio_dir) # ich mag explizite Programmierung. Also wavdir=...,
    # damit sehen wir sofort welche variable wie verarbeitet wird. Erleichtert die Lesbarkeit.


    # ergibt ein Tupel aus Spaltenname und Serie für jede Spalte im Datenrahmen:
    for (columnName, columnData) in df.iteritems():
        plotSounds(data=columnData, filename=columnName)
