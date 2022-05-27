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

import tqdm
import os



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


from tqdm.notebook import trange, tqdm


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
                #df['samplerate'] = samplerate
                #break
        t.update()
        t.set_postfix(dir=wavdir)
    t.close()

    return wavfiles, df


from playsound import playsound
from multiprocessing import Process
import pandas
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import MidTermFeatures as mF
import numpy as np
import plotly.graph_objs as go
import plotly
import IPython

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
    win, step = 0.03, 0.03

    for audiofile in wavfiles:
        if not '.wav' in audiofile:
            continue
        if '16_10_21' in audiofile:
                continue

        audiofile = os.path.join(audio_dir + audiofile)
        print(audiofile)
        fs, s = aIO.read_audio_file(audiofile)
        duration = len(s) / float(fs)
        time = np.arange(0, duration - step, win)

        # extract short-term features using a 50msec non-overlapping windows

        [f, fn] = aF.feature_extraction(s, fs, int(fs * win),
                                        int(fs * step))

        if feature_frame.empty:
            feature_frame = pandas.DataFrame(f, index=fn)
        #feature_frame.append(pandas.DataFrame(f, index=fn))
        t.update()
        print(feature_frame.axes)

        feature_frame.plot(y="energy" )

        break






    t.set_postfix(dir=audio_dir)
    t.close()




    if do_plot:
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
