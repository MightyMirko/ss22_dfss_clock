# @Author Mirko Matosin und Alexander Wünstel
# @Date 14.02.2022
#


# fft anschauen. clustern, standardisieren, normalisieren
# isolation forest
# db scan

import os
from datetime import datetime
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
from scipy.io import wavfile
from scipy.stats import t

###################
# Anlegen globaler Var
###################
plt.rcParams['figure.dpi'] = 150
# plt.rcParams['interactive'] = True
plt.rcParams['figure.figsize'] = (12, 9)


class ZeigerWinkel(Enum):
    """
    """
    HOCH = 1
    RUNTER = 2
    UNSICHER1 = 3
    UNSICHER2 = 4


def plotsounds(ax, data, abszisse, param_dict=None, filename='N/A', samplerate=48000, log=True,
               xlab='Time [signal]', ylab='Amplitude', title='Ticken im Original'):
    """
    Plottet eine Spalte oder ein Vektor in Timedomain

    :param title:
    :param log:
    :param param_dict:
    :param abszisse:
    :param ax:
    :param data: hier kommt eine Series (Spalte Dataframe) hinein.
    :param filename: Legendenname
    :param samplerate: 96000 sind Standard. Wird für die Zeitachse genommen
    :param xlab: Label auf Abszisse
    :param ylab: Label auf Ordinate
    :return:
    """
    if param_dict is None:
        param_dict = {}
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


def process_folder(audiofile, audio_dir, win=0.005, step=0.005):
    """

    :param audiofile: muss sanitized sein
    :param audio_dir: muss sanitized sein
    :param win:
    :param step:
    :return:
    """

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


def get_energies(d, wavfiles):
    """
    Diese Funktion bestimmt die Schwellenwerte aller Signale und erstellt ein Dataframe

    :param d: wo soll der suchen
    :param wavfiles: übergabe der zu siebenden Daten
    :return:
    """
    # try:
    #     file_list = wavfiles[100:9000]
    # except:
    file_list = wavfiles

    number_of_files = len(file_list)
    print(number_of_files)
    energie = np.zeros(number_of_files)

    # Energie bestimmen absolut
    for idx in range(len(file_list)):
        # einlesen
        fn = file_list[idx]
        samplerate, data = wavfile.read(d + ('\\') + file_list[idx])
        # Berechnung
        signal_2 = data ** 2
        energie[idx] = signal_2.sum()
        if idx % int(number_of_files / 10) == 0:
            print(str(idx) + " von " + str(number_of_files))

    df = pandas.DataFrame(data=energie, index=file_list, columns={'GesamtEnergie'})
    return df


def prognose(data, gamma=0.95, bereich='beide'):
    """
    Berechnung des Prognosebereichs
    :param art:
    :param gamma:
    :param data:
    :return:
    """

    N = np.size(data)
    data_mean = np.mean(data)
    data_var = np.var(data, ddof=1)
    data_std = np.std(data, ddof=1)

    Einheit_unit = 'W'
    if bereich == 'rechts':
        c = t.ppf((gamma), N - 1)
        x_prog_max = data_mean + c * data_std * np.sqrt(1 + 1 / N)
        print('Prognosewert: x', '>=\t', round(x_prog_max, 9), Einheit_unit)
        return x_prog_max
    if bereich == 'beide':
        c1 = t.ppf((1 - gamma) / 2, N - 1)
        c2 = t.ppf((1 + gamma) / 2, N - 1)
        x_prog_min = data_mean + c1 * data_std * np.sqrt(1 + 1 / N)
        x_prog_max = data_mean + c2 * data_std * np.sqrt(1 + 1 / N)
        print('Prognosewert:', round(x_prog_min, 9), Einheit_unit, '< x <=', round(x_prog_max, 9), Einheit_unit)
        return x_prog_min, x_prog_max
    if bereich == 'links':
        c1 = t.ppf((1 - gamma), N - 1)
        x_prog_min = data_mean + c1 * data_std * np.sqrt(1 + 1 / N)
        print('Prognosewert: x', '<=\t', round(x_prog_min, 9), Einheit_unit)
        return x_prog_min
    else:
        return 0
    # Unbekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG


def getsecs(fromdf):
    '''

    :param fromdf:
    :return:
    '''
    form = '%Y%m%d_%H_%M_%S'
    dt_list = []
    for row_index, row in fromdf.iterrows():
        # dat = os.path.splitext(row_index)[0]
        dat = os.path.basename(row_index).lstrip()
        dat = dat.split(".")[0]
        date_string = datetime.strptime(dat, form)
        sekzeiger = date_string.second
        status = "'"
        try:
            if not sekzeiger <= 0 and sekzeiger < 3:
                status = ZeigerWinkel(3)
            if not sekzeiger <= 3 and sekzeiger < 27:
                status = ZeigerWinkel(2)
            if not sekzeiger <= 27 and sekzeiger < 33:
                status = ZeigerWinkel(4)
            if not sekzeiger <= 33 and sekzeiger < 57:
                status = ZeigerWinkel(2)
            if not sekzeiger <= 57 and sekzeiger < 0:
                status = ZeigerWinkel(3)
        except:
            pass
        dt_list.append((date_string, status))

    dtdf = pandas.DataFrame(dt_list, index=fromdf.index, columns=['rectime', 'ZeigerWinkel'])
    return dtdf


if __name__ == "__main__":

    ################################################
    # Anlegen der Kontrollvariablen
    ################################################
    print('hello World')
    do_plot = False  # Plotten der Graphen zum Debuggen
    do_test_mode = True  # Diverse Beschleuniger
    do_play = False  # außer Betrieb

    plt.clf()
    ################################################
    # Arbeits Variablen
    ################################################
    csv_exp = []  # Der exportierende
    zeilennamen = []
    errata = []
    audio_dir = r'H:\Messung_BluetoothMikro\Messung 3\Audios'

    ################################################
    # Überprüfung
    ################################################
    if do_test_mode:
        audio_dir = r'data'
    else:
        if not os.path.exists(audio_dir):
            raise IOError
        else:
            audio_dir = audio_dir
    ################################################
    # Anlegen der Variablen und Wav-Liste
    ################################################

    duration = 0
    timew = 0
    iteration_over_file = 0
    anzahl_bearbeitet = 0
    wavfiles = []
    ################################################
    # Desinfizieren der Wavliste
    ################################################

    with os.scandir(audio_dir) as it:
        for entry in it:
            if entry.name.endswith('.wav') and entry.is_file():
                wavfiles.append(entry.name)

    # wavfiles = wavfiles[:400]
    anzahl = len(wavfiles)
    anzahlnochnicht = anzahl
    csvlength = 300  # Achtung es werden die Zeilen 2x gezählt -> 50 dateien = 100 zeilen

    ################################################
    # Schätzung unbekannter Parameter über die t-verteilte Grundgesamtheit
    ################################################
    # gesamtenergie hat einen Median und anhand dessen kann ich doch auch bereits Ausreisser erkennen?
    if not do_test_mode:
        gesamtenergien = get_energies(audio_dir, wavfiles)
        sieb_energien = gesamtenergien.drop_duplicates()
    else:
        sieb_energien = pandas.read_csv('gesamtdaten_energien.csv', index_col=0)
    ################################################
    # Plotte die Grundgesamtheit und dann jedes mal wieder nach dem Sieben mittels prognose
    ################################################
    fig, ax = plt.subplots(3, 2, sharex='all')  # , sharey='all')
    pwr = sieb_energien['GesamtEnergie'].copy()

    qu = sieb_energien['GesamtEnergie'].quantile(0.1)
    qo = sieb_energien['GesamtEnergie'].quantile(0.999)
    v=pwr.shape[0]
    minimalschwellle = qu  # eigentlich lieber mit der fft für frequenzen kleiner 300 Hz
    maximalschwelle = qo# nur einen rauskicken
    for x in pwr.index:
        if pwr.loc[x] > minimalschwellle and pwr.loc[x] < maximalschwelle:  # or pwr.loc[x] < progmin:
            pass
        else:
            pwr.drop(x, inplace=True)
    rausgeworfen = v - pwr.shape[0]
    if rausgeworfen == 0:
        print('Keine ausgesiebt')
    else :
        print('{} Audiofiles rausgeworfen'.format(rausgeworfen))

    pwr_box = pwr.copy()

    ### TODO: Das hier muss an die richtige Stelle geschoben werden :-) Ist nur zum testen um die Ausführung an einer geeigneten Stelle zu unterbrechen
    if do_test_mode:
        plt.close('all')
        exit(1)

    idx = 0
    while idx < 2:
        sieb_energien = sieb_energien.dropna()
        # Die Prognose darf nicht mit allen Daten gespeist werden, es muss eine !gute! Stichprobe sein
        if idx < 1:
            progmax = prognose(sieb_energien[["GesamtEnergie"]].to_numpy(), gamma=0.997)
        else:
            progmax = prognose(sieb_energien[["GesamtEnergie"]].to_numpy(), gamma=0.95)

        # siebe nun anhand der prognostizierten Schwellenwerte.
        for x in sieb_energien.index:
            if sieb_energien.loc[x, "GesamtEnergie"] > progmax:
                sieb_energien.drop(x, inplace=True)
        idx += 1
        print(sieb_energien.shape[0])

    # sieb_energien.to_csv('sieb.csv', index=True)
    wavfiles = sieb_energien.copy()
    wavfiles = wavfiles.join(getsecs(wavfiles))

    ################################################
    # Untersuchen des Signals und Fenstern
    ################################################
    win, step = 0.005, 0.005  # Laufendes Fenster, keine Überlappung
    if not do_test_mode:
        # if do_test_mode:
        for audiofile, row in sieb_energien.iterrows():
            # audiofile =  audiofile.lstrip()
            anzahl_bearbeitet += 1
            anzahlnochnicht -= 1
            iteration_over_file = 0
            ################################################
            # Laufe 2x über das Signal. Es stecken meistens 2 Ticks pro File
            ################################################
            while iteration_over_file < 2:
                txt = ('Dies ist die {}. csvDatei von {} im {}. Durchlauf').format(anzahl_bearbeitet,
                                                                                anzahl,
                                                                                iteration_over_file + 1)
                txt1 = ('Bearbeiten von {}.').format(audiofile)
                print(txt, '\n', txt1)

                ################################################
                # Rolling Window
                ################################################
                f, fn, fs, signal = process_folder(audiofile, audio_dir, win, step)

                time = np.linspace(0., len(signal) / float(fs), len(signal))
                duration = len(signal) / float(fs)
                timew = np.arange(0, duration - step, win)

                ################################################
                # Nehme den gefunden Index und schneide signal heraus in tmps
                ################################################
                olds, tmps, news = cut_signal(f, fn, signal)

                zeile = audiofile.strip(audio_dir).strip('.wav') + 'tick' + str(iteration_over_file)
                zeilennamen.append(zeile)

                ################################################
                # Versuche das geschnittene Signal in die csv zu drücken.. Wenn es nicht, da nicht gleich lang so passt
                # das Programm das geschnittene Signal an und füllt es mit einem konstanten Wert..
                #
                ################################################
                try:
                    csv_exp.append(tmps)
                except ValueError:
                    errata.append((zeile,tmps))
                    print('Diese Datei muss näher untersucht werden:\t' + zeile)
                    # Tatsächlich habe ich die Dateien bereits verworfen gehabt.. aber dann doch wieder eingebaut..
                    # try:
                    #    angepasst = np.pad(tmps, [csv_exp.shape[0] - tmps.shape[0], 0], 'constant')
                    #    csv_exp.append(angepasst)
                    # except ValueError:
                    #    pass
                ################################################
                # Export der csv datei wenn länger als x
                ################################################
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

                ################################################
                # Setzen vor Rekursion Ende
                ################################################

                signal = news.copy()
                iteration_over_file += 1


                ################################################
                # achtung while Schleife ende
                ################################################

   
    plt.close('all')

    print('bye world')

'''
The join() method inserts column(s) from another DataFrame, or Series.
'''
