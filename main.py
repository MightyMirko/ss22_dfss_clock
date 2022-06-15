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
from scipy.io import wavfile
from scipy.stats import t

###################
# Anlegen globaler Var
###################
plt.rcParams['figure.dpi'] = 150
# plt.rcParams['interactive'] = True
plt.rcParams['figure.figsize'] = (12, 9)

def cut_signal(nrg, s, fs =48000):
    energy = nrg  # Alle Daten aus der index nummer 1 ziehen..
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
        samplerate, data = wavfile.read(d + ('\\') + file_list[idx], mmap=True)
        # Berechnung
        energie[idx] = sum(data ** 2)
        try:
            if idx % int(number_of_files / 10) == 0:
                print(str(idx) + " von " + str(number_of_files))
        except:
            print(idx)
            pass
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


def get_zeigerwinkel(fromdf):
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
        Zeiger = date_string.second
        out =''
        if (Zeiger >= 0) and (Zeiger < 7.5):
            out = 'oben'
        elif (Zeiger > 7.5) and (Zeiger < 22.5):
            out = 'rechts'
        elif (Zeiger > 22.5) and (Zeiger < 37.5):
            out = 'unten'
        elif (Zeiger > 37.5) and (Zeiger < 52.5):
            out = 'links'
        elif (Zeiger > 52.5) and (Zeiger < 60):
            out = 'oben'
        else:
            out = 'Fehler!!!'
        print(out)
        dt_list.append((date_string, out))

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
    on_ms_surface = True
    plt.clf()

    ################################################
    # Arbeits Variablen
    ################################################
    tickSignal_liste = []  # Der exportierende
    tick_folge = []
    zeilennamen = []
    errata = []

    ################################################
    # Zuweisung und Überprüfung des Ordners anhand verschiedener Test-Variablen
    ################################################
    audio_dir = r'H:\Messung_BluetoothMikro\Messung 3\Audios'
    if do_test_mode or on_ms_surface:
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
    csvlength = 300  # Achtung es werden die Zeilen 2x gezählt -> 50 dateien = 100 zeilen

    ################################################
    # Anlegen und holen aller Daten aus der audioDir, dann Desinfizieren der Wavliste
    ################################################
    with os.scandir(audio_dir) as it:
        for entry in it:
            if entry.name.endswith('.wav') and entry.is_file():
                wavfiles.append(entry.name)

    wavfiles = np.asarray(wavfiles)
    # wavfiles = wavfiles[:400]
    anzahl = len(wavfiles)
    anzahlnochnicht = anzahl
    #wavfiles = np.random.choice(wavfiles,5)
    signals = []
    ################################################
    # Schätzung unbekannter Parameter über die t-verteilte Grundgesamtheit
    ################################################

    ################################################
    # Plotte die Grundgesamtheit und dann jedes mal wieder nach dem Sieben mittels prognose
    ################################################
    #fig, ax = plt.subplots(3, 2, sharex='all')  # , sharey='all')


    ################################################
    # Harken (grobes Sieben) durch die Daten und wegkicken
    ################################################
    ################################################
    # boxcox
    ################################################
    ################################################
    # Prognose und wegkicken
    ################################################

    ################################################
    # Untersuchen des Signals und Fenstern
    ################################################
    win, step = 0.005, 0.005  # Laufendes Fenster, keine Überlappung
    timew = np.arange(0, duration - step, win)
    if do_test_mode:

        ################################################
        # Erste for schleife um die Grundgesamtheit GG zu sammeln und zu lesen
        ################################################
        for audiofile in wavfiles:
            anzahl_bearbeitet += 1
            anzahlnochnicht -= 1

            audiofile =  audiofile.lstrip()
            filepath = os.path.join(audio_dir, audiofile)
            fs, signal = wavfile.read(os.path.join(audio_dir, audiofile), mmap=True)
            signals.append(signal)

        ################################################
        # Zweite Schleife um GG zu verkleinern
        ################################################
        asarr = np.asarray(signals)
        nrg = np.sum(asarr**2, axis=1)

        ### Bis hierhin ist neu geschrieben..
        #TODO: Das hier muss an die richtige Stelle geschoben werden :-)
        # Ist nur zum testen um die Ausführung an einer geeigneten Stelle zu unterbrechen

        if do_test_mode:
            plt.close('all')
            exit(1)

        kleinereGG = 1

        for audiofile in kleinereGG:
            # extract short-term features using non-overlapping windows
            f, fn = aF.feature_extraction(signal, fs, int(fs * win), int(fs * step), True)
            print(f[fn[0]])
            ################################################
            # Laufe 2x über das Signal. Es stecken meistens 2 Ticks pro File
            ################################################

            iteration_over_file = 0
            while iteration_over_file < 2:
                txt = ('Dies ist die {}. csvDatei von {} im {}. Durchlauf').format(anzahl_bearbeitet,
                                                                                anzahl,
                                                                                iteration_over_file + 1)
                txt1 = ('Bearbeiten von {}.').format(audiofile)
                print(txt, '\n', txt1)
                ################################################
                # Rolling Window
                ################################################
                ###############
                # Anlegen der Variablen
                ###############

                time = np.linspace(0., len(signal) / float(fs), len(signal))
                duration = len(signal) / float(fs)


                ################################################
                # Nehme den gefunden Index und schneide signal heraus in tmps
                ################################################
                olds, tmps, news = cut_signal(f, fn, signal)
                ################################################
                # Validierung des Tick Signals..
                ################################################

                if len(tmps) > 6720:
                    tmps = tmps[:6720]

                # zeile = audiofile.strip(audio_dir).strip('.wav') + 'tick' + str(iteration_over_file)
                tickit = 'tick' + str(iteration_over_file)
                ################################################
                # Versuche das geschnittene Signal in die csv zu drücken.. Wenn es nicht, da nicht gleich lang so passt
                # das Programm das geschnittene Signal an und füllt es mit einem konstanten Wert..
                #
                ################################################
                try:
                    tick_folge.append(tickit)
                    tickSignal_liste.append(tmps)
                    zeilennamen.append(audiofile)
                except ValueError:
                    errata.append((audiofile,tmps, 'Randwert'))
                    print('Diese Datei muss näher untersucht werden:\t' + audiofile)
                    # Tatsächlich habe ich die Dateien bereits verworfen gehabt.. aber dann doch wieder eingebaut..
                    # try:
                    #    angepasst = np.pad(tmps, [csv_exp.shape[0] - tmps.shape[0], 0], 'constant')
                    #    csv_exp.append(angepasst)
                    # except ValueError:
                    #    pass
                ################################################
                # Export der csv datei wenn länger als x
                ################################################
                # TODO: wenn Daten für vollständige CSV nicht mehr langen, dann sollte dies ebenfalls behandelt werden
                # #if wfdf.loc[audiofile] == wfdf.iloc[-1]:
               #    print("Ende")

                if len(tickSignal_liste) >= csvlength * 2:
                    outn = str(anzahlnochnicht ) + '-' + str(anzahl_bearbeitet) + "-output.csv"
                    tsamples_df = pd.DataFrame(tickSignal_liste, index=zeilennamen )

                    tsamples_df.head()
                    tfdf = pd.DataFrame(tick_folge, columns=['tickfolge'], index=zeilennamen)
                    conc = pd.concat([tfdf, tsamples_df], axis=1)
                    r = pd.merge(wfdf, conc, left_index=True, right_index=True)
                    # r = conc.join(wfdf, how='inner', rsuffix='_other')
                    # r = pd.merge(tfdf,tsamples_df, left_index=True, right_index=True, how='outer')

                    r.drop_duplicates(inplace=True)
                    # r = pd.DataFrame(tickSignal_liste, index=zeilennamen)

                    if r.isnull().values.any():
                        print('Achtung NaNs')
                        r.dropna(inplace=True)
                    if not do_test_mode:
                        try:
                            output = os.path.join(audio_dir + '\\' + 'csv' + '\\' + outn)
                            r.to_csv(output, index=True)
                        except PermissionError:
                            outn = 'io_hand' + outn
                            output = os.path.join(audio_dir + '\\' + 'csv' + '\\' + outn)
                            r.to_csv(output, index=True)
                        r = pd.DataFrame()
                        tickSignal_liste, tick_folge, zeilennamen = [], [], []

                ################################################
                # Setzen vor Rekursion Ende
                ################################################

                signal = news.copy()
                iteration_over_file += 1


                ################################################
                # achtung while Schleife ende
                ################################################

    outn = str(anzahl_bearbeitet - csvlength) + '-' + str(anzahl_bearbeitet) + "-output.csv"
    tsamples_df = pd.DataFrame(tickSignal_liste, index=zeilennamen)
    tsamples_df.head()
    tfdf = pd.DataFrame(tick_folge, columns=['tickfolge'], index=zeilennamen)
    conc = pd.concat([tfdf, tsamples_df], axis=1)
    r = pd.merge(wfdf, conc, left_index=True, right_index=True)
    # r = conc.join(wfdf, how='inner', rsuffix='_other')
    # r = pd.merge(tfdf,tsamples_df, left_index=True, right_index=True, how='outer')

    r.drop_duplicates(inplace=True)
    # r = pd.DataFrame(tickSignal_liste, index=zeilennamen)
    if not do_test_mode:
        try:
            output = os.path.join(audio_dir + '\\' + 'csv' + '\\' + outn)
            r.to_csv(output, index=True)
        except PermissionError:
            outn = 'io_hand' + outn
            output = os.path.join(audio_dir + '\\' + 'csv' + '\\' + outn)
            r.to_csv(output, index=True)
        except FileNotFoundError:
            os.mkdir(os.path.join(audio_dir, "csv"))
            output = os.path.join(audio_dir + '\\' + 'csv' + '\\' + outn)
            r.to_csv(output, index=True)

        r = pd.DataFrame()
        tickSignal_liste, tick_folge, zeilennamen = [], [], []


    plt.close('all')

    print('bye world')

'''
The join() method inserts column(s) from another DataFrame, or Series.
'''
# https://stackoverflow.com/questions/41705776/combine-two-dataframes-with-same-index-unordered
# https://stackoverflow.com/questions/28773683/combine-two-pandas-dataframes-with-the-same-index
# TODO: https://de.wikipedia.org/wiki/Zentrierung_(Statistik)
