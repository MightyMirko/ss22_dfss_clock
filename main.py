# @Author Mirko Matosin und Alexander Wünstel
# @Date 14.02.2022
#


# fft anschauen. clustern, standardisieren, normalisieren
# isolation forest
# db scan

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.stats import t

###################
# Anlegen globaler Var
###################
plt.rcParams['figure.dpi'] = 150
# plt.rcParams['interactive'] = True
plt.rcParams['figure.figsize'] = (12, 9)


def cut_signal(energy, signal, win=0.05, fs=48000,
               davor_insek=0.04,
               danach_insek=0.1):
    """
    TODO: Funktion sollte auf integer umgebaut werden

    Diese Funktion nimmt den Energie Vector der vorher erstellt worden ist, leitet diesen ab und detektiert die
    Stelle (indexofpeak) wo das Maximum ist (quasi die x = y.max()). Dann wird geschnitten, anhand der Anzahl Samples
    zurück und vor, in Abhängigkeit der Samplerate
    :param davor_insek: Wie viele Sekunden vor dem Ticken soll geschnitten werden
    :param danach_insek: Wieviel Sekunden danach
    :param energy: Vector der Energie
    :param signal: Signal (numpy array)
    :param win: Fensterbreite des Energie_vektors
    :param fs: samplerate
    :return:
    """
    d_energy = np.diff(energy)
    indexofpeak = d_energy.argmax()  # Wo ist das Maximum
    peak_in_ms = win * indexofpeak

    back_in_ms = peak_in_ms - davor_insek
    adv_in_ms = peak_in_ms + danach_insek

    back_in_sample = int(fs * back_in_ms)
    adv_in_sample = int(fs * adv_in_ms)

    olds = signal.copy()
    tmps = signal[back_in_sample:adv_in_sample]
    news = signal.copy()

    news[back_in_sample:adv_in_sample] = signal[back_in_sample:adv_in_sample] * 0

    return olds, tmps, news


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


def Klassenzuweisung(sec_wert):
    ''' Hier wird erstmal der Sekundenwert der Datei uebergeben
        zur Kompensation des statischen Fehlers werden noch 3 Sekunden
        dazu addiert
        Gibt die Kalassenzuweisung und den korrigierten Sekundenwert zur�ck '''

    # Kompensation des statischen Fehlers
    sec_wert = sec_wert + 3
    # �berlauf verhinden
    sec_wert = sec_wert % 60
    out = ''
    if (sec_wert >= 0) and (sec_wert < 7.5):
        out = 'oben'
    elif (sec_wert > 7.5) and (sec_wert < 22.5):
        out = 'rechts'
    elif (sec_wert > 22.5) and (sec_wert < 37.5):
        out = 'unten'
    elif (sec_wert > 37.5) and (sec_wert < 52.5):
        out = 'links'
    elif (sec_wert > 52.5) and (sec_wert < 60):
        out = 'oben'
    else:
        out = 'Fehler!!!'
    return out, sec_wert


def energy(frame):
    """Computes signal energy of frame"""
    out = 0
    try:
        out = np.sum(frame ** 2)
        a = np.float64(len(frame))
        out /= a
    except:
        print(a, out)

    return out


def extract_params(signal, pwin, pstep):
    '''
    Habe hier mein eigenes Rolling Window nur für die Energie geschrieben.
    Dann werden auch keine 67 andere Feature geschrieben, sondern nur die relevanten
    :param signal:
    :param pwin:
    :param pstep:
    :return:
    '''
    number_of_samples = len(signal)  # total number of samples
    current_position = 0
    count_fr = 0
    num_fft = int(pwin / 2)
    features = []
    window = int(pwin)
    step = int(pstep)

    while current_position + window - 1 < number_of_samples:
        count_fr += 1
        # get current window
        x = signal[current_position:current_position + window]
        # update window position
        current_position = current_position + pstep
        # short-term energy
        features.append(energy(x))
    return features


def check_dir(directory, testmode = True):
    if testmode:
        directory = r'data'
    elif not os.path.exists(directory):
        raise IOError
    return directory


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
    audio_dir = check_dir(r'H:\Messung_BluetoothMikro\Messung 3\Audios', testmode=do_test_mode)
    ################################################
    # Anlegen der Variablen und Wav-Liste
    ################################################
    iteration_over_file = 0
    anzahl_bearbeitet = 0
    wavfiles = []
    csvlength = 300  # Achtung es werden die Zeilen 2x gezählt -> 50 dateien = 100 zeilen
    signals = []

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
    # wavfiles = np.random.choice(wavfiles,5)
    ################################################
    # Schätzung unbekannter Parameter über die t-verteilte Grundgesamtheit
    ################################################
    ################################################
    # Plotte die Grundgesamtheit und dann jedes mal wieder nach dem Sieben mittels prognose
    ################################################
    ################################################
    # Harken (grobes Sieben) durch die Daten und wegkicken
    ################################################
    ################################################
    # boxcox
    ################################################
    ################################################
    # Prognose und wegkicken
    ################################################
    progmin, progmax = 3e-6, 1e3

    ################################################
    # Untersuchen des Signals und Fenstern
    ################################################
    window, step = 0.005, 0.005  # Laufendes Fenster, keine Überlappung
    duration = 2
    timew = np.arange(0, duration - step, window)
    if do_test_mode:
        ################################################
        # Erste for schleife um die Grundgesamtheit GG zu sammeln und zu lesen
        ################################################
        i = 0
        tickanzahl =2
        for audiofile in wavfiles:
            anzahl_bearbeitet += 1
            anzahlnochnicht -= 1

            audiofile = audiofile.lstrip()
            filepath = os.path.join(audio_dir, audiofile)
            fs, signal = wavfile.read(os.path.join(audio_dir, audiofile))#, mmap=True)
            ################################################
            # Berechnung der Gesamtenergie und anschließendes Kicken. Eventuell kann man vorher downsamplen?
            ################################################
            # mmap = np.asarray(signal)
            nrg = np.sum(signal ** 2, axis=0)
            if nrg <= progmin and nrg > progmax:
                pass  # hier kann der prognose bereich schonmal kommen :)
            else:
                pass

            ################################################
            # Downsampling
            ################################################
            i = 1
            while i < tickanzahl:
                ################################################
                # Frame Analyse / Rolling Window
                ################################################
                feat = extract_params(signal,window,step)
                signals.append(signal)
                ################################################
                # Finde den Tick..
                ################################################
                olds, tmps, news = cut_signal(feat[0], signal)
                i+=1

        asarr = np.asarray(signals)
        nrg = np.sum(asarr ** 2, axis=1)
        # TODO: Das hier muss an die richtige Stelle geschoben werden :-)
        # Ist nur zum testen um die Ausführung an einer geeigneten Stelle zu unterbrechen

        if do_test_mode:
            plt.close('all')
            print('Test mode :-) ')
            exit(1)
        ################################################
        # Nehme den gefunden Index und schneide signal heraus in tmps
        ################################################
        #

        # df = pd.DataFrame({'energie':nrg}, index=wavfiles)
        # df2 = pd.concat([df,pandas.DataFrame(asarr)])
        ### Bis hierhin ist neu geschrieben..

        kleinereGG = 1

        for audiofile in kleinereGG:
            # extract short-term features using non-overlapping windows
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
                    errata.append((audiofile, tmps, 'Randwert'))
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
                    outn = str(anzahlnochnicht) + '-' + str(anzahl_bearbeitet) + "-output.csv"
                    tsamples_df = pd.DataFrame(tickSignal_liste, index=zeilennamen)

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
