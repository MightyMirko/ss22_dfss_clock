# @Author Mirko Matosin und Alexander Wünstel
# @Date 14.02.2022
#


# fft anschauen. clustern, standardisieren, normalisieren
# isolation forest
# db scan

import os
import sys  # garbage collector anpassen
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import decimate
from scipy.stats import t

###################
# Anlegen globaler Var
###################
CSV_V___ = 'csv_v2_1'
plt.rcParams['figure.dpi'] = 150
# plt.rcParams['interactive'] = True
plt.rcParams['figure.figsize'] = (12, 9)


def plotsounds(ax, data, filename='N/A', samplerate=48000, log=True,
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
    tt = np.arange(0, len(data), 1)
    out = ax.plot(tt, data, label=filename)
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


def obj_size_fmt(num):
    if num < 10 ** 3:
        return "{:.2f}{}".format(num, "B")
    elif ((num >= 10 ** 3) & (num < 10 ** 6)):
        return "{:.2f}{}".format(num / (1.024 * 10 ** 3), "KB")
    elif ((num >= 10 ** 6) & (num < 10 ** 9)):
        return "{:.2f}{}".format(num / (1.024 * 10 ** 6), "MB")
    else:
        return "{:.2f}{}".format(num / (1.024 * 10 ** 9), "GB")


def memory_usage():
    memory_usage_by_variable = pd.DataFrame({k: sys.getsizeof(v) for (k, v) in globals().items()}, index=['Size'])
    memory_usage_by_variable = memory_usage_by_variable.T
    memory_usage_by_variable = memory_usage_by_variable.sort_values(by='Size', ascending=False).head(10)
    memory_usage_by_variable['Size'] = memory_usage_by_variable['Size'].apply(lambda x: obj_size_fmt(x))
    return memory_usage_by_variable


class CTick:
    '''

    '''

    # [vars(c) for c in tick_vector]
    def as_dict(self):
        return {'filename': self.filename, 'rectime': self.rectime, 'Zeigerwinkel': self.sektor,
                'tickfolge': self.tickfolge}

    def get_tick(self):
        return pd.Series(self.ticksignal)

    def __init__(self, filename, ticksignal, whichtick):
        self.filename = filename
        dat = os.path.basename(filename).lstrip().split(".")[0]
        # dat = dat.split(".")[0]
        self.rectime = datetime.strptime(dat, form)
        self.sektor = Klassenzuweisung(self.rectime.second)
        self.tickfolge = whichtick
        self.ticksignal = ticksignal

    def plotme(self):
        '''
        Kann genutzt werden um das Signal zu plotten. Würde dies gerne mit einer FFT ausbauen.
        :return:
        '''
        fig, axs = plt.subplots(2, 1, figsize=(12, 9))
        tt = np.arange(0, len(self.ticksignal), 1)
        axs[0].plot(tt, self.ticksignal, label='Ticksignal')
        axs[0].legend()
        axs[0].set_title('Ticken im Signal {}'.format(self.filename))
        # axs[0].scatter(x=indexofpeak, y=d_energy.max(), label='Max gefunden')
        # tt = np.arange(0, len(signal), 1)
        # axs[1].scatter(x=indexofpeak, y=d_energy.max(), label='Max gefunden')
        # axs[1].plot(tt, signal, label='Energie')

        plt.show()
        plt.close(fig)

        pass

    def add_tick(self, ):
        pass

    def create_data(self):
        pass


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
    return out


def getTicks_fromSignal(energy, signal, win=0.05, fs=48000,
                        samples_before=0.04,
                        samples_after=0.1):
    """
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
    samples_after = samples_after * fs
    samples_before = samples_before * fs

    d_energy = np.diff(energy)
    indexofpeak = 0
    # while :
    indexofpeak = d_energy.argmax()
    index_im_signalbereich = indexofpeak * fs * window
    # Wo ist das Maximum
    cutback_samples = int(index_im_signalbereich - samples_before)
    cutfwd_samples = int(index_im_signalbereich + samples_after)

    if cutback_samples <= 0:
        raise ValueError('Zu nah am Anfang')
    if cutfwd_samples >= len(signal):
        raise ValueError('Zu nah am Ende')

    olds = signal.copy()
    tmps = signal[cutback_samples:cutfwd_samples]
    news = signal.copy()

    news[cutback_samples:cutfwd_samples] = signal[cutback_samples:cutfwd_samples] * 0

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
    features = []
    window = int(pwin)
    pstep = int(pstep)

    while current_position + window - 1 < number_of_samples:
        count_fr += 1
        # get current window
        x = signal[current_position:current_position + window]
        # update window position
        current_position = current_position + pstep
        # short-term energy
        features.append(energy(x))
    return features


def check_dir(directory, testmode=True):
    if testmode:
        directory = r'data'
        return directory
    if not os.path.exists(directory):
        raise IOError
    else:
        return directory


def speichere_Dataframe(tickv, anzahl, bearbeitet, filepath):
    df = pd.DataFrame([vars(x) for x in tickv])
    df3 = pd.DataFrame(df['ticksignal'].tolist())
    dfsave = pd.concat([df, df3], axis=1)
    dfsave.drop('ticksignal', axis=1, inplace=True)
    outn = str(bearbeitet) + '_von_' + str(anzahl) + "-output.csv"

    try:
        output = os.path.join(filepath + '\\' + CSV_V___ + '\\' + outn)
        dfsave.to_csv(output, index=True)
    except PermissionError:
        outn = 'io_hand' + outn
        output = os.path.join(filepath + '\\' + 'csv' + '\\' + outn)
        dfsave.to_csv(output, index=True)

    finally:
        print(outn)
        del df, dfsave, df3


if __name__ == "__main__":

    ################################################
    # Anlegen der Kontrollvariablen
    ################################################
    print('hello World')
    do_test_mode = True  # Diverse Beschleuniger
    on_ms_surface = False
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
    audio_dir = check_dir(r'H:\Messung_BluetoothMikro\Messung 3\Audios', testmode=False)
    ################################################
    # Anlegen der Variablen und Wav-Liste
    ################################################
    form = '%Y%m%d_%H_%M_%S'
    iteration_over_file = 0
    anzahl_bearbeitet = 0
    csvlength = 300  # Achtung es werden die Zeilen 2x gezählt -> 50 dateien = 100 zeilen
    wavfiles = []
    signals = []
    sigexp = []
    ################################################
    # Anlegen und holen aller Daten aus der audioDir, dann Desinfizieren der Wavliste
    ################################################
    with os.scandir(audio_dir) as it:
        for entry in it:
            if entry.name.endswith('.wav') and entry.is_file():
                wavfiles.append(entry.name)

    # wavfiles = np.asarray(wavfiles)
    # wavfiles = wavfiles[:400]
    anzahl = len(wavfiles)
    anzahlnochnicht = anzahl
    if do_test_mode:
        wavfiles = np.random.choice(wavfiles, 23)

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
    progmin, progmax = 0.3, 3.12

    ################################################
    # Untersuchen des Signals und Fenstern
    ################################################
    window, step = 0.005, 0.005  # Laufendes Fenster, keine Überlappung
    duration = 2
    timew = np.arange(0, duration - step, window)

    ################################################
    # Erste for schleife um die Grundgesamtheit GG zu sammeln und zu lesen
    ################################################
    i = 0
    tickanzahl = 2
    tick_vector = []
    for audiofile in wavfiles:
        anzahl_bearbeitet += 1
        anzahlnochnicht -= 1

        # audiofile = audiofile.lstrip()
        filepath = os.path.join(audio_dir, audiofile)
        fs, signal = wavfile.read(os.path.join(audio_dir, audiofile))  # , mmap=True)

        ################################################
        # Berechnung der Gesamtenergie und anschließendes Kicken. Eventuell kann man vorher downsamplen?
        ################################################
        # mmap = np.asarray(signal)

        nrg = np.sum(signal ** 2, axis=0)
        if nrg <= progmin and nrg > progmax:
            errata.append(audiofile)
            continue
        elif nrg > 10:
            errata.append(audiofile)
            continue
        ################################################
        # Downsampling für v2.1
        ################################################
        signal = decimate(signal, 2)  # keine Vorfilterung notwendig!
        fs = 24e3
        ################################################
        # Extrahiere Tick
        ################################################
        i = 1
        while i <= tickanzahl:
            ################################################
            # Frame Analyse / Rolling Window
            ################################################
            feat = extract_params(signal, window * fs, step * fs)
            # signals.append(signal)

            ################################################
            # Finde den Tick..
            ################################################
            try:
                olds, tmps, news = getTicks_fromSignal(feat, signal, fs=fs)
            except ValueError:
                break
            # fig, ax = plt.subplots(2,1); plotsounds(ax[0],signal, log=False); plotsounds(ax[1],tmps, log=False); plt.show()
            ################################################
            # Validierung des Tick Signals..
            ################################################
            if len(tmps) > 6720:
                errata.append(audiofile)
                tmps = tmps[:6720]

            ################################################
            # Speichern und Klassierung des Tick Signals..
            ################################################

            tickit = 'tick' + str(i)
            signal = news
            tick_object = CTick(audiofile, tmps, tickit)
            tick_vector.append(tick_object)
            i += 1

            if do_test_mode:
                tick_object.plotme()
        ################################################
        # achtung while Schleife ende
        ################################################

        if len(tick_vector) >= csvlength * 2:
            # pick = np.random.choice(tick_vector,1)
            # [x.plotme() for x in pick]
            speichere_Dataframe(tick_vector, anzahl=anzahl, bearbeitet=anzahl_bearbeitet, filepath=audio_dir)
            tick_vector = []
        else:
            continue
    # speichere_Dataframe(tick_vector, anzahl=anzahl, bearbeitet=anzahl_bearbeitet, filepath=audio_dir)
    tick_vector = []
    try:
        outn = (str(datetime.now()) + 'errata.csv')
        output = os.path.join(filepath + '\\' + CSV_V___ + '\\' + outn)
        pd.DataFrame(errata).to_csv(output, index=True)
    except:
        print('Errata konnte nicht gespeichert werden')

    # TODO: Das hier muss an die richtige Stelle geschoben werden :-)
    # Ist nur zum testen um die Ausführung an einer geeigneten Stelle zu unterbrechen
    if do_test_mode:
        plt.close('all')
        print('Test mode :-) ')
        exit(1)

    plt.close('all')
    print('bye world')

'''
The join() method inserts column(s) from another DataFrame, or Series.
'''
# https://stackoverflow.com/questions/41705776/combine-two-dataframes-with-same-index-unordered
# https://stackoverflow.com/questions/28773683/combine-two-pandas-dataframes-with-the-same-index
# TODO: https://de.wikipedia.org/wiki/Zentrierung_(Statistik)
