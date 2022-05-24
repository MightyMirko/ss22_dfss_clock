# Sommersemester 2022 - Design for Six Sigma Uhrenprojekt :) 

### Projektbericht

-----------
Mirko Matosin und Alexander Wuenstel
-----------

### Problem:
Das Ziel ist nun die wesentlichen Signalanteile in der Wav zu finden. Dies erfordert das Nutzen der FFT. Die FFT 
wird das Signal in seine spektralen Bestandteile zerlegen. 

### Nullhypothese:
Hypothese ist das ein lautes Grundrauschen bis ca. 800 Hz enthalten ist. Das Ticken nach unten (0-30s) selbst sollte im 
Frequenzbereich um 1k Hz liegen. Dies können wir nutzen um das Ticken zu isolieren und zu verstärken

### Alternativhypothese: 
Es gibt keinen Unterschied von hoch/runter.. Die Eigenschwingung \omega_E,R ist stets konstant.

### Vorgehen:
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


## Analyse der Datei mittels 3rd Party Software
### Audacity

 
Bild 1:
![alt text][fig1]

[fig1]: bilder_doku/Spurpanel001.png "Bild einer gebutterten FFT"


Bild 2:
![alt text][fig2]

[fig2]: bilder_doku/Spurpanel004.png "Vergrößerte Darstellung der ersten Spur und des ersten Tickens"


 
Bild 3:
![alt text][fig3]

[fig3]: bilder_doku/gebutterte_fft.png "Bild einer gebutterten FFT"

Das Bild [fig3] zeigt die Fourier-transformierte einer durch das Butterworth-Filter angepasste Sound-Datei. Diese Datei ist 11.5 Sekunden lang und beeinhaltet häufige Frequenzen um 2kHz und 7 kHz.



## Theorie
### Zitate

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

app.diagrams.net

FFT ist nun also die Summe aller sinus Funktionen normalisiert auf die Menge aller Datenpunkte N. 

$A(f_k) = \frac{1}{N} \sum_{n=0}^{N-1} a(t_n) e^{-i \frac{2 \pi kn}{N}}$





## Source
https://stackoverflow.com/questions/2060628/reading-wav-files-in-python
https://www.programcreek.com/python/example/93227/scipy.io.wavfile.read
https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
https://klyshko.github.io/teaching/2019-02-22-teaching


Wissenswertes:
Wird ein zeitlich unbegrenztes Signal während einer bestimmten Beobachtungszeit TB erfasst,
so steigt mit TB auch die gemessene Energie über alle Schranken. Die Energie pro Zeiteinheit,
also die Leistung, bleibt jedoch auf einem endlichen Wert. Bei vielen Signalen wird diese Leis-
tung sogar mit steigendem TB auf einen konstanten Wert konvergieren, z.B. bei periodischen
Signalen und stationären stochastischen Signalen. Diese Signale heissen darum Leistungssig-
nale. Die mittlere Leistung berechnet sich nach Gleichung (2.4).


 W < , P  0 endliche Signalenergie, verschwindende Signalleistung: Energiesignale
Diese Signale sind entweder zeitlich begrenzt (z.B. ein Einzelpuls) oder ihre Amplituden klin-
gen ab (z.B. ein Ausschwingvorgang). „Abklingende Amplitude“ kann dahingehend interpre-
tiert werden, dass unter einer gewissen Grenze das Signal gar nicht mehr erfasst werden kann,
man spricht von pseudozeitbegrenzten oder transienten Signalen. Etwas vereinfacht kann man
darum sagen, dass alle Energiesignale zeitlich begrenzt sind.
Ist die Beobachtungszeit TB gleich der Existenzdauer des Signals, so steigt bei weiter wachsen-
dem TB die gemessene Energie nicht mehr an. Die gemittelte Energie pro Zeiteinheit, also die
gemittelte Leistung, fällt jedoch mit weiter wachsendem TB auf Null ab. Diese Signale heissen
darum Energiesignale.

Bei transienten Signalen beeinflusst die Fensterlänge den Betrag der gemessenen Spektren, die
Ergebnisse der FFT müssen demnach noch skaliert werden: ein Rechteckpuls der Breite  und
der Höhe A hat bei  = 0 nach (2.27) die spektrale Amplitudendichte A..



Nur Zufallssignale können Träger
von unbekannter Information sein (konstante und periodische Signale tragen keine Informati-
on!). Diese Tatsache macht diese Signalklasse vor allem in der Nachrichtentechnik wichtig.
Störsignale gehören häufig auch zu dieser Signalklasse (Rauschen).

Wird ein Signal abgetastet, so wird sein Spektrum periodisch fortgesetzt mit der Abtastfrequenz fA bzw. $ \omega_A $ und gewichtet mit dem Abtastintervall T = 1/fA.

Ein kontinuierliches Tiefpass-Signal muss mit einer Frequenz abgetastet werden, die mehr als doppelt so gross ist wie die höchste im Signal vorkommende Frequenz.

Insbesondere bei Zufallssignalen lässt sich das Spektrum nicht exakt messen, man spricht deshalb oft von einer Schätzung des Spektrums und nicht von einer Messung.


Die Anzahl der Datenpunkte (k), die Sie vor- und zurückgehen lassen, ist der springende Punkt eines Filters zur Mittelwertglättung. Dies wird als die Ordnung des Filters bezeichnet.

Schwerpunktwellenlänge
Die Schwerpunktwellenlänge λ c {\displaystyle \lambda _{c}} \lambda _{c} (Index c für engl. centroid = Schwerpunkt) wird in der digitalen Signalverarbeitung als Maß eingesetzt, um ein Frequenzspektrum zu charakterisieren. Sie gibt an, wo sich der "Mittelpunkt" des Spektrums befindet.
Die Schwerpunktwellenlänge wird berechnet als gewichtetes arithmetisches Mittel der Wellenlängen λ {\displaystyle \lambda } \lambda , gewichtet mit ihren Amplituden anhand der Verteilungsfunktion s ( λ ) {\displaystyle s(\lambda )} {\displaystyle s(\lambda )}:[1] 
Die Schwerpunktwellenlänge wird in der digitalen Audiotechnik bzw. Akustik zur Bestimmung der Klangfarbe eingesetzt.[2][3]
https://de.wikipedia.org/wiki/Schwerpunktwellenl%C3%A4nge
