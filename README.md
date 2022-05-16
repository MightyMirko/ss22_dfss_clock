## ss22_dfss_clock
#Sommersemester 2022 - Design for Six Sigma Uhrenprojekt :) 

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


## Source
https://stackoverflow.com/questions/2060628/reading-wav-files-in-python
https://www.programcreek.com/python/example/93227/scipy.io.wavfile.read
https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
https://klyshko.github.io/teaching/2019-02-22-teaching

