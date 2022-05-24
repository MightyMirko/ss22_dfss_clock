================
Projektbericht
================
-----------
Mirko Matosin und Alexander Wuenstel
-----------


.. image:: bilder_doku/gebutterte_fft.png
    :alt: 'Bild einer gebutterten FFT'
    :scale: 50 %
.. _Bild:

.. ![Image](Icon-pictures.png "Bild der gebutterten FFT")
::

Das Bild_ zeigt die Fourier-transformierte einer durch das Butterworth-Filter angepasste Sound-Datei. Diese Datei ist 11.5 Sekunden lang und beeinhaltet häufige Frequenzen um 2kHz und 7 kHz.


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
