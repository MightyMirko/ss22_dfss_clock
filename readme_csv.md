

## CSV Aufbau
### Name

*Anzahlnochnichtbearbeitet - Anzahlbearbeitet - output.csv*

### Spalten
<index>,GesamtEnergie,rectime,ZeigerWinkel,tickfolge,0,1,2,

* Voller Dateiname (meistens 2x vorhanden..)
* Energie der vollen 2 Sekunden
* dateformat der Aufnahmezeit
* Zeigerwinkel anhand der Aufnahmesekunde: 
    
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


* Tickfolge bedeutet ob es der 1. oder 2. erkannte Tick war. Es erfolgt nur eine grobe Validierung. Aber es sollte erstmal passen. 

* 0,1,...,6719
Dies sind die float 32 PCM Werte.. Sie gehen von -1 bis +1. 
Es wurde nicht überprüft ob der Wertebereich eingehalten worden ist
>   IEEE float PCM in 32- or 64-bit format is supported, with or without mmap. Values exceeding [-1, +1] are not clipped. [scipy.wavfile]


