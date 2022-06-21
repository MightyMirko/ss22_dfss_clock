#%%


import os.path
from os import listdir

import pandas
from tqdm.notebook import tqdm

dirs = [r'H:\Messung_BluetoothMikro\Messung 3\Audios', r'H:\Messung_BluetoothMikro\Messung 3\Bilder']


#%%


# dauert ca 1 Minute... ==> Forken :-)

columns = [
'Dateiname',
'Datum',
'Stunde',
'Minute',
'Sekunde']

bilder = pandas.DataFrame(columns=columns)
wavs = pandas.DataFrame(columns=columns)

show_progress = True
t = tqdm(total=1, unit="file", disable=not show_progress)

for d in dirs:
    if not os.path.exists(d):
        raise IOError("Cannot find:" + d)
    for p in listdir(d):
        if 'png' in p:
            a = p.strip('.png')
            b = a.split('_')
            bild = [p,b[0],b[1],b[2],b[3]]
            row = pandas.Series(bild, index=bilder.columns)
            bilder = bilder.append(row, ignore_index=True)
        elif 'wav' in p :
             a = p.strip('.wav')
             b = a.split('_')
             wav = [p,b[0],b[1],b[2],b[3]]
             row = pandas.Series(wav, index=wavs.columns)
             wavs = wavs.append(row, ignore_index=True)
        else:
            print(p)
        t.update()
    t.set_postfix(dir=d)

t.close()

#wavs = np.asarray(wavs)
#bilder = np.asarray(bilder)
bilder.head()


#%%





#%%


for spalten, werte in bilder.iteritems():

    if 'Sekunde' in spalten:
        for i in werte:
            sekzeiger = werte
            try:
                if not sekzeiger <= 0 and sekzeiger < 3:
                   print('Zeiger unsicher')
                if not sekzeiger <= 3 and sekzeiger < 27:
                   print('Zeiger fällt')
                if not sekzeiger <= 27 and sekzeiger < 33:
                   print('Zeiger unsicher')
                if not sekzeiger <= 33 and sekzeiger < 57:
                   print('Zeiger steigt')
                if not sekzeiger <= 57 and sekzeiger < 0:
                   print('Zeiger unsicher')
            except:
                pass


#%%


for name, values in bilder[['Sekunde']].iteritems():
    sekzeiger = values.to_list()
    for i in sekzeiger:
        try:
            if not sekzeiger <= 0 and sekzeiger < 3:
               print('Zeiger unsicher')
            if not sekzeiger <= 3 and sekzeiger < 27:
               print('Zeiger fällt')
            if not sekzeiger <= 27 and sekzeiger < 33:
               print('Zeiger unsicher')
            if not sekzeiger <= 33 and sekzeiger < 57:
               print('Zeiger steigt')
            if not sekzeiger <= 57 and sekzeiger < 0:
               print('Zeiger unsicher')
        except:
            pass


#%%


import os
import shutil

oldpath = ''
newpath = ''
pfad = 'Bilder'
cwd = ''
cwd =os.path.join('h:',os.sep,'Messung_BluetoothMikro', 'Messung 3')
# Path
print(cwd)


#%%


directory = 'hoch'
path = os.path.join(cwd, directory)
if not os.path.exists(path):
    try:
        os.makedirs(path, exist_ok = True)
        print("Directory '%s' created successfully" % directory)
    except OSError as error:
        print("Directory '%s' can not be created" % directory)

directory = 'runter'
path = os.path.join(cwd, directory)
if not os.path.exists(path):
    try:
        os.makedirs(path, exist_ok = True)
        print("Directory '%s' created successfully" % directory)
    except OSError as error:
        print("Directory '%s' can not be created" % directory)

ser = bilder['Sekunde'].to_numpy()
for i in ser:
    sekzeiger = int(i)
    oldpath = os.path.join(cwd,pfad, bilder.at[sekzeiger,'Dateiname'].lstrip())

    try:
        if not sekzeiger <= 0 and sekzeiger < 3:
            print('Zeiger unsicher')
        if not sekzeiger <= 3 and sekzeiger < 27:
            print('Zeiger fällt')
            newpath = os.path.join(os.path.join(cwd,pfad, 'runter', bilder.at[sekzeiger,'Dateiname'].lstrip()))
            print(newpath)
        if not sekzeiger <= 27 and sekzeiger < 33:
           print('Zeiger unsicher')
        if not sekzeiger <= 33 and sekzeiger < 57:
            print('Zeiger steigt')
            newpath = os.path.join(os.path.join(cwd,pfad, 'hoch', bilder.at[sekzeiger,'Dateiname'].lstrip()))
            print(newpath)
        if not sekzeiger <= 57 and sekzeiger < 0:
           print('Zeiger unsicher')
    except (AttributeError, TypeError):
        print('Dumm? Wertefehler')

    print(oldpath)
    print(newpath)

    try:
        if os.path.isfile(oldpath):
            shutil.copyfile(oldpath,newpath)
        else:
            raise IOError("Cannot find:" + oldpath)
    except IOError:
        print("Cannot find:\t" + oldpath)
        pass
    break



