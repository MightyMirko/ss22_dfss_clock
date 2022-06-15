# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:09:38 2022

@author: Alexander
"""



Zeiger = 0

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
    out = 'Zeiger Fehler!!!'

print(out)