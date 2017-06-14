import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

def readDataSet ():
    ll = list()
    with open('2008_2014_TANCAT_sense_V.csv', 'rt') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            auxDict = dict()
            auxDict['id'] = row["CIP"]
            # agafa nomes els pacients que tenen CIP
            if auxDict['id'].strip() == "":
                continue
            # Dades que s'obtenen directament de l'arxiu csv
            auxDict['dataIn'] = datetime.date(int(row["D_ingres"][:4]), int(row["D_ingres"][4:6]), int(row["D_ingres"][6:]))
            auxDict['dataOut'] = datetime.date(int(row["D_alta"][:4]), int(row["D_alta"][4:6]), int(row["D_alta"][6:]))
            if (datetime.date(2015, 1, 1) - auxDict['dataOut']).days < 30:
                continue # no pots arribar a saber si ha patit un reingres
            tipus = int('0'+row["C_ingres"]) # 1 -> urgent; 2 -> programat
            auxDict['diagPrinc'] = '_' + familiesDiag[max([i*(row["DP"][:3] >= familiesDiag[i]) for i in range(len(familiesDiag))])]
            auxDict['diagSec'] = '_' + familiesDiag[max([i*(row["DS1"][:3] >= familiesDiag[i]) for i in range(len(familiesDiag))])]

            # Dades que es dedueixen d'altres dades
            auxDict['diesIngr'] = (auxDict['dataOut'] - auxDict['dataIn']).days
            # primavera -> 0, estiu -> 1, tardor -> 2, hivern -> 3
            auxDict['estacioAny'] = '_' + str((auxDict['dataOut'].month - 3) % 12 // 3)
            auxDict['reingres'] = False # per defecte
            # si l'ingres anterior correspon al mateix pacient que l'ingres actual
            if len(ll) > 0 and ll[-1]['id'] == auxDict['id']:
                auxDict['nIngr'] = ll[-1]['nIngr'] + 1
                auxDict['nUrg'] = ll[-1]['nUrg'] + (tipus == 1)
                difDies = (ll[-1]['dataOut'] - auxDict['dataIn']).days
                ll[-1]['reingres'] = tipus == 1 and difDies < 30
            else:
                auxDict['nIngr'] = 1
                auxDict['nUrg'] = 1

            ll.append(auxDict)
    return ll

# families de diagnostics
familiesDiag = ['001', '140', '240', '280', '290', '320', '360', '390', '460', '520', '580', '630', \
                '680', '710', '740', '760', '780', '800']
df = pd.DataFrame(data=readDataSet(), columns=['diesIngr', 'nIngr', 'nUrg', 'estacioAny', \
                                               'diagPrinc', 'diagSec', 'reingres'])
df.to_csv(path_or_buf='dadesSantPauProc.csv')
