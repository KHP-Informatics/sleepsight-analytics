import os
import numpy as np
import pandas as pd
from tools import Participant

class Aggregates:

    def __init__(self, fileSelector, pathData, pathPlot):
        self.fileSelector = fileSelector
        self.pathData = pathData
        self.pathPlot = pathPlot
        self.aggregates = self.loadParticipants()

    def loadParticipants(self):
        files = os.listdir(self.pathData)
        participants = []
        for fileName in files:
            if self.fileSelector in fileName:
                p = Participant(id=fileName[0:2], path=self.pathData)
                p.activeSensingFilenameSelector = 'diary'
                p.metaDataFileName = 'meta_patients.json'
                p.load()
                participants.append(p)
        return participants

    def getMissingness(self):
        count = dict()
        dfDaily = dict()
        for i in range(0, len(self.aggregates)):
            participant = self.aggregates[i]
            missingness = self.getVariable(participant, 'missingness')
            count[participant.id] = missingness['count']
            for key in missingness['daily'].keys():
                missingness['daily'][key] = pd.Series(missingness['daily'][key])
            dfDaily[participant.id] = pd.DataFrame(missingness['daily'])
        dfCount = pd.DataFrame(count)
        return (dfCount, dfDaily)

    def getVariable(self, participant, variableName):
        return getattr(participant, variableName)

    def getPariticpantsInfo(self):
        dfInfo = pd.io.json.json_normalize(self.aggregates[0].info)
        for i in range(1,len(self.aggregates)):
            infoTmp = pd.DataFrame(pd.io.json.json_normalize(self.aggregates[i].info))
            dfInfo = pd.concat([dfInfo, infoTmp], ignore_index=True, verify_integrity=True)
        dfMedication = self.formatMedication(dfInfo['medication'])
        dfInfo = pd.concat((dfInfo, dfMedication), axis=1)
        return dfInfo

    def formatMedication(self, arr, hasDrug='Clozapine'):
        hasDrugBool = np.zeros(len(arr), dtype=bool)
        drugCount = np.zeros(len(arr))
        arrIndex = 0
        for drugs in arr:
            drugCount[arrIndex] = len(drugs)
            for drug in drugs:
                if drug['name'] in hasDrug:
                    hasDrugBool[arrIndex] = True
                    break
            arrIndex += 1
        dfMedicationFormat = pd.DataFrame({hasDrug:hasDrugBool, 'No.of.Drugs':drugCount})
        return dfMedicationFormat



