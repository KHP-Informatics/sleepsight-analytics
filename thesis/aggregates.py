import os
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

