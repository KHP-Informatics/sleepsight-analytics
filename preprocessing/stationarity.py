import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class Stationarity:

    @property
    def stationaryData(self):
        return self.__stationaryData

    @stationaryData.setter
    def stationaryData(self, sD):
        self.__stationaryData = sD

    @property
    def stationaryStats(self):
        return self.__stationaryStats

    @stationaryStats.setter
    def stationaryStats(self, sS):
        self.__stationaryStats = sS

    def __init__(self, data):
        self.data = data.copy()
        self.features = self.data.columns
        self.stationaryData = data.copy()
        self.stationaryStats = []

    def makeStationary(self, show=False):
        stationaryDataTmp = []
        stationaryStatsTmp = []
        for feature in self.features:
            series = pd.Series(self.data[feature])
            seriesVal = series.values
            if (feature not in 'timestamp') and (feature not in 'datetime'):
                seriesVal = np.nan_to_num(np.fromstring(seriesVal))
                result = adfuller(seriesVal)
                integrationOrder = 0
                while result[0] > result[4]['5%']:
                    series = series.diff()
                    seriesVal = np.nan_to_num(series.values)
                    result = adfuller(seriesVal)
                    integrationOrder += 1
                stationaryStatsTmp.append([feature, result[0], result[4]['5%'], result[1], integrationOrder])
            stationaryDataTmp.append(seriesVal)
        stationaryDataTmp = pd.DataFrame(stationaryDataTmp)
        self.stationaryStats = pd.DataFrame(stationaryStatsTmp, columns=['Feature','T-value', 'Critical value (5%)', 'p-value', 'I(r)'])
        self.stationaryData = stationaryDataTmp.T
        self.stationaryData.columns = self.features
        if show:
            print(self.stationaryStats)

