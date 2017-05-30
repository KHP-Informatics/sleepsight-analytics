import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

#ToDO: insert into preprocessing class: differencing, limit to 56 days

class TimeSeries:

    def __init__(self, identifier, sensorName):
        self.id = identifier
        self.sensor = sensorName
        print(sensorName)

    def addObservtions(self, obs):
        self.ts = obs

    def diff(self):
        series = pd.Series(self.ts)
        seriesVal = series.values
        result = adfuller(seriesVal)
        integrationOrder = 0
        while result[0] > result[4]['5%']:
            series = series.diff()
            seriesVal = np.nan_to_num(series.values)
            result = adfuller(seriesVal)
            integrationOrder += 1
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        print('I(r): %f' % integrationOrder)
        #diff = series.diff()
        #diffVal = np.nan_to_num(diff.values)
        #adfResult = adfuller(diffVal)

