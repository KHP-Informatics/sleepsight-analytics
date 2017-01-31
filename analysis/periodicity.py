# !/bin/python3
import numpy as np
from tools import QuickPlot

# testing signal's serial dependency
# determining periodicity using ACF (auto-correlation function)
# testing periodicity using Pearson's correlation

class Periodicity:

    def __init__(self, identifier='SleepSight', sensorName='Sensor name', path='/'):
        self.observations = []
        self.sensorName = sensorName
        self.path = path
        self.id = identifier

    def addObservtions(self, observations):
        obs = np.array(observations, dtype='U32')
        obs_missing = np.where(obs == '-')
        obs[obs_missing] = 999999
        obs_masked = np.ma.masked_array(obs, dtype='float32')
        obs_masked[obs_missing] = np.ma.masked
        self.observations = obs_masked

        # serial-correlation function

    def serial_corr(self, step=1, nSteps=10):
        self.scf = []
        n = len(self.observations)
        for i in range(int(nSteps/step)):
            lag = step*i
            y1 = self.observations[lag:]
            y2 = self.observations[:n - lag]
            self.scf.append(np.corrcoef(y1, y2, ddof=0)[0,1])

    # auto-correlation function
    def auto_corr(self):
        self.acf = np.correlate(self.observations, self.observations, mode='same')

    # pearson's correlation matrix
    def pearson_corr(self, lag=1440):
        n = int(len(self.observations)/lag) - 1
        observation_windows = []
        for i in range(n):
            observation_windows.append(self.observations[(i*lag):((i*lag)+lag)])
        self.pcf = np.corrcoef(observation_windows)

    def plot(self, type='all', show=True, save=False):
        if type is 'scf' or type is 'all':
            self.plotScf(show=show, save=save)
        if type is 'acf' or type is 'all':
            self.plotAcf(show=show, save=save)
        if type is 'pcf' or type is 'all':
            self.plotPcf(show=show, save=save)
        if type not in ['all', 'scf', 'acf', 'pcf']:
            print('[PERIODICITY] WARN: Did not plot. Choose from "all", "scf", "acf" or "pcf".')

    def plotScf(self, show=True, save=False):
        scfBetaOne = self.scf[1]
        text = 'Beta = 1; SCF = {}'.format(scfBetaOne)
        title = 'Serial-correlation: {}'.format(self.sensorName)
        qp = QuickPlot(path=self.path, identifier=self.id)
        qp.singlePlotOfTypeLine(self.scf, title=title, text=text, lineLabels=['SCF'], show=show, saveFigure=save)

    def plotAcf(self, show=True, save=True):
        title = 'Auto-correlation: {}'.format(self.sensorName)
        qp = QuickPlot(path=self.path, identifier=self.id)
        qp.singlePlotOfTypeLine(self.acf, title=title, lineLabels=['ACF'], show=show, saveFigure=save)

    def plotPcf(self, show=True, save=True):
        title = 'Pearson\'s correlation matrix: {}'.format(self.sensorName)
        qp = QuickPlot(path=self.path, identifier=self.id)
        qp.singlePlotOfTypeHeatmap(self.pcf, title=title, show=show, saveFigure=save)

