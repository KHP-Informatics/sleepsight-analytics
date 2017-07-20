# !/bin/python3
import numpy as np
import pandas as pd
from tools import QuickPlot, detect_peaks

# testing signal's serial dependency
# determining periodicity using ACF (auto-correlation function)
# testing periodicity using Pearson's correlation

class Periodicity:

    @property
    def periodicity(self):
        periodicity = {
            #'scf': self.scf,
            'acf': self.acf,
            #'pcf': self.pcf
        }
        return periodicity

    def __init__(self, log, identifier='SleepSight', sensorName='Sensor name', path='/'):
        self.log = log
        self.observations = []
        self.sensorName = sensorName
        self.path = path
        self.id = identifier
        self.observationValid = True

    def addObservtions(self, observations):
        try:
            obs = np.array(observations, dtype='U64')
            obs_missing = np.where(obs == '-')
            obs[obs_missing] = 999999
            obs_masked = np.ma.masked_array(obs, dtype='float64')
            obs_masked[obs_missing] = np.ma.masked
            self.observations = obs_masked
        except ValueError:
            self.observationValid = False

        # serial-correlation function

    def serial_corr(self, step=1, nSteps=10):
        if self.observationValid:
            self.scf = []
            n = len(self.observations)
            for i in range(int(nSteps/step)):
                lag = step*i
                y1 = self.observations[lag:]
                y2 = self.observations[:n - lag]
                self.scf.append(np.corrcoef(y1, y2, ddof=0)[0,1])

    # auto-correlation function
    def auto_corr(self, nMinutes=20160, detectPeaks=True):
        if self.observationValid:
            acf_full =  np.correlate(self.observations, self.observations, mode='full')
            # 2nd half
            N = len(acf_full)
            acf_half = acf_full[N // 2: (N // 2 + nMinutes)]
            # standardise
            lengths = range((N // 2 + nMinutes), N // 2, -1)
            acf_stand = acf_half / lengths
            # normalise
            acf_norm = acf_stand / acf_stand[0]
            if detectPeaks:
                self.detectPeaks(acf_norm)
            self.acf = acf_norm

    def cross_cor(self, targetObservation, lag):
        if self.observationValid:
            x = list(self.observations)
            y = list(targetObservation)
            windowLength = len(x) - 2*lag
            if windowLength >= 5:
                xWinIdx = list(range(lag, (lag+windowLength)))
                featureCcf = []
                for i in range(0, 2*lag):
                    yWinIdx = list(range(i, (i + windowLength)))
                    xOfInterest = [x[idx] for idx in xWinIdx]
                    yOfInterest = [y[idx] for idx in yWinIdx]
                    cross = np.correlate(xOfInterest, yOfInterest)
                    featureCcf.append(cross[0])
                maxIdx = np.where(featureCcf == np.max(featureCcf))
                if len(maxIdx[0]) > 0:
                    delay = maxIdx[0][0] - lag
                    return delay
                return np.nan

            else:
                self.log.emit('[WARN] No cross validation possible. Choose a smaller lag to evaluate', indents=1)

    # pearson's correlation matrix
    def pearson_corr(self, lag=1440):
        if self.observationValid:
            n = int(len(self.observations)/lag) - 1
            observation_windows = []
            for i in range(n):
                observation_windows.append(self.observations[(i*lag):((i*lag)+lag)])
            self.pcf = np.corrcoef(observation_windows)

    def detectPeaks(self, y):
        if self.observationValid:
            self.peaks = detect_peaks(y, mpd=720, kpsh=True)
            peaksMean, peaksStd = self.generatePeakStats(self.peaks)
            self.peakStats = {'mean': peaksMean, 'std': peaksStd}

    def generatePeakStats(self, peaks):
        pDiff = pd.Series(peaks).diff()
        mean = np.mean(pDiff)
        std = np.std(pDiff)
        return (mean, std)

    def plot(self, type='all', show=True, save=False):
        if self.observationValid:
            if type is 'scf' or type is 'all':
                self.plotScf(show=show, save=save)
            if type is 'acf' or type is 'all':
                self.plotAcf(show=show, save=save)
            if type is 'pcf' or type is 'all':
                self.plotPcf(show=show, save=save)
            if type not in ['all', 'scf', 'acf', 'pcf']:
                self.log.emit('[PERIODICITY] WARN: Did not plot. Choose from "all", "scf", "acf" or "pcf".', indents=1)

    def plotScf(self, show=True, save=False):
        scfBetaOne = self.scf[1]
        text = 'Beta = 1; SCF = {}'.format(scfBetaOne)
        title = 'Serial-correlation: {}'.format(self.sensorName)
        qp = QuickPlot(path=self.path, identifier=self.id)
        qp.singlePlotOfTypeLine(self.scf, title=title, text=text, lineLabels=['SCF'], show=show, saveFigure=save)

    def plotAcf(self, withPeak=True, show=True, save=True):
        nDays = (len(self.acf) // 1440) + 1
        ticks = np.arange(0, 1440*nDays, 1440)
        tickLabels = np.arange(0,nDays)
        title = 'Auto-correlation: {}'.format(self.sensorName)
        qp = QuickPlot(path=self.path, identifier=self.id)
        if withPeak:
            qp.singlePlotOfTypeLine(self.acf, title=title, lineLabels=['ACF'], ticks=ticks, tickLabels=tickLabels,
                                    show=show, saveFigure=save, highlightPoints=self.peaks)
        else:
            qp.singlePlotOfTypeLine(self.acf, title=title, lineLabels=['ACF'], ticks=ticks, tickLabels=tickLabels,
                                    show=show, saveFigure=save)

    def plotPcf(self, show=True, save=True):
        title = 'Pearson\'s correlation matrix: {}'.format(self.sensorName)
        qp = QuickPlot(path=self.path, identifier=self.id)
        qp.singlePlotOfTypeHeatmap(self.pcf, title=title, show=show, saveFigure=save)

