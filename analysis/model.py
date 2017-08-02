# !/bin/python3
import sys
import datetime
import numpy as np
import pandas as pd
from pyentrp import entropy as ent
from tools import QuickPlot, detect_peaks

class ModelPrep:

    @property
    def discretisedStationarySymptomScoreTable(self):
        return self.__discretisedStationarySymptomScoreTable

    @discretisedStationarySymptomScoreTable.setter
    def discretisedStationarySymptomScoreTable(self, dSST):
        self.__discretisedStationarySymptomScoreTable = dSST

    @property
    def discretisedRawSymptomScoreTable(self):
        return self.__discretisedRawSymptomScoreTable

    @discretisedRawSymptomScoreTable.setter
    def discretisedRawSymptomScoreTable(self, dSST):
        self.__discretisedRawSymptomScoreTable = dSST

    def __init__(self, log):
        self.log = log

    def discretiseSymtomScore(self, stationarySymptom, rawSymptom):
        if 'label' not in rawSymptom.columns:
            m = round(np.mean(stationarySymptom['total']), 1)
            sd = round(np.std(stationarySymptom['total']), 2)
            minorIdx = stationarySymptom['total'] >= (m + sd)
            labels = ['major'] * len(minorIdx)
            for i in np.where(minorIdx == True)[0]:
                labels[i] = 'minor'
            labelsTable = pd.DataFrame(labels)
            labelsTable.columns = ['label']
            labelsTable.index = stationarySymptom.index
            self.discretisedStationarySymptomScoreTable = pd.concat([stationarySymptom, labelsTable], axis=1)
            self.discretisedRawScoreTable = pd.concat([rawSymptom, labelsTable], axis=1)
        else:
            self.discretisedStationarySymptomScoreTable = stationarySymptom
            self.discretisedRawScoreTable = rawSymptom

    def removeEntriesPriorToStudyStart(self, info):
        dStart = datetime.datetime.strptime(info['startDate'], '%d/%m/%Y')
        dates = self.discretisedRawScoreTable['datetime']
        dDates = []
        for d in dates:
            if type(d) is type(' '):
                dDates.append(datetime.datetime.strptime(d, '%Y-%m-%d %H:%M'))
            else:
                dDates.append(datetime.datetime(1970, 1, 1, 0, 0, 0))
        validDatesIdxs = []
        for i in range(0, len(dDates)):
            if dDates[i] > dStart:
                validDatesIdxs.append(i)
        self.discretisedRawScoreTable = self.discretisedRawScoreTable.loc[validDatesIdxs, :]
        self.discretisedStationarySymptomScoreTable = self.discretisedStationarySymptomScoreTable.loc[validDatesIdxs, :]



class NonParaModel:

    def __init__(self, yFeature, log, dayDivisionHour=0):
        self.log = log
        self.divTime = datetime.time(hour=dayDivisionHour)
        self.yFeature = yFeature
        self.sleepFeaturesSum = ['startTime',
                              'minutesToFallAsleep',
                              'minutesAfterWakeup',
                              'timeInBed',
                              'minutesAsleep',
                              'restlessCount',
                              'restlessDuration',
                              'awakeCount',
                              'awakeDuration',
                              'efficiency'
                              ]
        self.sleepFeaturesQ = ['sleepQuality']
        self.restActivtyFeatures = ['L5', 'M10', 'RA', 'IV', 'IS']

    def submitData(self, participant, xFeatures):
        self.activeDataSy = participant.activeDataSymptom
        self.activeDataSl = participant.activeDataSleep
        self.sleepSummary = participant.sleepSummary
        self.yData = self.activeDataSy[['datetime', self.yFeature]]
        self.xFeatures = xFeatures
        self.xData = participant.passiveData[(['timestamp'] + self.xFeatures)]
        self.xDataNorm = self.xData[self.xFeatures] / self.xData[self.xFeatures].max()
        self.enrolmentDate = datetime.datetime.strptime(participant.info['startDate'], '%d/%m/%Y')

    def constructModel(self):
        self.log.emit('[STATUS] Creating index table.', indents=1)
        self.createIndexTable()
        self.log.emit('[STATUS] Extracting rest-activity features.', indents=1)
        dfRestActivity = self.extractRestActivityFeatures(leadFeature='intra_steps')
        self.log.emit('[STATUS] Extracting disorganisation features.', indents=1)
        dfDisorganisation = self.extractDisorganisationFeatures()
        self.log.emit('[STATUS] Extracting sleep features.', indents=1)
        dfSleep = self.extractSleepFeatures()
        self.features = pd.concat([dfRestActivity, dfDisorganisation, dfSleep], axis=1)

    def createIndexTable(self):
        self.indexDict = []
        self.extractDateIdxsFromYData()
        self.extractDateIdxsFromXDataBasedOnY()
        self.removeIncompleteIndexs()

    def extractDateIdxsFromYData(self):
        for i in range(len(self.yData)):
            entry = {'index': i}
            entry['y'] = float(self.yData[i:i+1][self.yFeature])
            startDate, endDate = self.determineDatesFromYData(i)
            entry['dateStart'] = startDate
            entry['dateEnd'] = endDate
            if self.enrolmentDate.date() <= startDate.date():
                self.indexDict.append(entry)

    def determineDatesFromYData(self, index):
        dt_str = list(self.activeDataSy[index:(index+1)]['datetime'])[0]
        dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
        dtEnd = dt.replace(hour=self.divTime.hour, minute=0)
        tDay = datetime.timedelta(days=1)
        tMin = datetime.timedelta(minutes=1)
        dtStart= dtEnd - tDay + tMin
        return (dtStart, dtEnd)

    def extractDateIdxsFromXDataBasedOnY(self):
        idxStart = 0
        idxEnd = 0
        currentTableIndex = 0
        for i in range(len(self.xData)):
            dateStart = self.indexDict[currentTableIndex]['dateStart']
            dateEnd = self.indexDict[currentTableIndex]['dateEnd']
            dateXDataStr = list(self.xData[i:(i + 1)]['timestamp'])[0]
            dateXData = datetime.datetime.strptime(dateXDataStr, '%Y-%m-%d %H:%M')
            if dateXData <= dateStart and dateXData < dateEnd:
                idxStart = i
            if dateXData <= dateEnd:
                idxEnd = i
            if dateXData >= dateEnd or i == (len(self.xData) - 1):
                self.indexDict[currentTableIndex]['indexStart'] = idxStart
                self.indexDict[currentTableIndex]['indexEnd'] = idxEnd
                currentTableIndex += 1
                if currentTableIndex >= len(self.indexDict):
                    break
    def removeIncompleteIndexs(self):
        newIndexDict = []
        for index in self.indexDict:
            try:
                test = index['indexStart']
                newIndexDict.append(index)
            except KeyError:
                self.log.emit('[WARN] Removing index {}, due to missing \'indexStart\'.'.format(index['index']), indents=1)
        self.indexDict = newIndexDict

    def extractSleepFeatures(self):
        featureSleepTmp = []
        indexDates = []
        cols = self.sleepFeaturesSum + self.sleepFeaturesQ
        for index in self.indexDict:
            featureSleepSum = self.extractSleepSummarySample(index['dateStart'])
            featureSleepQue = self.extractSleepQuestionnaire(index['dateStart'])
            features = featureSleepSum + featureSleepQue
            featureSleepTmp.append(features)
            indexDates.append(index['dateStart'])
        featuresSleep = pd.DataFrame(featureSleepTmp, columns=cols)
        featuresSleep.index = indexDates
        return featuresSleep

    def extractSleepSummarySample(self, date):
        index = self.sleepSummary.index
        for i in range(0, len(index)):
            sample = self.sleepSummary.loc[index[i]]
            dateOfInterest = datetime.datetime.strptime(sample['dateOfSleep'], '%d/%m/%Y')
            if date.date() == dateOfInterest.date():
                return list(sample[self.sleepFeaturesSum])
        return ['NaN'] * len(self.sleepFeaturesSum)

    def extractSleepQuestionnaire(self, date):
        index = self.activeDataSl.index
        for i in range(0, len(index)):
            sample = self.activeDataSl.loc[index[i]]
            dateOfInterest = datetime.datetime.strptime(sample['dateTime'], '%Y-%m-%d %H:%M')
            if date.date() == dateOfInterest.date():
                return list(sample[self.sleepFeaturesQ])
        return ['NaN'] * len(self.sleepFeaturesQ)

    def extractRestActivityFeatures(self, leadFeature):
        featureRATmp = []
        indexDates = []
        cols = self.formatColumns(self.xFeatures, prefixes=['L5', 'M10', 'RA', 'IV', 'IS'])
        for index in self.indexDict:
            l5Idx, m10Idx = self.determineL5M10Indexes(index, leadFeature)
            L5 = self.xData.loc[l5Idx, self.xFeatures].mean()
            M10 = self.xData.loc[m10Idx, self.xFeatures].mean()
            RA = list(self.computeRA(L5, M10))
            IV = list(self.computeIntraDayVariability(index))
            IS = list(self.computeInterDayStability(index))
            RATmp = list(L5) + list(M10) + RA + IV + IS
            featureRATmp.append(RATmp)
            indexDates.append(index['dateStart'])
        featuresRestActivity = pd.DataFrame(featureRATmp, columns=cols)
        featuresRestActivity.index = indexDates
        return featuresRestActivity

    def determineL5M10Indexes(self, index, leadFeature):
        rawSamplePeriod = self.xData.loc[index['indexStart']:index['indexEnd']]
        leadFeatureRaw = rawSamplePeriod[leadFeature]
        leadFeatureRawSorted = leadFeatureRaw.sort_values()
        l5Idx = list(leadFeatureRawSorted.index[0:(60 * 5)])
        m10Idx = list(leadFeatureRawSorted.index[
                      (len(leadFeatureRawSorted.index) - (60 * 10)):(len(leadFeatureRawSorted.index) - 1)])
        return (l5Idx, m10Idx)

    def computeRA(self, L5, M10):
        M10PlusL5 = M10.add(L5, fill_value=0)
        M10MinusL5 = M10.subtract(L5, fill_value=0)
        RA = M10MinusL5.div(M10PlusL5, fill_value=0)
        return RA

    def computeIntraDayVariability(self, index):
        idxStart = index['indexStart']
        idxEnd = index['indexEnd']
        pData = self.xDataNorm.loc[idxStart:idxEnd, self.xFeatures]
        pDataDiffSquared = pData.diff() * pData.diff()
        nominatorP = len(self.xDataNorm) * pDataDiffSquared.sum()
        nData = self.xDataNorm[self.xFeatures]
        nDataMinusMean = nData - nData.mean()
        nDataSquared = nDataMinusMean * nDataMinusMean
        denominatorN = (len(self.xDataNorm)-1) * nDataSquared.sum()
        IV = nominatorP / denominatorN
        return IV

    def computeInterDayStability(self, index):
        idxStart = index['indexStart']
        idxEnd = index['indexEnd']
        pData = self.xDataNorm.loc[idxStart:idxEnd, self.xFeatures]
        pDataMinusMean = pData - pData.mean()
        pDataSquared = pDataMinusMean * pDataMinusMean
        nominatorP = len(self.xData) * pDataSquared.sum()
        nData = self.xDataNorm[self.xFeatures]
        nDataMinusMean = nData - nData.mean()
        nDataSquared = nDataMinusMean * nDataMinusMean
        denominatorN = len(pData) * nDataSquared.sum()
        IS = nominatorP / denominatorN
        return IS

    def extractDisorganisationFeatures(self):
        disorgFeaturesTmp = []
        indexDates = []
        cols = self.formatColumns(self.xFeatures, prefixes=['MSE', 'DWT'])
        for index in self.indexDict:
            MSE = self.computeMSE(index)
            DTWDist = self.computeDTW(index)
            disorgFeaturesTmp.append((MSE + DTWDist))
            indexDates.append(index['dateStart'])
        disorganisationFeatures = pd.DataFrame(disorgFeaturesTmp, columns=cols)
        disorganisationFeatures.index = indexDates
        return disorganisationFeatures

    def computeMSE(self, index, m_length=20):
        indexStart = index['indexStart']
        indexEnd = index['indexEnd']
        dfData = self.xData.loc[indexStart:indexEnd]
        MSE_means = []
        for feature in self.xFeatures:
            ts = list(dfData[feature])
            tsStd = np.std(ts)
            MSEs = ent.multiscale_entropy(ts, sample_length=m_length, tolerance=0.2*tsStd)
            MSE_means.append(np.mean(MSEs))
        return MSE_means

    def computeDTW(self, index):
        indexStart = index['indexStart']
        indexEnd = index['indexEnd']
        indexPrev = indexStart - 1440
        try:
            dfData = self.xData.loc[indexStart:indexEnd]
            dfDataPrev = self.xData.loc[indexPrev:indexStart]
            dtwDist = []
            for feature in self.xFeatures:
                ts = np.array(list(dfData[feature]))
                tsPrev = np.array(list(dfDataPrev[feature]))
                dist, path = self.DTW(ts, tsPrev)
                dtwDist.append(dist)
        except IndexError:
            dtwDist = ['NaN'] * len(self.xFeatures)
        return dtwDist

    def DTW(self, A, B,  window=sys.maxsize, d=lambda x, y: abs(x - y)):
        # create the cost matrix
        M, N = len(A), len(B)
        cost = sys.maxsize * np.ones((M, N))
        # initialize the first row and column
        cost[0, 0] = d(A[0], B[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(A[i], B[0])
        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(A[0], B[j])
        # fill in the rest of the matrix
        for i in range(1, M):
            for j in range(max(1, i - window), min(N, i + window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(A[i], B[j])
        # find the optimal path
        n, m = N - 1, M - 1
        path = []
        while (m, n) != (0, 0):
            path.append((m, n))
            m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key=lambda x: cost[x[0], x[1]])
        path.append((0, 0))
        return cost[-1, -1], path

    def formatColumns(self, features, prefixes):
        cols = []
        for prefix in prefixes:
            tmpF = []
            for i in range(0, len(features)):
                f = features[i].replace('_', ' ')
                if f not in ['LAM', 'FAM', 'VAM']:
                    f = f.capitalize()
                tmpF.append('{} {}'.format(prefix, f))
            cols = cols + tmpF
        return cols


class GpModel:

    def __init__(self, xFeatures, yFeature, log, dayDivisionHour=0):
        self.log = log
        self.divTime = datetime.time(hour=dayDivisionHour)
        self.xFeatures = xFeatures
        self.yFeature = yFeature

    def submitData(self, active, passive):
        self.activeData = active
        self.passiveData = passive
        self.yData = self.activeData[['datetime', self.yFeature]]
        xSelection = ['timestamp'] + self.xFeatures
        self.xData = self.passiveData[xSelection]

    def createIndexTable(self):
        self.indexDict = []
        self.extractDateIdxsFromYData()
        self.extractDateIdxsFromXDataBasedOnY()

    def extractDateIdxsFromYData(self):
        for i in range(len(self.yData)):
            entry = {'index': i}
            entry['y'] = float(self.yData[i:i+1][self.yFeature])
            startDate, endDate = self.determineDatesFromYData(i)
            entry['dateStart'] = startDate
            entry['dateEnd'] = endDate
            self.indexDict.append(entry)

    def determineDatesFromYData(self, index):
        dt_str = list(self.activeData[index:(index+1)]['datetime'])[0]
        dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
        dtEnd = dt.replace(hour=self.divTime.hour, minute=0)
        tDay = datetime.timedelta(days=1)
        tMin = datetime.timedelta(minutes=1)
        dtStart= dtEnd - tDay + tMin
        return (dtStart, dtEnd)

    def extractDateIdxsFromXDataBasedOnY(self):
        idxStart = 0
        idxEnd = 0
        currentTableIndex = 0
        for i in range(len(self.xData)):
            dateStart = self.indexDict[currentTableIndex]['dateStart']
            dateEnd = self.indexDict[currentTableIndex]['dateEnd']
            dateXDataStr = list(self.xData[i:(i + 1)]['timestamp'])[0]
            dateXData = datetime.datetime.strptime(dateXDataStr, '%Y-%m-%d %H:%M')
            if dateXData <= dateStart and dateXData < dateEnd:
                idxStart = i
            if dateXData <= dateEnd:
                idxEnd = i
            if dateXData == dateEnd or i == (len(self.xData) - 1):
                self.indexDict[currentTableIndex]['indexStart'] = idxStart
                self.indexDict[currentTableIndex]['indexEnd'] = idxEnd
                currentTableIndex += 1
                if currentTableIndex >= len(self.indexDict):
                    break


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

