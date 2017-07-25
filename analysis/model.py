# !/bin/python3
import sys
import datetime
import numpy as np
import pandas as pd
from pyentrp import entropy as ent

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
        dDates = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M') for d in dates]
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
        self.log.emit('[STATUS] Extracting disorganisation features.')
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

