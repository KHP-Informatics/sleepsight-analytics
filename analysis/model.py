# !/bin/python3

import datetime
import numpy as np
import pandas as pd

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

    def __init__(self):
        pass

    def discretiseSymtomScore(self, stationarySymptom, rawSymptom):
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

class NonParaModel:

    def __init__(self, yFeature, dayDivisionHour=0):
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

    def submitData(self, participant):
        self.activeDataSy = participant.activeDataSymptom
        self.activeDataSl = participant.activeDataSleep
        self.sleepSummary = participant.sleepSummary
        self.yData = self.activeDataSy[['datetime', self.yFeature]]
        self.xFeatures = participant.passiveSensors
        self.xData = participant.passiveData[(['timestamp'] + self.xFeatures)]
        self.xDataNorm = self.xData[self.xFeatures] / self.xData[self.xFeatures].max()

    def constructModel(self):
        self.createIndexTable()
        dfSleep = self.extractSleepFeatures()
        dfRestActivity = self.extractRestActivityFeatures(leadFeature='intra_steps')

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
            if dateXData == dateEnd or i == (len(self.xData) - 1):
                self.indexDict[currentTableIndex]['indexStart'] = idxStart
                self.indexDict[currentTableIndex]['indexEnd'] = idxEnd
                currentTableIndex += 1
                if currentTableIndex >= len(self.indexDict):
                    break

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
        self.featuresSleep = pd.DataFrame(featureSleepTmp, columns=cols)
        self.featuresSleep.index = indexDates

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
        self.featuresRestActivity = pd.DataFrame(featureRATmp, columns=cols)
        self.featuresRestActivity.index = indexDates
        print(self.featuresRestActivity)

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




class GpModel:

    def __init__(self, xFeatures, yFeature, dayDivisionHour=0):
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

