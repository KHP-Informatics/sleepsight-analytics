# !/bin/python3

import numpy as np
import pandas as pd
from datetime import datetime as dt
from collections import Counter
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN, SMOTE
import matplotlib.pyplot as plt

class InfoGain:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.features = self.data.columns
        self.entropy = self.calcEntropyOfSet(self.labels)
        self.infoGainTable = pd.DataFrame(data=np.zeros((len(self.features), 2)),
                                          columns=['Info. gain', 'Threshold'],
                                          index=self.features)

    def calcInfoGain(self):
        for feature in self.features:
            gain, igClass, igEntropy = self.calcInfoGainOfFeatureAccountingForContinuous(self.data, self.labels, feature)

            self.infoGainTable.loc[feature, 'Info. gain'] = gain
            self.infoGainTable.loc[feature, 'Threshold'] = igClass
        self.infoGainTable = self.infoGainTable.sort_values(by='Info. gain', ascending=False)

    def calcEntropyOfSet(self, labels):
        nLabels = len(labels)
        uniqueLabels = np.unique(labels)
        labelCounts = self.compUniqueValueCount(labels, uniqueLabels)
        entropy = 0
        for labelIndex in range(len(uniqueLabels)):
            entropy += self.calcEntropy(float(labelCounts[labelIndex])/nLabels)
        return entropy

    def calcInfoGainOfFeature(self, data, labels):
        gain = self.entropy
        nData = len(data)
        valueIndex = 0
        values = np.unique(data)
        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        for value in values:
            dataIndex = 0
            valueOrderedLabels = []
            for datapoint in data:
                if datapoint == value:
                    featureCounts[valueIndex] += 1
                    valueOrderedLabels.append(labels[dataIndex])
                dataIndex += 1
            labelValues = np.unique(valueOrderedLabels)
            classCounts = self.compUniqueValueCount(valueOrderedLabels, labelValues)
            for classIndex in range(len(classCounts)):
                entropy[valueIndex] += self.calcEntropy(float(classCounts[classIndex])/sum(classCounts))

            gain -= float(featureCounts[valueIndex])/nData * entropy[valueIndex]
            valueIndex += 1

        igEntropy, igClass = self.getClassWithGreatestGain(entropy, values)
        return (gain, igClass, igEntropy)

    def calcInfoGainOfFeatureAccountingForContinuous(self, dataSet, labels, feature):
        data = dataSet[feature]
        isContinuous = self.isContinuous(dataSet[feature][0])
        if isContinuous:
            gainTmp = []; igClassTmp = []; igEntropyTmp = []
            for i in range(len(data)):
                data = self.discretise(dataSet[feature], i)
                g, c, e = self.calcInfoGainOfFeature(data, labels)
                gainTmp.append(g)
                igClassTmp.append(c)
                igEntropyTmp.append(e)
            gain, igEntropy = self.getClassWithGreatestGain(gainTmp, igEntropyTmp)
            gain, igClass = self.getClassWithGreatestGain(gainTmp, igClassTmp)
            gain, igClassLabel = self.getClassWithGreatestGain(gainTmp, dataSet[feature])
            igClass = str(igClass) + str(igClassLabel)
        else:
            gain, igClass, igEntropy = self.calcInfoGainOfFeature(data, labels)
        return (gain, igClass, igEntropy)

    def isContinuous(self, value):
        if hasattr(value, 'dtype'):
            isContinuous = np.issubdtype(value.dtype, np.number)
        else:
            try:
                int(value)
                isContinuous = True
            except ValueError:
                isContinuous = False
        return isContinuous

    def getClassWithGreatestGain(self, ofMax, getAtIndex):
        idx = np.argmax(ofMax)
        return (ofMax[idx], getAtIndex[idx])


    def compUniqueValueCount(self, values, uniqueValues):
        valueIndex = 0
        valueCounts = np.zeros(len(uniqueValues))
        for uniqueValue in uniqueValues:
            for value in values:
                if value == uniqueValue:
                    valueCounts[valueIndex] += 1
            valueIndex += 1
        return valueCounts

    def calcEntropy(self, p):
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0

    def discretise(self, data, atIndex):
        discretisedData = np.zeros(len(data), dtype=str)
        discretisedData[data < data[atIndex]] = '<'
        discretisedData[data >= data[atIndex]] = '>='
        return discretisedData

    def __str__(self):
        rendered = '\nInformation Gain Output\n(Set Entropy: {})\n\n'.format(self.entropy)
        rendered += '{}'.format(self.infoGainTable)
        return rendered

######################### Example Data ########################################
# features = ['Deadline', 'Party', 'Lazy']
# deadline = ['Urgent', 'Urgent', 'Near', 'None', 'None', 'None', 'Near', 'Near', 'Near', 'Urgent']
# isParty = ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No']
# amLazy = ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No']
# data = pd.DataFrame({'Deadline':deadline, 'Party':isParty, 'Lazy':amLazy})
# labels = ['Party','Study','Party','Party', 'Pub', 'Party', 'Study', 'Tv', 'Party', 'Study']
# ig = InfoGain(data, labels)
# ig.calcInfoGain()
# print(ig)
###############################################################################


class Rebalance:

    def __init__(self, X, y, log):
        self.log = log
        xIdxs, yIdxs = self.alignXandYIndexes(X.index, y['datetime'])
        self.featureList = X.columns
        self.X = self.formatX(X, xIdxs)
        self.Y = self.formatY(y, yIdxs)
        self.log.emit('N-samples X:{} Y:{}'.format(len(xIdxs), len(yIdxs)), indents=1)
        self.rebalanced = dict()

    def formatY(self, y, yIdxs):
        tmpY = list(y['label'])
        tmpYnumeric = []
        for i in range(0, len(tmpY)):
            c = 1
            if 'minor' in tmpY[i]:
                c = 0
            tmpYnumeric.append(c)
        y = [tmpYnumeric[idx] for idx in yIdxs]
        return y

    def alignXandYIndexes(self, xIdx, yIdx):
        yDates = list(yIdx)
        xIdxs = []
        yIdxs = []
        for i in range(0, len(xIdx)):
            for j in range(0, len(yDates)):
                d = dt.strptime(yDates[j], '%Y-%m-%d %H:%M')
                if xIdx[i].year == d.year and xIdx[i].month == d.month and xIdx[i].day == d.day:
                    yIdxs.append(j)
                    xIdxs.append(i)
        return xIdxs, yIdxs

    def formatX(self, X, xIdxs):
        tmpX = X
        tmpX['startTime'] = self.formatStartTimeIntoDeltaMinutes(X['startTime'])
        tmpX = tmpX.apply(pd.to_numeric, args=('coerce',))
        tmpX = tmpX.replace([np.inf, -np.inf], np.nan)
        tmpX = tmpX.fillna(value=0)
        tmpX = tmpX.reset_index().values.tolist()
        tmpXwithoutIndex = [np.array(sample[1:len(sample)]) for sample in tmpX]
        x = [tmpXwithoutIndex[idx] for idx in xIdxs]
        return x

    def formatStartTimeIntoDeltaMinutes(self, st):
        stdt = []
        for d in st:
            if 'NaN' not in d and type(d) is type(''):
                tmpD = dt.strptime(d, '%Y-%m-%dT%H:%M:%S.000')
                tmpV = tmpD.hour * 60 + tmpD.minute
                stdt.append(tmpV)
            else:
                tmpD = 'NaN'
                stdt.append(tmpD)
        return stdt

    def runADASYN(self):
        ada = ADASYN()
        self.Xadasyn, self.Yadasyn = ada.fit_sample(self.X, self.Y)
        self.rebalanced['ADASYN'] = {'X':self.Xadasyn, 'y': self.Yadasyn, 'f': self.featureList}
        self.log.emit('ADASYN: Original dataset shape {}'.format(Counter(self.Y)), indents=1)
        self.log.emit('ADASYN: Resampled dataset shape {}'.format(Counter(self.Yadasyn)), indents=1)

    def runSMOTE(self):
        try:
            sm = SMOTE(kind='regular')
            self.Xsmote, self.Ysmote = sm.fit_sample(self.X, self.Y)
            self.rebalanced['SMOTE'] = {'X': self.Xsmote, 'y': self.Ysmote, 'f': self.featureList}
            self.log.emit('SMOTE: Original dataset shape {}'.format(Counter(self.Y)), indents=1)
            self.log.emit('SMOTE: Resampled dataset shape {}'.format(Counter(self.Ysmote)), indents=1)
        except ValueError:
            self.log.emit('SMOTE ABORTED: Not enough samples of minor class: {}'.format(Counter(self.Y)), indents=1)

    def plot(self, show=False, save=True, path='', pid=''):
        runAnalyses = list(self.rebalanced.keys())

        if len(runAnalyses) > 0:
            self.log.emit('Plotting {}...'.format(runAnalyses), indents=1)
            pca = PCA(n_components=2)
            f, axes = plt.subplots(1, len(runAnalyses)+1)

            visX = pca.fit_transform(self.X)
            y0 = [i for i in range(0, len(self.Y)) if self.Y[i] == 0]
            y1 = [i for i in range(0, len(self.Y)) if self.Y[i] == 1]
            c0 = axes[0].scatter(visX[y0, 0], visX[y0, 1], label="Minor class",
                             alpha=0.5)
            c1 = axes[0].scatter(visX[y1, 0], visX[y1, 1], label="Major class",
                             alpha=0.5)
            axes[0].set_title('Original set')

            visXada = pca.transform(self.Xadasyn)
            y0 = [i for i in range(0, len(self.Yadasyn)) if self.Yadasyn[i] == 0]
            y1 = [i for i in range(0, len(self.Yadasyn)) if self.Yadasyn[i] == 1]
            axes[1].scatter(visXada[y0, 0], visXada[y0, 1],
                        label="Minor class", alpha=.5)
            axes[1].scatter(visXada[y1, 0], visXada[y1, 1],
                        label="Major class", alpha=.5)
            axes[1].set_title('ADASYN')

            if 'SMOTE' in runAnalyses:
                visXsm = pca.transform(self.Xsmote)
                y0 = [i for i in range(0, len(self.Ysmote)) if self.Ysmote[i] == 0]
                y1 = [i for i in range(0, len(self.Ysmote)) if self.Ysmote[i] == 1]
                axes[2].scatter(visXsm[y0, 0], visXsm[y0, 1],
                            label="Minor class", alpha=.5)
                axes[2].scatter(visXsm[y1, 0], visXsm[y1, 1],
                            label="Major class", alpha=.5)
                axes[2].set_title('SMOTE')

            # make nice plotting
            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
                ax.spines['left'].set_position(('outward', 10))
                ax.spines['bottom'].set_position(('outward', 10))

            plt.figlegend((c0, c1), ('Minor class', 'Major class'), loc='lower center',
                          ncol=2, labelspacing=0.)
            plt.tight_layout(pad=3)
            if show:
                plt.show()
            if save:
                figurePath = '{}{}_PCA_rebalanced_dataset.png'.format(path, pid)
                plt.savefig(figurePath)
        else:
            self.log.emit('Plot ABORTED: No dataset was rebalanced. Try runADASYN() or runSMOTE().', indents=1)


from skfeature.function.information_theoretical_based import MRMR, MIFS


class FeatureSelection:

    def __init__(self, data, log):
        self.log = log
        self.data = data
        self.selectedFeatures = dict()

    def runMIFS(self):
        datasetKeys = self.data.keys()
        for datasetKey in datasetKeys:
            self.log.emit('MIFS feature selection on {} dataset...'.format(datasetKey), indents=1)
            f = self.data[datasetKey]['f']
            X = self.data[datasetKey]['X']
            y = self.data[datasetKey]['y']
            fIdxs = MIFS.mifs(X, y, n_selected_features=10)
            fRank = [f[i] for i in fIdxs]
            self.addToSelectedFeatures('MIFS', datasetKey, fOrig=f, fIdxs=fIdxs, fRank=fRank)

    def runMRMR(self):
        datasetKeys = self.data.keys()
        for datasetKey in datasetKeys:
            self.log.emit('mRMR feature selection on {} dataset...'.format(datasetKey), indents=1)
            f = self.data[datasetKey]['f']
            X = self.data[datasetKey]['X']
            y = self.data[datasetKey]['y']
            fIdxs = MRMR.mrmr(X, y, n_selected_features=10)
            fRank = [f[i] for i in fIdxs]
            self.addToSelectedFeatures('mRMR', datasetKey, fOrig=f, fIdxs=fIdxs, fRank=fRank)

    def addToSelectedFeatures(self, methodName, datasetKey, fOrig, fIdxs, fRank):
        addEntry =  {
            'fOrig': fOrig,
            'fIdxs': fIdxs,
            'fRank': fRank
        }
        try:
            self.selectedFeatures[methodName][datasetKey] = addEntry
        except KeyError:
            newEntry = {datasetKey:addEntry}
            self.selectedFeatures[methodName] = newEntry


    def __str__(self):
        rendered = 'FEATURE SELECTION INFO:\n'
        for methodKey in self.selectedFeatures.keys():
            rendered += '{}:\n'.format(methodKey)
            for datasetKey in self.selectedFeatures[methodKey].keys():
                rendered += '\t{}:\t{}\n'.format(datasetKey, self.selectedFeatures[methodKey][datasetKey]['fRank'][0:10])
        return rendered





