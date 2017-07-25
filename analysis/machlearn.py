# !/bin/python3

import numpy as np
import pandas as pd

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

#from collections import Counter
#from sklearn.datasets import make_classification
#from imblearn.over_sampling import ADASYN
#
#class Rebalance:
#
#    def __init__(self, X, y):
#        self.pdX = X
#        self.pdY = y
#        print(y)
#
#    def test(self):
#        X, y = make_classification(n_classes=2, class_sep=2, weights = [0.1, 0.9], n_informative = 3, n_redundant = 1,
#                                   flip_y = 0, n_features = 20, n_clusters_per_class = 1, n_samples = 1000, random_state = 10)
#        print('Original dataset shape {}'.format(Counter(y)))
#        ada = ADASYN(random_state=42)
#        X_res, y_res = ada.fit_sample(X, y)
#        print('Resampled dataset shape {}'.format(Counter(y_res)))