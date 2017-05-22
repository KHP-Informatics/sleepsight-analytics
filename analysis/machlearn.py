# !/bin/python3

import numpy as np
import pandas as pd

class InfoGain:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.features = self.data.columns
        self.entropy = self.calcEntropyOfSet(self.labels)
        self.infoGainTable = pd.DataFrame(data=np.zeros(len(self.features)),
                                          columns=['Information Gain'],
                                          index=self.features)

    def calcInfoGain(self):
        for feature in self.features:
            gain = self.calcInfoGainOfFeature(self.data, self.labels, feature)
            self.infoGainTable['Information Gain'][feature] = gain
        self.infoGainTable = self.infoGainTable.sort_values(by='Information Gain', ascending=False)

    def calcEntropyOfSet(self, labels):
        nLabels = len(labels)
        uniqueLabels = np.unique(labels)
        labelCounts = self.compUniqueValueCount(labels, uniqueLabels)
        entropy = 0
        for labelIndex in range(len(uniqueLabels)):
            entropy += self.calcEntropy(float(labelCounts[labelIndex])/nLabels)
        return entropy

    def calcInfoGainOfFeature(self, data, labels, feature):
        gain = self.entropy
        nData = len(data)
        valueIndex = 0
        values = np.unique(data[feature])
        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        for value in values:
            dataIndex = 0
            valueOrderedLabels = []
            for datapoint in data[feature]:
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
        return gain

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