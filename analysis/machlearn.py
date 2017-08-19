# !/bin/python3

import numpy as np
import pandas as pd
from datetime import datetime as dt
from collections import Counter
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN, SMOTE
import matplotlib.pyplot as plt
from skfeature.function.information_theoretical_based import MRMR, MIFS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import GPy


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
            if type(tmpY[i]) is type(' ') and 'minor' in tmpY[i]:
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
            if type(d) is type('') and 'NaN' not in d:
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


class SVMMLWrapper:

    def __init__(self, data, features, log):
        self.data = data
        self.features = features
        self.log = log
        self.results = dict()

    def runSVM(self, nFeatures, nCPUs=2):
        for fsMethod in self.features:
            for dataset in self.data:
                self.log.emit('Fitting SVM on {}-{}...'.format(fsMethod, dataset), indents=1)
                f = self.features[fsMethod][dataset]['fIdxs'][0:nFeatures]
                X = self.extractNFeatures(self.data[dataset]['X'], f)
                y = self.data[dataset]['y']
                result = self.fitSVM(X, y, nCPUs=nCPUs)
                self.addResults(result, fsMethod=fsMethod, dataset=dataset)

    def fitSVM(self, X, y, nCPUs=2, lossFunctions=['recall']):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0)
        #gamma_range = np.logspace(-9, 3, 13)
        optimisationParameters = {'kernel': ['linear'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

        scores = lossFunctions
        for score in scores:
            self.log.emit("Tuning hyper-parameters for %s..." % score, indents=2)

            clf = GridSearchCV(SVC(C=1), optimisationParameters, cv=3, n_jobs=nCPUs,
                               scoring=score)
            clf.fit(X_train, y_train)

            self.log.emit("Best parameters set found on development set:", indents=2)
            self.log.emit(clf.best_params_, indents=2)

            self.log.emit("Grid scores on development set:", indents=2)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                self.log.emit("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), indents=2)

            self.log.emit("Classification report:")
            y_true, y_pred = y_test, clf.predict(X_test)
            cReport = classification_report(y_true, y_pred)
            self.log.emit(cReport, indents=2)
            cMtrx = confusion_matrix(y_true, y_pred)
            self.log.emit(cMtrx, indents=2)

            return self.formatExports(model=clf, confusionMatrix=cMtrx, classificationReport=cReport)

    def extractNFeatures(self, X, fIdxs):
        Xextract = []
        for x in X:
            xFm = [x[i] for i in fIdxs]
            Xextract.append(xFm)
        return Xextract

    def formatExports(self, model, confusionMatrix, classificationReport):
        exports = {
            'model': model,
            'confusionMatrix': confusionMatrix,
            'classificationReport': classificationReport
        }
        return exports

    def addResults(self, result, fsMethod, dataset):
        try:
            self.results[fsMethod][dataset] = result
        except KeyError:
            newFsMethodResult = {
                dataset: result
            }
            self.results[fsMethod] = newFsMethodResult

from sklearn.gaussian_process import GaussianProcess
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit

class GPMLWrapper:

    def __init__(self, gpm, plotPath, log, decisionBoundary=0.5):
        self.log = log
        self.log.emit('Setting up GPMLWrapper ...')
        self.gpm = gpm
        self.path = plotPath
        self.decisionBoundary = decisionBoundary

    def prepareGP(self, feature):
        self.log.emit('Extracting features...', indents=1)
        self.feature = feature
        self.Y = np.transpose(self.getY(self.feature, label='all'))
        self.T = self.getLabels(label='all')

    def fit(self, nSplits=1):
        predictions = []
        splits = self.splitSamples(self.Y, self.T, nSplits=nSplits)
        for i in range(0, nSplits):
            kernel = GPy.kern.GridRBF(input_dim=1439)
            self.m = GPy.models.GPClassification(X=self.Y, Y=self.T, kernel=kernel)
            self.m.optimize(messages=True, max_iters=200)
            for idx in range(0, len(splits['Xtest'][i])):
                posterior = self.m.predict(Xnew=np.array([splits['Xtest'][i][idx]]))
                predictions.append(self.formatPosterior(posterior[0][0]))
        targets = []
        for i in range(0, nSplits):
            targets += list(np.transpose(splits['Ytest'][i])[0])
        self.confusionMtrxFit = confusion_matrix(targets, predictions, [1, 0])
        self.log.emit("Fit Classification Report {}:\n{}".format(
            self.feature, metrics.classification_report(targets, predictions)), indents=1)


    def simulate(self):
        self.log.emit('Begin simulation...', indents=1)
        T = self.getLabels(label='all')
        Tts = list([T[i] for i in range(1, len(T))])
        Y = np.transpose(self.getY(self.feature, label='all'))
        ts = [Y[i] for i in range(1, len(Y))]
        inputVector = Y[0]

        output = {'x':[], 'y':[]}
        target = {'x':[], 'y':[]}
        xIdx = 0
        target['x'].append(0)
        target['y'].append(T[0][0])
        for i in range(0, len(ts)):
            for j in range(0, len(ts[i])):
                inputVector[j] = ts[i][j]
                posterior = self.m.predict(Xnew=np.array([inputVector]))
                output['x'].append(xIdx)
                output['y'].append(self.formatPosterior(posterior[0][0]))
                xIdx += 1
            target['x'].append(xIdx)
            target['y'].append(Tts[i][0])

        plt.plot(output['x'], output['y'])
        plt.plot(target['x'], target['y'], 'ro')
        plt.savefig(self.path + 'GP_sim')

    def formatPosterior(self, val):
        if val == self.decisionBoundary:
            return val
        if val < self.decisionBoundary:
            return 0
        return 1

    def getX(self):
        rangeIndex = self.gpm.indexDict[0]['indexEnd'] - self.gpm.indexDict[0]['indexStart']
        X = np.array([np.array([i]) for i in range(0, rangeIndex)])
        return X

    def getY(self, feature, label='all'):
        samples = self.gpm.getSamplesOfClassT(label)
        rangeIndex = samples[0]['indexEnd'] - samples[0]['indexStart']
        yData = self.gpm.passiveData[feature]
        Y = []
        for i in range(0, rangeIndex):
            dataPointsAtX = []
            for sample in samples:
                dataPoint = yData[yData.index[(sample['indexStart'] + i)]]
                dataPointsAtX.append(dataPoint)
            Y.append(np.array(dataPointsAtX))
        return np.array(Y)

    def getYSampled(self, feature, label='all'):
        samples = self.gpm.getSamplesOfClassT(label)
        rangeIndex = samples[0]['indexEnd'] - samples[0]['indexStart']
        yData = self.gpm.passiveData[feature]
        Y = []
        for i in range(0, rangeIndex):
            dataPointsAtX = []
            for sample in samples:
                dataPoint = yData[yData.index[(sample['indexStart'] + i)]]
                dataPointsAtX.append(dataPoint)
            randomSample = np.random.choice(dataPointsAtX)
            Y.append(np.array([randomSample]))
        return np.array(Y)

    def getLabels(self, label='all'):
        samples = self.gpm.getSamplesOfClassT(label)
        labels = []
        for sample in samples:
            if sample['y'] in 'major':
                labels.append([0])
            else:
                labels.append([1])
        return np.array(labels)

    def splitSamples(self, X, Y, nSplits=1):
        s = StratifiedShuffleSplit(n_splits=nSplits, test_size=0.3)
        splits = {'Xtrain':[], 'Xtest':[], 'Ytrain':[], 'Ytest':[]}
        for trainIdx, testIdx in s.split(X, Y):
            splits['Xtrain'].append(X[trainIdx])
            splits['Xtest'].append(X[testIdx])
            splits['Ytrain'].append(Y[trainIdx])
            splits['Ytest'].append(Y[testIdx])
        return splits



