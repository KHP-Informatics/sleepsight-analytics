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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit


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


class GPMLWrapper:

    @property
    def gpResults(self):
        results = {
            'gpcs': self.gpcs,
            'gprAUCs': self.gprAUCs,
            'varXidxs': self.varXidxs,
            'varsUpper': self.varsUpper,
            'varsLower': self.varsLower,
            'confusionMatrices': self.confusionMatrices
        }
        return results

    @gpResults.setter
    def gpResults(self, results):
        self.gpcs = results['gpcs']
        self.gprAUCs = results['gprAUCs']
        self.varXidxs = results['varXidxs']
        self.varsUpper = results['varsUpper']
        self.varsLower = results['varsLower']
        self.confusionMatrices = results['confusionMatrices']

    @property
    def simResults(self):
        return self.__simResults

    @simResults.setter
    def simResults(self, results):
        self.__simResults = results

    def __init__(self, gpm, plotPath, log, decisionBoundary=0.5):
        self.log = log
        self.log.emit('Setting up GPMLWrapper ...')
        self.gpm = gpm
        self.path = plotPath
        self.decisionBoundary = decisionBoundary
        self.features = self.gpm.xFeatures
        self.T = self.getLabels(label='all')

    def fitHSGP(self, kFold=9):

        self.gpcs = []
        self.varXidxs = []
        self.gprAUCs = []
        self.varsUpper = []
        self.varsLower = []
        self.confusionMatrices = []

        kFolds = self.genFolds(kFold)

        for k in range(0, len(kFolds)):
            self.log.emit('Fold {}: {}'.format(k, kFolds[k]))
            gpcsTmp = {}
            gprAUCsTmp = []
            varsUpperTmp = []
            varsLowerTmp = []
            cmTmp = {}

            for f in range(0, len(self.features)):
                self.log.emit('Fitting GP-Regression on {}...'.format(self.features[f]), indents=1)
                T = self.genTrainSet(kFolds[k], self.T)
                Y, std = self.getYSampled(self.features[f], label='major')
                Y, std = self.genTrainSet(kFolds[k], Y), self.genTrainSet(kFolds[k], std)
                X = self.getX()
                X = self.genTrainSet(kFolds[k], X)

                m = GPy.models.GPRegression(X=X, Y=Y)
                try:
                    m.optimize(messages=True, max_iters=200)
                except np.linalg.linalg.LinAlgError:
                    self.log.emit('[WARN] LinAlgError: not positive definite, even with jitter. Continuing with non-optmised mean function.', indents=1)
                muX, muY = self.getGpMean(model=m, period=len(X)-1)

                s = GPy.models.GPRegression(X=Y, Y=std)
                try:
                    s.optimize(messages=True, max_iters=200)
                except np.linalg.linalg.LinAlgError:
                    self.log.emit(
                        '[WARN] LinAlgError: not positive definite, even with jitter. Continuing with non-optmised mean function.',
                        indents=1)

                stdTrim = []
                for x in muX:
                    mPredicted = m.predict(np.array([[x]]))
                    sPredicted = s.predict(mPredicted[0])
                    stdTrim.append(sPredicted[0][0][0])

                varUpper = [muY[i] + stdTrim[i]*1.96 for i in range(0, len(muY))]
                varLower = [muY[i] - stdTrim[i]*1.96 for i in range(0, len(muY))]
                varsUpperTmp.append(varUpper)
                varsLowerTmp.append(varLower)

                samples = np.transpose(self.getY(feature=self.features[f], label='all'))
                samplesTrain = self.genTrainSet(kFolds[k], samples)
                gprAUC = self.computeAUCs(samples=samplesTrain, varX=muX, varUpper=varUpper, varLower=varLower)
                gprAUCsTmp.append(gprAUC)
                self.log.emit('Fitting GP-Classification...', indents=1)

                aucInput = np.array([[auc] for auc in gprAUC])
                gpc = GaussianProcessClassifier()
                gpc.fit(X=aucInput, y=T)
                gpcsTmp[f] = gpc

                Ttest = self.genTestSet(kFolds[k], self.T)
                samplesTest = self.genTestSet(kFolds[k], samples)
                gprAUCsTest = self.computeAUCs(samplesTest, muX, varUpper, varLower)
                aucInputTest = np.array([[auc] for auc in gprAUCsTest])
                predictions = []
                for i in range(0, len(aucInputTest)):
                    preProb = gpc.predict_proba([aucInputTest[i]])
                    predictions.append(self.formatPosterior(preProb[0][1]))
                    print('{} {}'.format(Ttest[i], preProb))
                cmTmp[self.features[f]] = confusion_matrix(Ttest, predictions)
                self.log.emit("Fit Classification Report SK:\n{}".format(metrics.classification_report(Ttest, predictions)),
                              indents=1)
            self.gpcs.append(gpcsTmp)
            self.gprAUCs.append(gprAUCsTmp)
            self.varXidxs.append(muX)
            self.varsUpper.append(varsUpperTmp)
            self.varsLower.append(varsLowerTmp)
            self.confusionMatrices.append(cmTmp)

    def genFolds(self, kFolds, enableAll=True):
        sampleSize = len(self.T)
        fold = int(sampleSize/kFolds)
        folds = []
        if enableAll:
            f = {'start': -2, 'end': -1}
            folds.append(f)
        for i in range(0, kFolds):
            f = {'start': i*fold, 'end': (i+1)*fold-1}
            folds.append(f)
        return folds

    def genTrainSet(self, fold, samples):
        samplesTrain = []
        for i in range(0, len(samples)):
            if i < fold['start'] or i > fold['end']:
                samplesTrain.append(samples[i])
        return np.array(samplesTrain)

    def genTestSet(self, fold, samples):
        samplesTest = []
        for i in range(0, len(samples)):
            if i >= fold['start'] or i <= fold['end']:
                samplesTest.append(samples[i])
            if fold['start'] == -2:
                samplesTest.append(samples[i])
        return np.array(samplesTest)

    def getGpMean(self, model, period=1438):
        model.plot_mean()
        ax = plt.gca()
        line = ax.lines[0]
        mux = np.round(line.get_xdata())
        muxTrim = mux[(mux >= 0) & (mux <= period)]
        muy = line.get_ydata()
        muyTrim = muy[(mux >= 0) & (mux <= period)]
        muxFull = np.concatenate(([0], muxTrim, [period]))
        muyFull = np.concatenate(([muyTrim[0]], muyTrim, [muyTrim[(len(muyTrim) - 1)]]))
        return (muxFull, muyFull)

    def computeAUCs(self, samples, varX, varUpper, varLower):
        gprAUC = []
        for i in range(0, len(samples)):
            sampleTrim = [samples[i][int(x)] for x in varX]
            aucUpper = self.computeAUC(Y1=varUpper, Y2=sampleTrim, subtraction='Y2-Y1')
            aucLower = self.computeAUC(Y1=varLower, Y2=sampleTrim, subtraction='Y1-Y2')
            auc = aucUpper + aucLower
            gprAUC.append(auc)
        return gprAUC

    def computeAUC(self, Y1, Y2, dx=10, subtraction='Y1-Y2'):
        auc = []
        for i in range(0, len(Y1)):
            if subtraction in 'Y1-Y2' and Y1[i] > Y2[i]:
                subArea = (Y1[i] - Y2[i]) * dx
                auc.append(subArea)
            if subtraction in 'Y2-Y1' and Y2[i] > Y1[i]:
                subArea = (Y2[i] - Y1[i]) * dx
                auc.append(subArea)
            if subtraction not in 'Y1-Y2' and subtraction not in 'Y2-Y1':
                self.log.emit('[WARN] Wrong argument for "computeAUC": {}'.format(subtraction), indents=1)
        aucSum = np.sum(auc)
        return aucSum

    def simulate(self, participantId):
        self.log.emit('Begin simulation...', indents=1)

        T = self.getLabels(label='all')
        self.outputs = {'x': [], 'y': [], 'p': []}
        self.targets = {'x': [], 'y': []}

        for f in range(0, len(self.features)):
            self.log.emit('Simulating {}...'.format(self.features[f]), indents=1)
            Y = np.transpose(self.getY(self.features[f], label='all'))
            inputVectors = Y[0]

            output = {'x': [], 'y': [], 'p':[]}
            target = {'x': [], 'y': []}
            xIdx = 0
            target['x'].append(0)
            target['y'].append(T[0][0])
            for i in range(0, len(Y)):
                for j in range(0, len(Y[i])):
                    aucVector = []
                    inputVectors[j] = Y[i][j]
                    sampleTrim = [inputVectors[int(x)] for x in self.varXidx]
                    # GP-regression outputs
                    aucUpper = self.computeAUC(Y1=self.varsUpper[f], Y2=sampleTrim, subtraction='Y2-Y1')
                    aucLower = self.computeAUC(Y1=self.varsLower[f], Y2=sampleTrim, subtraction='Y1-Y2')
                    auc = aucUpper + aucLower
                    aucVector.append(auc)
                    # GP-classification output
                    posterior = self.gpcs[f].predict_proba(np.array([aucVector]))
                    output['x'].append(xIdx)
                    output['p'].append(posterior[0][1])
                    output['y'].append(self.formatPosterior(posterior[0][1]))
                    xIdx += 1
                target['x'].append(xIdx)
                target['y'].append(self.T[i][0])

            self.outputs['x'].append(output['x'])
            self.outputs['y'].append(output['y'])
            self.outputs['p'].append(output['p'])
            self.targets['x'] = target['x']
            self.targets['y'] = target['y']

        self.outputs['x'] = self.outputs['x'][0]
        self.outputs['yMean'] = np.mean(self.outputs['y'], axis=0)
        self.outputs['pMean'] = np.mean(self.outputs['p'], axis=0)

        self.simResults = {
            'outputs': self.outputs,
            'targets': self.targets,
            'confusionMatrices': self.confusionMatrices
        }
        plt.figure()
        plt.plot(self.outputs['x'], self.outputs['yMean'])
        plt.plot(self.outputs['x'], self.outputs['pMean'])
        plt.plot(self.targets['x'], self.targets['y'], 'ro')
        plt.savefig(self.path + '{}_GP_sim'.format(participantId))

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
        Ystd = []
        for i in range(0, rangeIndex):
            dataPointsAtX = []
            for sample in samples:
                dataPoint = yData[yData.index[(sample['indexStart'] + i)]]
                dataPointsAtX.append(dataPoint)
            randomSample = np.random.choice(dataPointsAtX)
            Y.append(np.array([randomSample]))
            Ystd.append(np.array([np.std(dataPointsAtX)]))
        return (np.array(Y), np.array(Ystd))

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



