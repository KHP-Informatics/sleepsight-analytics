import numpy as np
import pandas as pd
from analysis import InfoGain
import matplotlib.pyplot as plt
from collections import Counter


class Compliance:
    def __init__(self, aggregates, log):
        self.log = log
        self.aggr = aggregates
        self.dfCount, self.dfDaily = self.aggr.getMissingness()
        self.passiveIdxNames = ['No Missingness', 'Not Charged', 'Not Worn', 'Transmission Failure']
        self.passiveDenominator = 60 * 24 * 56
        self.passiveDenominatorDaily = 60 * 24
        self.activeIdxNames = ['symptom', 'sleep']
        self.activeDenominator = 56
        self.activeDenominatorDaily = 1

    def normaliseMissingness(self):
        self.dfCount.loc[self.passiveIdxNames] = round(self.dfCount.loc[self.passiveIdxNames] / self.passiveDenominator * 100)
        self.dfCount.loc[self.activeIdxNames] = round(self.dfCount.loc[self.activeIdxNames] / self.activeDenominator * 100)
        for key in self.dfDaily.keys():
            self.dfDaily[key][self.passiveIdxNames] = round(
                self.dfDaily[key][self.passiveIdxNames] / self.passiveDenominatorDaily * 100)
            self.dfDaily[key][self.activeIdxNames] = self.dfDaily[key][self.activeIdxNames] / self.activeDenominatorDaily * 100

    def formatComplianceOverTime(self):
        dfCountGroup = self.dfCount.T
        self.dfCountMean = dfCountGroup.mean()
        self.dfCountSEM = dfCountGroup.sem()

        dailyList = [self.dfDaily[key] for key in self.dfDaily.keys()]
        dfConcat = pd.concat((dailyList))
        dfConcatRow = dfConcat.groupby(dfConcat.index)
        self.dfDailyMean = dfConcatRow.mean()
        self.dfDailySEM = dfConcatRow.sem()

    def plot(self, show=False):
        plt.close('all')
        plt.figure(figsize=(9, 9))

        plt.subplot(2, 5, (1, 5))
        self.dfCount.T.plot(kind='bar', ax=plt.gca())
        plt.axhline(y=70, color='black', linewidth=0.7)
        plt.title('A', loc='left', size='16')
        plt.xlabel('Participant')
        plt.ylabel('Data completeness (%)')
        plt.legend([])

        plt.subplot(2, 5, (6,8))
        self.dfDailyMean.plot(yerr=self.dfDailySEM ,ax=plt.gca(), elinewidth=0.6)
        plt.axhline(y=70, color='black', linewidth=0.7)
        plt.title('B', loc='left', size='16')
        plt.xlabel('Study period (in days)')
        plt.ylabel('Data completeness (%)')
        plt.legend(bbox_to_anchor=(0, -0.5, 1., .102), loc=8,
                   ncol=3, mode="expand", borderpad=0.6, frameon=True)

        plt.subplot(2, 5, (9,10))
        self.dfCountMean.plot(kind='bar',yerr=self.dfCountSEM)
        plt.axhline(y=70, color='black', linewidth=0.7)
        plt.xticks(fontsize='9', rotation=90)
        plt.title('C', loc='left', size='16')
        plt.xlabel('Missingness categories')
        plt.ylabel('Data completeness (%)')
        plt.tight_layout()

        if show:
            plt.show()
        return plt

    def generateFigure(self, show=False, save=True):
        self.log.emit('Generating figure...', indents=1)
        self.normaliseMissingness()
        self.formatComplianceOverTime()
        plt = self.plot()
        if save:
            self.log.emit('Exporting figure...', indents=1)
            path = self.aggr.pathPlot + 'Compliance.png'
            plt.savefig(path, dpi=600)
        if show:
            plt.show()

    def exportLatexTable(self, save=True):
        self.log.emit('Exporting table...', indents=1)
        tmpTable = pd.concat((self.dfCountMean, self.dfCountSEM), axis=1)
        tmpTable.columns = ['Mean (%)', 'SD (%)']
        tmpTable.index = [str.capitalize(idx) for idx in tmpTable.index]
        tmpTable['Mean (%)'] = tmpTable['Mean (%)'].astype(int)
        tmpTable['SD (%)'] = tmpTable['SD (%)'].round(1)
        latextTable = tmpTable.to_latex()
        if save:
            path = self.aggr.pathPlot + 'DataCompliance.tex'
            f = open(path, 'w')
            f.write(latextTable)
            f.close()


class InfoGainTable:

    def __init__(self, infoTable, labels, log):
        self.log = log
        self.labelsOfLabels = labels.keys()
        self.labels = labels
        self.info = infoTable
        self.features = self.info.columns
        self.normalisedColumns = ['Info. gain (%)', 'Threshold']
        self.entropies = []

    def run(self):
        self.log.emit('Generating...', indents=1)
        resultTables = []
        for labelOfLabels in self.labelsOfLabels:
            labels = self.discretiseLabels(self.labels[labelOfLabels])
            ig = InfoGain(self.info, labels)
            ig.calcInfoGain()
            resultNormalise = self.normaliseInfoGain(ig.infoGainTable, ig.entropy)
            columnsMultiIndex = [(labelOfLabels, self.normalisedColumns[0]), (labelOfLabels, self.normalisedColumns[1])]
            resultNormalise.columns = pd.MultiIndex.from_tuples(columnsMultiIndex)
            resultTables.append(resultNormalise)
            self.entropies.append({labelOfLabels:ig.entropy})
        self.outputTable = pd.concat(resultTables, axis=1)

    def discretiseLabels(self, rawLabels):
        classifiedLabels = ['Non-compliance', 'Reduced compliance', 'Sufficient compliance', 'High compliance']
        labels = []
        for i in range(len(rawLabels)):
            if rawLabels[i] >= 85:
                labels.append(classifiedLabels[3])
            elif rawLabels[i] >= 70:
                labels.append(classifiedLabels[2])
            elif rawLabels[i] >= 50:
                labels.append(classifiedLabels[1])
            else:
                labels.append(classifiedLabels[0])
        return labels

    def normaliseInfoGain(self, result, entropy):
        result['Info. gain'] = result['Info. gain']/entropy*100
        result = result.fillna(0)
        result = result.round({'Info. gain':1})
        result.columns = self.normalisedColumns
        return result

    def exportLatexTable(self, plotPath, orderedBy, save=True):
        self.log.emit('Exporting table...', indents=1)
        tmpTable = self.outputTable
        tmpTable.index = self.formatFeatures(tmpTable.index)
        tmpTable = tmpTable.sort_index(level=1)
        tmpTable = tmpTable.sort_values([(orderedBy, self.normalisedColumns[0])], ascending=False)
        latexTable = tmpTable.to_latex()
        if save:
            path = plotPath + 'DataComplianceInfoGain.tex'
            f = open(path, 'w')
            f.write(latexTable)
            f.close()

    def formatFeatures(self, features):
        formated = []
        for feature in features:
            tmp = feature.replace('.', ' ')
            tmp = tmp.capitalize()
            if tmp in 'Durationillness':
                tmp = 'Duration of illness'
            formated.append(tmp)
        return formated

    def __str__(self):
        rendered = 'Information Gain for {}\n\n'.format(self.labelsOfLabels)
        rendered += 'Entropies\n{}\n\n'.format(pd.DataFrame(self.entropies))
        rendered += 'Compliance Info Gain\n{}\n\n'.format(self.outputTable)
        return rendered

class StationaryTable:

    def __init__(self, aggr, log):
        self.log = log
        self.aggr = aggr
        self.outputPath = aggr.pathPlot

    def run(self):
        self.log.emit('Generating...', indents=1)
        statsExtract = []
        for participant in self.aggr.aggregates:
            statsExtractTmp = []
            sleepsightId = 'Participant ' + str(participant.id)
            stats = participant.stationaryPassiveStats
            statsIrL = list(stats['I(r)'].values)
            statsPL = list(stats['p-value'].values)
            columns = list(stats['Feature'])
            statsExtractTmp.append(statsIrL)
            statsExtractTmp.append(statsPL)
            statsExtractTmpTable = pd.DataFrame(statsExtractTmp, columns=columns)
            statsExtractTmpTable.index = ['I(r)', 'p-value']
            statsExtractTmpTableT = statsExtractTmpTable.T
            decimals = pd.Series([0], index=['I(r)'])
            statsExtractTmpTableT = statsExtractTmpTableT.round(decimals)
            columnsMultiIndex = [(sleepsightId, 'I(r)'), (sleepsightId, 'p-value')]
            statsExtractTmpTableT.columns = pd.MultiIndex.from_tuples(columnsMultiIndex)
            statsExtract.append(statsExtractTmpTableT)
        self.outputTable = pd.concat(statsExtract, axis=1)
        self.outputTable = self.outputTable.dropna(axis=0)

    def exportLatexTable(self, show=False, save=True):
        self.log.emit('Exporting table...', indents=1)
        formatedTable = self.outputTable
        formatedTable.index = self.formatIndex(self.outputTable.index)
        latexTable = formatedTable.to_latex()
        if show:
            print(latexTable)
        if save:
            path = self.outputPath + 'DataStationarityStats.tex'
            f = open(path, 'w')
            f.write(latexTable)
            f.close()

    def formatIndex(self, index):
        formatedIndex = []
        for val in index:
            if val not in ['FAM', 'LAM','VAM']:
                val = val.replace('_', ' ')
                val = val.capitalize()
            formatedIndex.append(val)
        return formatedIndex


class DiscretisationTable:

    def __init__(self, aggr, log):
        self.log = log
        self.aggr = aggr

    def run(self):
        self.log.emit('Generating...', indents=1)
        sampleSet = []
        id = 1
        for aggregate in self.aggr.aggregates:
            m = round(np.mean(aggregate.activeDataSymptom['total']), 1)
            sd = round(np.std(aggregate.activeDataSymptom['total']), 2)
            major = len(aggregate.activeDataSymptom[aggregate.activeDataSymptom['total'] < (m + sd)])
            minor = len(aggregate.activeDataSymptom[aggregate.activeDataSymptom['total'] >= (m + sd)])
            majorPercent = int(round((major / (major + minor)) * 100, 0))
            minorPercent = int(round((minor / (major + minor)) * 100, 0))
            sampleSet.append(
                [str(id), str(m), str(sd), '{} ({})'.format(major, majorPercent), '{} ({})'.format(minor, minorPercent)])
            id += 1
        self.sampleTable = pd.DataFrame(sampleSet, columns=['Participant', 'Mean', 'SD', 'Major class (%)', 'Minor class (%)'])

    def exportLatexTable(self, show=False, save=True):
        self.log.emit('Exporting table...', indents=1)
        latexTable = self.sampleTable.to_latex(index=False)
        if save:
            path = self.aggr.pathPlot + 'DataSampleSymptomClass.tex'
            f = open(path, 'w')
            f.write(latexTable)
            f.close()

class PeriodictyTable:

    def __init__(self, aggr, log):
        self.log = log
        self.aggr = aggr

    def run(self):
        self.log.emit('Generating...', indents=1)
        periodicityStats = []
        periodicityStatsIndex = []
        acfPeakStatsCols = self.aggr.aggregates[0].acfPeakStats.keys()
        acfPeakStatsCols = self.removeCol(acfPeakStatsCols, 'intra_calories')
        for participant in self.aggr.aggregates:
            participantId = participant.id
            pFormat = []
            pMean = []
            for key in acfPeakStatsCols:
                mean = np.round(participant.acfPeakStats[key]['mean']/60, 1)
                std = np.round(participant.acfPeakStats[key]['std']/60, 2)
                pFormat.append('{} ({})'.format(mean, std))
                pMean.append(mean)
            pMeanTotal = '{} ({})'.format(np.mean(pMean).round(1), np.std(pMean).round(2))
            pFormat.append(pMeanTotal)
            periodicityStats.append(pFormat)
            periodicityStatsIndex.append(participantId)
        cols = acfPeakStatsCols + ['Period µ']
        self.outputTable = pd.DataFrame(periodicityStats, columns=self.formatIndex(cols))
        self.outputTable.index = periodicityStatsIndex

    def removeCol(self, l, v):
        ll = list(l)
        vIdxs = [i for i in range(len(ll)) if v in ll[i]]
        [ll.pop(vIdx) for vIdx in vIdxs]
        return ll

    def formatIndex(self, index):
        formatedIndex = []
        for val in index:
            if val not in ['FAM', 'LAM','VAM']:
                val = val.replace('_', ' ')
                val = val.capitalize()
            val = '{} (SD)'.format(val)
            formatedIndex.append(val)
        return formatedIndex

    def exportLatexTable(self, show=False, save=True, summary=False):
        self.log.emit('Exporting table...', indents=1)
        latexTable = self.outputTable.to_latex(index=False)
        fileName = 'DataPeriodicityStats.tex'
        if summary:
            latexTable = self.outputTable[['Period µ (SD)']].to_latex(index=True)
            fileName = 'DataPeriodicityStatsSummary.tex'
        if show:
            print(latexTable)
        if save:
            path = self.aggr.pathPlot + fileName
            f = open(path, 'w')
            f.write(latexTable)
            f.close()

class DelayEval:

    def __init__(self, aggr, log):
        self.log = log
        self.aggr = aggr

    def test(self):
        print('test')

class FeatureSelectionEval:

    def __init__(self, aggr, log):
        self.log = log
        self.aggr = aggr
        self.fComb = self.generateCombinedFeatureTable()
        self.histogramsFs = []

    def generateCombinedFeatureTable(self):
        fsMethodsComb = dict()
        for p in self.aggr.aggregates:
            features = p.nonParametricFeaturesSelected
            fsMethodsComb[p.id] = self.extractRankedFeatures(features)
        return fsMethodsComb

    def extractRankedFeatures(self, features):
        featuresRanked = dict()
        for fsMethod in features:
            for dataset in features[fsMethod]:
                try:
                    featuresRanked[fsMethod][dataset] = features[fsMethod][dataset]['fRank']
                except KeyError:
                    newEntry = {
                        dataset: features[fsMethod][dataset]['fRank']
                    }
                    featuresRanked[fsMethod] = newEntry
        return featuresRanked

    def generateHistogramForNTopFeatures(self, nFeatures):
        mComb = self.reformatFCombToNFeaturesComb(nFeatures)
        for mFs in mComb.keys():
            letter_counts = Counter(mComb[mFs])
            df = pd.DataFrame.from_dict(letter_counts, orient='index')
            df.columns = [mFs]
            dfSorted = df.sort_values(by=[mFs], ascending=False)
            self.histogramsFs.append(dfSorted)


    def reformatFCombToNFeaturesComb(self, nFeatures):
        reformatedFComb = dict()
        for p in self.fComb.keys():
            for fsMethod in self.fComb[p].keys():
                for dataset in self.fComb[p][fsMethod]:
                    tmpF = [self.fComb[p][fsMethod][dataset][i] for i in range(0, nFeatures)]
                    try:
                        reformatedFComb['{}-{}'.format(fsMethod, dataset)] += tmpF
                    except KeyError:
                        reformatedFComb['{}-{}'.format(fsMethod, dataset)] = tmpF
        return reformatedFComb

    def generateFigure(self, show=False, save=True):
        self.log.emit('Generating figure...', indents=1)
        plt = self.plot()
        if save:
            self.log.emit('Exporting figure...', indents=1)
            path = self.aggr.pathPlot + 'FeatureSelection.png'
            plt.savefig(path, dpi=600)
        if show:
            plt.show()

    def plot(self, show=False):
        maxCounts = []
        for df in self.histogramsFs:
            maxCounts.append(np.max(df))
        yMax = np.max(list(maxCounts))

        plt.close('all')
        plt.figure(figsize=(10, 9))
        for i in range(0, len(self.histogramsFs)):
            plt.subplot(2, len(self.histogramsFs)/2, (i+1))
            self.histogramsFs[i].plot(kind='bar', ax=plt.gca())
            plt.ylim(ymax=yMax)
            plt.title('{}'.format(self.histogramsFs[i].columns[0]), loc='left', size='14')
            plt.xlabel('Features')
            plt.ylabel('Count')
            plt.legend([])
        plt.tight_layout()

        if show:
            plt.show()
        return plt


class NonParametricSVMEval:

    def __init__(self, aggr, log):
        self.log = log
        self.aggr = aggr
        self.summary = pd.DataFrame()

    def summarise(self):
        self.log.emit('Summarising SVM results...', indents=1)
        for p in self.aggr.aggregates:
            try:
                row = self.createParticipantSVMSummary(p.nonParametricResults)
                multiIndex = [(p.id, indexName) for indexName in row.index]
                row.index = pd.MultiIndex.from_tuples(multiIndex)
                self.summary = pd.concat([self.summary, row], axis=0)

            except AttributeError:
                self.log.emit('Participant {} has not had its SVM fitted yet.'.format(p.id), indents=1)


    def logClassificationReports(self, results):
        for m in results:
            for d in results[m]:
                self.log.emit('{}-{}:'.format(m, d), indents=1)
                self.log.emit(results[m][d]['classificationReport'], indents=1)

    def createParticipantSVMSummary(self, results):
        participantRow = dict()
        for m in results.keys():
            for d in results[m].keys():
                index = ['Precision', 'Recall', 'F1 Score']
                precision = self.computePrecision(results[m][d]['confusionMatrix'])
                recall = self.computeRecall(results[m][d]['confusionMatrix'])
                f1Score = self.computeF1Score(precision, recall)
                df = pd.DataFrame([precision, recall, f1Score], index=index, columns=[m])
                try:
                    participantRow[d] = pd.concat([participantRow[d], df], axis=1)
                except KeyError:
                    participantRow[d] = df

        for d in participantRow.keys():
            multiIndex = [(d, columnName) for columnName in participantRow[d].columns]
            participantRow[d].columns = pd.MultiIndex.from_tuples(multiIndex)

        datasets = list(participantRow.keys())
        datasetsSorted = np.sort(datasets)
        pRowList = [participantRow[d] for d in datasetsSorted]
        participantRowConcat = pd.concat(pRowList, axis=1)

        return participantRowConcat

    def computePrecision(self, confusionMatrix):
        tpC0 = confusionMatrix[0][0]
        fpC0 = confusionMatrix[1][0]
        p0 = tpC0 / (tpC0+fpC0)
        tpC1 = confusionMatrix[1][1]
        fpC1 = confusionMatrix[0][1]
        p1 = tpC1 / (tpC1+fpC1)
        p = np.mean([p0, p1])
        p = self.rescueScore(p)
        return p

    def computeRecall(self, confusionMatrix):
        tpC0 = confusionMatrix[0][0]
        tnC0 = confusionMatrix[0][1]
        r0 = tpC0 / (tpC0 + tnC0)
        tpC1 = confusionMatrix[1][1]
        tnC1 = confusionMatrix[1][0]
        r1 = tpC1 / (tpC1 + tnC1)
        r = np.mean([r0, r1])
        r = self.rescueScore(r)
        return r

    def computeF1Score(self, precision, recall):
        f = 2*precision*recall / (precision+recall)
        return f

    def rescueScore(self, score):
        if score < 0.5:
            return 1 - score
        return score

    def exportLatexTable(self, show=False, save=True):
        self.log.emit('Exporting table...', indents=1)
        outputTable = self.summary.round(2)
        latexTable = outputTable.to_latex(index=True, na_rep='-')
        fileName = 'DataNonParametric-SVM-Summary.tex'
        if show:
            print(latexTable)
        if save:
            path = self.aggr.pathPlot + fileName
            f = open(path, 'w')
            f.write(latexTable)
            f.close()




