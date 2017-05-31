import numpy as np
import pandas as pd
from analysis import InfoGain
import matplotlib.pyplot as plt


class Compliance:
    def __init__(self, aggregates):
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
        self.normaliseMissingness()
        self.formatComplianceOverTime()
        plt = self.plot()
        if save:
            path = self.aggr.pathPlot + 'Compliance.png'
            plt.savefig(path, dpi=600)
        if show:
            plt.show()

    def exportLatexTable(self, save=True):
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

    def __init__(self, infoTable, labels):
        self.labelsOfLabels = labels.keys()
        self.labels = labels
        self.info = infoTable
        self.features = self.info.columns
        self.normalisedColumns = ['Info. gain (%)', 'Threshold']
        self.entropies = []

    def run(self):
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

    def __init__(self, aggr):
        self.aggr = aggr
        self.outputPath = aggr.pathPlot

    def run(self):
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
        formatedTable = self.formatIndex(self.outputTable)
        latexTable = formatedTable.to_latex()
        if show:
            print(latexTable)
        if save:
            path = self.outputPath + 'DataStationarityStats.tex'
            f = open(path, 'w')
            f.write(latexTable)
            f.close()

    def formatIndex(self, table):
        index = table.index
        formatedIndex = []
        for val in index:
            if val not in ['FAM', 'LAM','VAM']:
                val = val.replace('_', ' ')
                val = val.capitalize()
            formatedIndex.append(val)
        table.index = formatedIndex
        return table

class DiscretisationTable:

    def __init__(self, aggr):
        self.aggr = aggr

    def run(self):
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
        latexTable = self.sampleTable.to_latex(index=False)
        if save:
            path = self.aggr.pathPlot + 'DataSampleSymptomClass.tex'
            f = open(path, 'w')
            f.write(latexTable)
            f.close()
