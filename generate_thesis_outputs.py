from thesis import Aggregates, Compliance, InfoGainTable
import numpy as np
import pandas as pd

path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/plots/'
# Load Participants
aggr = Aggregates('.pkl', path, plot_path)

# Export Participant Info
participantInfo = aggr.getPariticpantsInfo()
features = [
            'id',
            'gender',
            'age',
            'durationIllness',
            'PANSS.general',
            'PANSS.negative',
            'PANSS.positive',
            'PANSS.total',
            'Clozapine',
            'No.of.Drugs'
]
participantInfoSelect = participantInfo[features]
aggr.exportLatexTable(participantInfoSelect, 'DataParticipantInfo')

# Compliance Figure
comp = Compliance(aggr)
comp.generateFigure(show=False, save=True)
comp.exportLatexTable(save=True)

# Compliance Information Gain
comp = Compliance(aggr)
comp.normaliseMissingness()
labelsNoMissingness = comp.dfCount.T['No Missingness']
labelsSleep = comp.dfCount.T['sleep']
labelsSymptom = comp.dfCount.T['symptom']
infoTable = aggr.getPariticpantsInfo()
labels = {'Passive data': labelsNoMissingness,
          'Active (Sleep Q.)': labelsSleep,
          'Active (Symptoms Q.)': labelsSymptom}
features = [
            'PANSS.general',
            'PANSS.negative',
            'PANSS.positive',
            'PANSS.total',
            'age',
            'durationIllness',
            'gender',
            'Clozapine',
            'No.of.Drugs'
            ]
igTable = InfoGainTable(infoTable[features], labels)
igTable.run()
igTable.exportLatexTable(aggr.pathPlot, orderedBy='Passive data', save=True)
print(igTable)

# Symptom Score discretisation
#ToDO: insert into preprocessing class: differencing, limit to 56 days
sampleSet = []
id = 1
for aggregate in aggr.aggregates:
    m = round(np.mean(aggregate.activeDataSymptom['total']), 1)
    sd = round(np.std(aggregate.activeDataSymptom['total']), 2)
    major = len(aggregate.activeDataSymptom[aggregate.activeDataSymptom['total'] < (m+sd)])
    minor = len(aggregate.activeDataSymptom[aggregate.activeDataSymptom['total'] >= (m+sd)])
    majorPercent = int(round((major / (major + minor)) * 100, 0))
    minorPercent = int(round((minor / (major + minor)) * 100,0))
    sampleSet.append([str(id), str(m), str(sd), '{} ({})'.format(major, majorPercent), '{} ({})'.format(minor, minorPercent)])
    id += 1
sampleTable = pd.DataFrame(sampleSet, columns=['Participant', 'Mean', 'SD', 'Major class (%)', 'Minor class (%)'])
latexTable = sampleTable.to_latex(index=False)
plotPath = aggr.pathPlot
path = plotPath + 'DataSampleSymptomClass.tex'
f = open(path, 'w')
f.write(latexTable)
f.close()
