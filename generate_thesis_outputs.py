from thesis import Aggregates, Compliance, InfoGainTable, StationaryTable, DiscretisationTable, PeriodictyTable


path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/plots/'
# Load Participants
aggr = Aggregates('.pkl', path, plot_path)

# Export Periodicity tables
pt = PeriodictyTable(aggr)
pt.run()
pt.exportLatexTable(summary=False)
pt.exportLatexTable(summary=True)


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

# Stationarity results
stTable = StationaryTable(aggr)
stTable.run()
stTable.exportLatexTable(show=False, save=True)

# Symptom Score discretisation
disTable = DiscretisationTable(aggr)
disTable.run()
disTable.exportLatexTable(show=False, save=True)



