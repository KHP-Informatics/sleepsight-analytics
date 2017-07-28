from tools import Logger
from thesis import Aggregates, Compliance, InfoGainTable, StationaryTable, DiscretisationTable, PeriodictyTable


path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/plots/'
log_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/logs/'

options = {'periodicity': True,
           'participant-info': True,
           'compliance': True,
           'stationarity': True,
           'symptom-score-discretisation': True,
           'feature-selection': True,
           'non-parametric-svm': True
           }

log = Logger(log_path, 'thesis_outputs.log', printLog=True)

# Load Participants
log.emit('Loading participants...')
aggr = Aggregates('.pkl', path, plot_path)


# Export Periodicity tables
if options['periodicity']:
    log.emit('Generating PERIODCITY table...')
    pt = PeriodictyTable(aggr, log)
    pt.run()
    pt.exportLatexTable(summary=False)
    pt.exportLatexTable(summary=True)


# Export Participant Info
if options['participant-info']:
    log.emit('Generating PARTICIPANTS-INFO table...')
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

# Compliance
if options['compliance']:
    log.emit('Generating COMPLIANCE figure and table...')
    # Compliance Figure
    comp = Compliance(aggr, log)
    comp.generateFigure(show=False, save=True)
    comp.exportLatexTable(save=True)

    # Compliance Information Gain
    comp = Compliance(aggr, log)
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
if options['stationarity']:
    log.emit('Generating STATIONARITY table...')
    stTable = StationaryTable(aggr, log)
    stTable.run()
    stTable.exportLatexTable(show=False, save=True)

# Symptom Score discretisation
if options['symptom-score-discretisation']:
    log.emit('Generating SYMPTOM-SCORE-DISCRETISATION table...')
    disTable = DiscretisationTable(aggr, log)
    disTable.run()
    disTable.exportLatexTable(show=False, save=True)

# feature selection with MIFS & mRMR
if options['feature-selection']:
    log.emit('Generating FEATURE-SELECTION table...')

# SVM-linear results
if options['non-parametric-svm']:
    log.emit('Generating NON-PARAMETRIC-SVM table...')

