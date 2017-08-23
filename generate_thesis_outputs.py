
import numpy as np
from tools import Logger, Participant
import thesis as T


path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/plots/'
log_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/logs/'

options = {'periodicity': False,
           'participant-info': False,
           'compliance': False,
           'stationarity': False,
           'symptom-score-discretisation': False,
           'feature-delay': False,
           'feature-selection': False,
           'non-parametric-svm': False,
           'non-parametric-gp': True
           }

log = Logger(log_path, 'thesis_outputs.log', printLog=True)

# Load Participants
log.emit('Loading participants...', newRun=True)
aggr = T.Aggregates('.pkl', path, plot_path)


# Export Periodicity tables
if options['periodicity']:
    log.emit('Generating PERIODCITY table...')
    pt = T.PeriodictyTable(aggr, log)
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
    comp = T.Compliance(aggr, log)
    comp.generateFigure(show=False, save=True)
    comp.exportLatexTable(save=True)

    # Compliance Information Gain
    comp = T.Compliance(aggr, log)
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
    igTable = T.InfoGainTable(infoTable[features], labels)
    igTable.run()
    igTable.exportLatexTable(aggr.pathPlot, orderedBy='Passive data', save=True)


# Stationarity results
if options['stationarity']:
    log.emit('Generating STATIONARITY table...')
    stTable = T.StationaryTable(aggr, log)
    stTable.run()
    stTable.exportLatexTable(show=False, save=True)

# Symptom Score discretisation
if options['symptom-score-discretisation']:
    log.emit('Generating SYMPTOM-SCORE-DISCRETISATION table...')
    disTable = T.DiscretisationTable(aggr, log)
    disTable.run()
    disTable.exportLatexTable(show=False, save=True)

# feature delay
if options['feature-delay']:
    log.emit('Generating FEATURE-DELAY table...')
    dEval = T.DelayEval(aggr, log)
    dEval.generateDelayTable()
    dEval.exportLatexTable()

# feature selection with MIFS & mRMR
if options['feature-selection']:
    log.emit('Generating FEATURE-SELECTION table...')
    fs = T.FeatureSelectionEval(aggr, log)
    fs.generateHistogramForNTopFeatures(nFeatures=10)
    fs.generateFigure(show=True)

# SVM-linear results
if options['non-parametric-svm']:
    log.emit('Generating NON-PARAMETRIC-SVM table...')

    fs = T.FeatureSelectionEval(aggr, log)
    fs.generateHistogramForNTopFeatures(nFeatures=10)
    fMifs = []
    fMrmr = []
    for table in fs.histogramsFs:
        if 'MIFS-ADASYN' in table.columns:
            fMifs = list(table.index[0:10])
        if 'mRMR-ADASYN' in table.columns:
            fMrmr = list(table.index[0:10])
    totalF = {
        'mRMR': {'ADASYN': {'fRank': fMifs}},
        'MIFS': {'ADASYN': {'fRank': fMrmr}}
    }

    results = T.compute_SVM_on_all_participants(aggr, totalF, log)
    pTotal = Participant(id=99, path=path)
    pTotal.id = 'Total'
    pTotal.nonParametricResults = results
    aggr.aggregates.append(pTotal)
    npEval = T.NonParametricSVMEval(aggr, log)
    npEval.logClassificationReports()
    npEval.summarise()
    npEval.exportLatexTable(show=True)

    log.emit('\n{}'.format(np.mean(npEval.summary)), indents=1)
    log.emit('\n{}'.format(np.std(npEval.summary)), indents=1)

# GP results
if options['non-parametric-gp']:
    gpEval = T.GaussianProcessEval(aggr, log)
    gpEval.logClassificationReports()
    gpEval.exportLatexTable(mean=True)
