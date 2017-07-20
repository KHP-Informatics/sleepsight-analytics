# !/usr/bin/python3
import matplotlib
## importing 'matplotlib' here to instruct the use of any X-backend window for plots (necessary for servers)
matplotlib.use('Agg')

import sys
from tools import Participant, Logger
from preprocessing import KalmanImputation, Stationarity
import pandas as pd
from analysis import MissingnessDT, Periodicity, GpModel, ModelPrep, NonParaModel

# Overarching SleepSight pipeline script
participantID = 1
path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/plots/'
log_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/logs/'

args = sys.argv
if len(args) > 1:
    print('Input arguments: {}'.format(args))
    participantID = args[1]
    path = args[2]
    plot_path = args[3]
    log_path = args[4]


p = Participant(id=participantID, path=path)
p.activeSensingFilenameSelector = 'diary'
p.metaDataFileName = 'meta_patients.json'
p.sleepSummaryFileName = 'FB_sleep_summary.csv'
p.load()
#p.pipelineStatus['periodicity'] = False
#p.pipelineStatus['non-parametric model prep'] = False
#p.pipelineStatus['delay determination'] = False
#p.saveSnapshot(p.path)
print(p)

log = Logger(log_path, p.id+'.txt', printLog=True)
log.emit('Begin analysis pipeline', newRun=True)

# Task: 'trim data' to Study Duration
if not p.isPipelineTaskCompleted('trim data'):
    log.emit('Continuing with TRIM DATA...')
    p.trimData(p.info['startDate'], duration=56)
    p.updatePipelineStatusForTask('trim data')
    p.saveSnapshot(path)
else:
    log.emit('Skipping TRIM DATA - already completed.')


# Task: 'missingness' (Decision tree: No missingness vs not worn vs not charged)
if not p.isPipelineTaskCompleted('missingness'):
    log.emit('Continuing with MISSINGNESS computation...')
    mdt = MissingnessDT(passiveData=p.passiveData,
                        activeDataSymptom=p.activeDataSymptom,
                        activeDataSleep=p.activeDataSleep,
                        startDate=p.info['startDate'])
    mdt.constructDecisionTree()
    mdt.run()
    mdt.formatMissingness()
    p.missingness = mdt.missingness
    p.updatePipelineStatusForTask('missingness')
    p.saveSnapshot(path)
else:
    log.emit('Skipping MISSINGNESS - already completed.')


# Task: 'imputation' (Kalman smoothing)
if not p.isPipelineTaskCompleted('imputation'):
    log.emit('Continuing with IMPUTATION...')
    for pSensor in p.passiveSensors:
        if pSensor not in 'timestamp':
            ki = KalmanImputation()
            ki.addObservtions(p.getPassiveDataColumn(pSensor))
            ki.fit(n_iterations=10)
            ki.smooth()
            ki.limitToPositiveVals()
            p.setPassiveDataColumn(ki.imputedObservations, col=pSensor)
    p.updatePipelineStatusForTask('imputation')
    p.saveSnapshot(path)
else:
    log.emit('Skipping IMPUTATION - already completed.')


# Task 'stationarity' (differencing)
if not p.isPipelineTaskCompleted('stationarity'):
    log.emit('Continuing with STATIONARITY...')
    st = Stationarity(data=p.activeDataSymptom)
    st.makeStationary(show=True)
    p.stationarySymptomData = st.stationaryData
    p.stationarySymptomStats = st.stationaryStats

    st = Stationarity(data=p.passiveData)
    st.makeStationary(show=True)
    p.stationaryPassiveData = st.stationaryData
    p.stationaryPassiveStats = st.stationaryStats
    p.updatePipelineStatusForTask('stationarity')
    p.saveSnapshot(path)
else:
    log.emit('Skipping STATIONARITY - already completed.')


# Task 'periodicity' (Determining time window of repating sequences)
if not p.isPipelineTaskCompleted('periodicity'):
    log.emit('Continuing with PERIODICITY...')
    periodicity = {}
    acfPeakStats = {}
    for pSensor in p.passiveSensors:
        if pSensor not in 'timestamp':
            pdy = Periodicity(identifier=p.id, sensorName=pSensor, path=plot_path, log=log)
            pdy.addObservtions(p.getPassiveDataColumn(pSensor, type='stationary'))
            pdy.auto_corr(nMinutes=(1440 * 28))
            pdy.plot('acf', show=False, save=True)
            acfPeakStats[pSensor] = pdy.peakStats
            periodicity[pSensor] = pdy.periodicity
    p.periodicity = periodicity
    p.acfPeakStats = acfPeakStats
    p.updatePipelineStatusForTask('periodicity')
    p.saveSnapshot(path)
else:
    log.emit('Skipping PERIODICITY - already completed.')


######## NON PARAMETRIC #################################################
# Task 'non-parametric modelprep'
if not p.isPipelineTaskCompleted('non-parametric model prep'):
    log.emit('Continuing with NON-PARAMETRIC MODEL PREP...')
    mp = ModelPrep(log=log)
    mp.discretiseSymtomScore(p.stationarySymptomData, p.activeDataSymptom)
    p.activeDataSymptom = mp.discretisedRawScoreTable
    p.stationarySymptomData = mp.discretisedStationarySymptomScoreTable
    npm = NonParaModel(yFeature='total', dayDivisionHour=12)
    npm.submitData(participant=p, xFeatures=p.passiveSensors)
    npm.constructModel()
    p.nonParametricFeatures = npm.features
    p.updatePipelineStatusForTask('non-parametric model prep')
    p.saveSnapshot(path)
else:
    log.emit('Skipping NON-PARAMETRIC MODEL PREP - already completed.')


# Task 'delay determination' (determine delay between active and passive data)
if not p.isPipelineTaskCompleted('delay determination'):
    log.emit('Continuing with DELAY DETERMINATION...')
    delayCCF = []
    features = p.nonParametricFeatures.columns
    for feature in features:
        pdy = Periodicity(identifier=p.id, sensorName=feature, path=plot_path, log=log)
        pdy.addObservtions(p.nonParametricFeatures[feature])
        delay = pdy.cross_cor(targetObservation=p.activeDataSymptom['total'], lag=14)
        delayCCF.append(delay)
    dfDelay = pd.DataFrame(delayCCF, columns=['Participant {}'.format(p.id)])
    dfDelay.index = features
    p.featuresDelay = dfDelay
    p.updatePipelineStatusForTask('delay determination')
    p.saveSnapshot(path)
else:
    log.emit('Skipping DELAY DETERMINATION - already completed.')

log.getLastMessage()

exit()

# Task 'gp-model gen' (Determining time window of repeating sequences)
if not p.isPipelineTaskCompleted('GP model gen.'):
    log.emit('Continuing with GP-MODEL GEN...')
    gpm = GpModel(xFeatures=p.passiveSensors, yFeature='total', dayDivisionHour=12, log=log)
    gpm.submitData(active=p.activeDataSymptom, passive=p.passiveData)
    gpm.createIndexTable()
    print(gpm.indexDict)
else:
    log.emit('Skipping GP-MODEL GEN - already completed.')