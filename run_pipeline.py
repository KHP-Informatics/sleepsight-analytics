# !/usr/bin/python3
import matplotlib
## importing 'matplotlib' here to instruct the use of any X-backend window for plots (necessary for servers)
matplotlib.use('Agg')

import sys
from tools import Participant
from preprocessing import KalmanImputation, Stationarity
from analysis import MissingnessDT, Periodicity, GpModel, ModelPrep

# Overarching SleepSight pipeline script
participantID = 3
path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/plots/'

args = sys.argv
if len(args) > 1:
    print('Input arguments: {}'.format(args))
    participantID = args[1]
    path = args[2]
    plot_path = args[3]

p = Participant(id=participantID, path=path)
p.activeSensingFilenameSelector = 'diary'
p.metaDataFileName = 'meta_patients.json'
p.load()
#p.pipelineStatus['periodicity'] = False
#p.pipelineStatus['non-parametric model prep'] = False
#p.pipelineStatus['delay determination'] = False
#p.saveSnapshot(p.path)
print(p)

print('\nBegin analysis pipeline:')


# Task: 'trim data' to Study Duration
if not p.isPipelineTaskCompleted('trim data'):
    print('\nContinuing with TRIM DATA...')
    p.trimData(p.info['startDate'], duration=56)
    p.updatePipelineStatusForTask('trim data')
    p.saveSnapshot(path)
else:
    print('\nSkipping TRIM DATA - already completed.')


# Task: 'missingness' (Decision tree: No missingness vs not worn vs not charged)
if not p.isPipelineTaskCompleted('missingness'):
    print('\nContinuing with MISSINGNESS computation...')
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
    print('\nSkipping MISSINGNESS - already completed.')


# Task: 'imputation' (Kalman smoothing)
if not p.isPipelineTaskCompleted('imputation'):
    print('\nContinuing with IMPUTATION...')
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
    print('\nSkipping IMPUTATION - already completed.')


# Task 'stationarity' (differencing)
if not p.isPipelineTaskCompleted('stationarity'):
    print('\nContinuing with STATIONARITY...')
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
    print('\nSkipping STATIONARITY - already completed.')


# Task 'periodicity' (Determining time window of repating sequences)
if not p.isPipelineTaskCompleted('periodicity'):
    print('\nContinuing with PERIODICITY...')
    periodicity = {}
    acfPeakStats = {}
    ccfPeakStats = {}
    for pSensor in p.passiveSensors:
        if pSensor not in 'timestamp':
            pdy = Periodicity(identifier=p.id, sensorName=pSensor, path=plot_path)
            pdy.addObservtions(p.getPassiveDataColumn(pSensor, type='stationary'))
            pdy.auto_corr(nMinutes=40320)
            pdy.plot('acf', show=False, save=True)
            acfPeakStats[pSensor] = pdy.peakStats
            periodicity[pSensor] = pdy.periodicity
    p.periodicity = periodicity
    p.acfPeakStats = acfPeakStats
    p.updatePipelineStatusForTask('periodicity')
    p.saveSnapshot(path)
else:
    print('\nSkipping PERIODICITY - already completed.')


######## NON PARAMETRIC #################################################
# Task 'non-parametric modelprep'
if not p.isPipelineTaskCompleted('non-parametric model prep'):
    print('\nContinuing with NON-PARAMETRIC MODEL PREP...')
    mp = ModelPrep()
    mp.discretiseSymtomScore(p.stationarySymptomData, p.activeDataSymptom)
    p.activeDataSymptom = mp.discretisedRawScoreTable
    p.stationarySymptomData = mp.discretisedStationarySymptomScoreTable
    #ToDo: feature generation
else:
    print('\nSkipping NON-PARAMETRIC MODEL PREP - already completed.')


# Task 'delay determination' (determine delay between active and passive data)
if not p.isPipelineTaskCompleted('delay determination'):
    print('\nContinuing with DELAY DETERMINATION...')
    #ToDo: delay determination
else:
    print('\nSkipping DELAY DETERMINATION - already completed.')
exit()


# Task 'gp-model gen' (Determining time window of repeating sequences)
if not p.isPipelineTaskCompleted('GP model gen.'):
    print('\nContinuing with GP-MODEL GEN...')
    gpm = GpModel(xFeatures=p.passiveSensors, yFeature='total', dayDivisionHour=12)
    gpm.submitData(active=p.activeDataSymptom, passive=p.passiveData)
    gpm.createIndexTable()
else:
    print('\nSkipping GP-MODEL GEN - already completed.')