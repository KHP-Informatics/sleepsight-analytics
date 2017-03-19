# !/usr/bin/python3
import matplotlib
## importing 'matplotlib' here to instruct the use of any X-backend window for plots (necessary for servers)
matplotlib.use('Agg')

import sys
from tools import Participant
from preprocessing import KalmanImputation, MissingnessDT
from analysis import Periodicity, GpModel

# Overarching SleepSight pipeline script

#ISS06 - remove timestamp from p.passiveSensors
participantID = 2
path = '/Users/Kerz/Documents/projects/SleepSight/Data-SleepSight/SleepSight_methods_paper_data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/Data-SleepSight/SleepSight_methods_paper_plots/'

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
    mdt = MissingnessDT()
    mdt.setSensors(p.passiveSensors)
    mdt.setDataset(p.passiveData)
    mdt.computeMissingness()
    print(mdt)
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


# Task 'periodicity' (Determining time window of repating sequences)
if not p.isPipelineTaskCompleted('periodicity'):
    print('\nContinuing with PERIODICITY...')
    periodicity = {}
    for pSensor in p.passiveSensors:
        if pSensor not in 'timestamp':
            pdy = Periodicity(identifier=p.id, sensorName=pSensor, path=plot_path)
            pdy.addObservtions(p.getPassiveDataColumn(pSensor))
            pdy.serial_corr(nSteps=100)
            pdy.auto_corr()
            pdy.pearson_corr()
            pdy.plot('all', show=False, save=True)
            periodicity[pSensor] = pdy.periodicity
    p.periodicity = periodicity
    p.updatePipelineStatusForTask('periodicity')
    p.saveSnapshot(path)
else:
    print('\nSkipping PERIODICITY - already completed.')

# Task 'gp-model gen' (Determining time window of repeating sequences)
if not p.isPipelineTaskCompleted('GP model gen.'):
    print('\nContinuing with GP-MODEL GEN...')
    gpm = GpModel(xFeatures=p.passiveSensors, yFeature='total', dayDivisionHour=12)
    gpm.submitData(active=p.activeData, passive=p.passiveData)
    gpm.createIndexTable()
else:
    print('\nSkipping GP-MODEL GEN - already completed.')