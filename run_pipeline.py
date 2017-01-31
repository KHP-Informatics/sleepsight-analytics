# !/usr/bin/python3
from tools import Participant
from preprocessing import KalmanImputation, MissingnessDT
from analysis import Periodicity

# Overarching SleepSight pipeline script

#ISS03 - confirm Kalman imputation
#ISS06 - remove timestamp from p.passiveSensors
#ISS07 - autocorrelation x label

path = '/Users/Kerz/Documents/projects/SleepSight/Data-SleepSight/SleepSight_methods_paper_data/'
p = Participant(id=1, path=path)
p.activeSensingFilenameSelector = 'diary'
p.load()
print(p)

print('\nBegin analysis pipeline:')
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
    p.saveSnapshot()
else:
    print('\nSkipping MISSINGNESS - already completed.')

# Task: 'imputation' (Kalman smoothing)
if p.isPipelineTaskCompleted('imputation'):
    print('\nContinuing with IMPUTATION...')
    for pSensor in p.passiveSensors:
        ki = KalmanImputation()
        ki.addObservtions(p.getPassiveDataColumn(pSensor))
        ki.fit(n_iterations=10)
        ki.smooth()
        ki.limitToPositiveVals()
        p.setPassiveDataColumn(ki.imputedObservations, col=pSensor)
    p.updatePipelineStatusForTask('imputation')
    p.saveSnapshot()
else:
    print('\nSkipping IMPUTATION - already completed.')


# Task 'periodicity' (Determining time window of repating sequences)
if not p.isPipelineTaskCompleted('periodicity'):
    print('\nContinuing with PERIODICITY...')
    for pSensor in p.passiveSensors:
        if pSensor not in 'timestamp':
            pdy = Periodicity(identifier=p.id, sensorName=pSensor, path=path)
            pdy.addObservtions(p.getPassiveDataColumn(pSensor))
            pdy.serial_corr(nSteps=100)
            pdy.auto_corr()
            pdy.pearson_corr()
            pdy.plot('scf', show=False, save=True)
            pdy.plot('acf', show=False, save=True)
            pdy.plot('pcf', show=False, save=True)
else:
    print('\nSkipping PERIODICITY - already completed.')
