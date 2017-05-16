from thesis import Aggregates, Compliance
from analysis import MissingnessDT, Periodicity

path = '{YYY}/ANALYSIS/data/'
plot_path = '{YYY}/ANALYSIS/plots/'

########################################################################################################################
##  STEP 1 - SETUP
##  load all participants data
def loadParticipants(path):
    aggr = Aggregates('.pkl', path, plot_path)
    return aggr
participants = loadParticipants(path)
##  selecting participant 0
pM = participants.aggregates[0]
pS = participants.aggregates[11]
# uncomment the three lines below if you run the STEP2 and STEP3 using this script
Compliance(participants)
pdy = Periodicity(identifier=pS.id, sensorName='heart_Minutes', path=plot_path)
MissingnessDT(passiveData=pM.passiveData, activeDataSymptom=pM.activeDataSymptom, activeDataSleep=pM.activeDataSleep, startDate=pM.info['startDate'])
print('\nGreat - you have loaded your data successfully and are ready to go!\n')
########################################################################################################################


##  printing participant information
#print(p)

##  STEP 2 - MISSINGNESS
######## Determine missingness #############
##  retrieve passive and active data from participant
#mdt = MissingnessDT(passiveData=pM.passiveData, activeDataSymptom=pM.activeDataSymptom, activeDataSleep=pM.activeDataSleep, startDate=pM.info['startDate'])
##  construct decision tree (see slide pack)
#mdt.constructDecisionTree()
##  run decision tree
#mdt.run()
##  retrieve and format result from decision tree
#mdt.formatMissingness()
##  print summary information
#print(mdt)


######## Generate Missingness Figure #######
##  load all participants
#comp = Compliance(participants)
##  generate figure
#comp.generateFigure(show=True, save=False)


##  STEP 3 - SEASONALITY
######### Determine periodicity ###########
##  Select sensor stream to evaluate
pSensor = 'heart_Minutes'
##  You can select from: [
#                           'FAM',
#                           'LAM',
#                           'VAM',
#                           'awake_count',
#                           'battery',
#                           'calories',
#                           'heart_Minutes',
#                           'intra_calories',
#                           'intra_steps',
#                           'light',
#                           'min_asleep',
#                           'sleep_Minutes',
#                           'steps'
#                         ]

##  Setup Periodicity from participant information
#pdy = Periodicity(identifier=pS.id, sensorName=pSensor, path=plot_path)
#pdy.addObservtions(pS.getPassiveDataColumn(pSensor))

##  Perform auto-correlation and plot
#pdy.auto_corr()
#pdy.plot('acf', show=True, save=False)

## Perform person's correlation on day by day time series
#pdy.pearson_corr()
#pdy.plot('pcf', show=True, save=False)