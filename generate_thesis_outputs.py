import numpy as np
from thesis import Aggregates, Compliance, InfoGainTable

path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/plots/'

aggr = Aggregates('.pkl', path, plot_path)
comp = Compliance(aggr)
#comp.generateFigure(show=False, save=True)
#comp.exportLatexTable(save=True)




comp = Compliance(aggr)
comp.normaliseMissingness()
passiveLabels = list(comp.dfCount.T['No Missingness'] > 70)

infoTable = aggr.getPariticpantsInfo()
labels = {'Passive': passiveLabels}
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
print(igTable)
