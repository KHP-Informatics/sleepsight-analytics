from thesis import Aggregates, Compliance, InfoGainTable

path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/plots/'

aggr = Aggregates('.pkl', path, plot_path)
comp = Compliance(aggr)
comp.generateFigure(show=False, save=True)
comp.exportLatexTable(save=True)

infoTable = aggr.getPariticpantsInfo()
labels = {'Test':['Yes','No','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No','Yes','No','Yes']}
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

