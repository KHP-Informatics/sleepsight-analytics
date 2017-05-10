import pandas as pd
import matplotlib.pyplot as plt
from thesis.aggregates import Aggregates

path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/data/'
plot_path = '/Users/Kerz/Documents/projects/SleepSight/ANALYSIS/plots/'


class Compliance:
    def __init__(self, aggregates):
        self.aggr = aggregates
        self.dfCount, self.dfDaily = self.aggr.getMissingness()
        print(self.dfCount)

    def plot(self):
        plt.figure()
        bp = self.dfCount.boxplot()
        plt.show()


aggr = Aggregates('.pkl', path, plot_path)
comp = Compliance(aggr)

