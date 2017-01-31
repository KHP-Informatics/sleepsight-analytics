# !/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# testing signal's serial dependency
# determining periodicity using ACF (auto-correlation function)
# testing periodicity using Pearson's correlation

class Periodicity:

    def __init__(self, sensorName="Sensor name"):
        self.observations = []
        self.sensorName = sensorName

    def addObservtions(self, observations):
        obs = np.array(observations, dtype='U32')
        obs_missing = np.where(obs == '-')
        obs[obs_missing] = 999999
        obs_masked = np.ma.masked_array(obs, dtype='float32')
        obs_masked[obs_missing] = np.ma.masked
        self.observations = obs_masked

        # serial-correlation function

    def serial_corr(self, step=1, steps=10):
        self.scf = []
        n = len(self.observations)
        for i in range(int(steps/step)):
            lag = step*i
            y1 = self.observations[lag:]
            y2 = self.observations[:n - lag]
            self.scf.append(np.corrcoef(y1, y2, ddof=0)[0,1])

    # auto-correlation function
    def auto_corr(self):
        self.acf = np.correlate(self.observations, self.observations, mode='same')

    # pearson's correlation matrix
    def pearson_corr(self, lag=1440):
        n = int(len(self.observations)/lag) - 1
        print(n)
        observation_windows = []
        for i in range(n):
            observation_windows.append(self.observations[(i*lag):((i*lag)+lag)])
        self.pcf = np.corrcoef(observation_windows)

    def plot(self, type='all', save=True):
        if type is 'scf' or type is 'all':
            self.plotScf(save)
        if type is 'acf' or type is 'all':
            self.plotAcf(save)
        if type is 'pcf' or type is 'all':
            self.plotPcf(save)
        if type not in ['all', 'scf', 'acf', 'pcf']:
            print('[PERIODICITY] WARN: Did not plot. Choose from "all", "scf", "acf" or "pcf".')


    def plotScf(self, save=True):
        pass

    def plotAcf(self, save=True):
        pass

    def plotPcf(self, save=True):
        pass


    def plot_line(self, ax, line_data, type='Label', fontsize=10):
        N = int(len(list(line_data)))
        ax.plot(np.arange(N), line_data, color='b', label=type)
        ax.set_ylabel(self.capitaliseFirstCharacter(self.withMeasure))
        ax.set_xlabel("Lag (in Minutes)")
        #ax.set_xticks(np.arange(N_half) * 2)
        #ax.set_xticklabels(np.arange(N_half) * 2, rotation=270, fontsize=fontsize)
        ax.set_title('{}: {}'.format(type, self.sensorName), fontsize=fontsize)
        ax.legend(loc='upper right', fontsize=8)



    def temp(self):
        plt.close('all')
        plt.style.use('ggplot')
        fig, ax = plt.subplots(ncols=self.nCols, nrows=int(self.nPlots/self.nCols), figsize=figsize)
        if self.nCols == 1:
            self.plotRows(ax, 0)
        else:
            for colIdx in range(self.nCols):
                self.plotRows(ax, colIdx)
        plt.tight_layout()

        if saveFigure:
            path = self.outputPath + 'HipDynamics_{}.png'.format(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
            plt.savefig(path)
        if show:
            plt.show()


    def plotRows(self, ax, colIdx):
        nRows = int(self.nPlots / self.nCols)
        for rowIdx in range(nRows):
            selectData = self.data[self.data[self.indexName] == self.indexVals[(rowIdx%nRows) + (nRows*colIdx)]]
            plotY = selectData.filter(regex=self.withMeasure)
            plotY_mtrx = np.array(plotY.as_matrix(columns=None), dtype=float)
            labelWithVals = np.array(self.index.loc[plotY.index.values, 'index-{}'.format(self.labelWith)])

            if self.nCols == 1:
                axI = ax[rowIdx]
            else:
                axI = ax[rowIdx][colIdx]

            self.plotFigure(axI, plotY_mtrx, plotY.columns.values,
                            self.indexVals[(rowIdx%nRows) + (nRows*colIdx)], labelWithVals)

    def determineNumberOfCols(self):
        nCols = 1
        if self.nPlots > 4:
            nCols = 2
        if self.nPlots > 11:
            nCols = 3
        return nCols

    def plotFigure(self, ax, mtrx, cols, indexVal, labelWithVals, fontsize=10):
        N = int(len(list(cols)))
        N_half = int(N/2)
        for i in range(len(mtrx)):
            ax.plot(np.arange(N), mtrx[i], color=self.colours[i % 8], label=labelWithVals[i])
        ax.set_ylabel(self.capitaliseFirstCharacter(self.withMeasure))
        ax.set_xlabel("Features")
        ax.set_xticks(np.arange(N_half)*2)
        ax.set_xticklabels(np.arange(N_half)*2, rotation=270, fontsize=fontsize)
        ax.set_title('{}: {}'.format(self.indexBy, indexVal), fontsize=fontsize)
        ax.legend(loc='upper right', title=self.labelWith, fontsize=8)
        if self.fixYaxis:
            ax.set_ylim((-0.05, 0.05))

    def capitaliseFirstCharacter(self, str):
        rendered = str
        rendered = rendered[0].capitalize() + rendered[1:len(rendered)]
        return rendered