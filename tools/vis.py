# !/bin/python3
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import datetime

class QuickPlot:

    def __init__(self, identifier='SleepSight', path='/'):
        self.outputPath = path
        self.identifier = identifier
        self.colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def singlePlotOfTypeLine(self, observation, title='Title', lineLabels=['Label'], ticks=[], tickLabels=[],
                       text='', matrix=False, show=True, saveFigure=False, figsize=(9, 6)):
        plt.close('all')
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=figsize)
        self.plot_line(ax, observation, lineLabels=lineLabels, matrix=matrix)

        if text not in '':
            ax.text(0, max(observation), text,
                    verticalalignment='bottom', horizontalalignment='left',
                    color='black', fontsize=8)
        ax.set_ylabel('Correlation')
        if len(ticks) > 0:
            plt.xticks(ticks)
            ax.xaxis.set_ticklabels(tickLabels)
            ax.set_xlabel('Lag (in Days)')
        else:
            ax.set_xlabel('Lag (in Minutes)')
        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        if saveFigure:
            path = self.outputPath + '{}_line_{}.png'.format(self.identifier, title)
            plt.savefig(path)
        if show:
            plt.show()

    def plot_line(self, ax, line_data, matrix=False,lineLabels=['Label']):
        N = int(len(list(line_data)))
        if matrix:
            linesN = len(line_data)
            for i in range(linesN):
                ax.plot(np.arange(N), line_data[i], color=self.colours[i%linesN], label=lineLabels[0])
        else:
            ax.plot(np.arange(N), line_data, color='b', label=lineLabels[0])

    def singlePlotOfTypeHeatmap(self, heatmap_data, title='Title', show=True, saveFigure=False, figsize=(9, 6)):

        sns.set(style="white")

        # Generate a large random dataset
        d = pd.DataFrame(heatmap_data)
        # Set N/As to 0
        d = d.fillna(0)

        # Generate a mask for the upper triangle
        mask = np.zeros_like(d, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=figsize)

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        g = sns.clustermap(d, linewidths=.3, cmap=cmap, vmax=0.5, figsize=(figsize[0], figsize[0]))

        plt.title(title, loc='left')
        xlabels = g.ax_heatmap.get_xticklabels()
        ylabels = g.ax_heatmap.get_yticklabels()
        plt.setp(xlabels, rotation=270, fontsize=7)
        plt.setp(ylabels, rotation=0, fontsize=7)

        if saveFigure:
            path = self.outputPath + '{}_heatmap_{}.png'.format(self.identifier, title)
            plt.savefig(path)
        if show:
            plt.show()
