import csv
import pandas
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plotBoxplot(filename):
    # f = open(filename, 'r')
    df = np.array(pandas.read_csv(filename))
    df = np.flip(df)
    fig, ax = plt.subplots(figsize=(8, 2))
    bp = plt.boxplot(df, 0, 'x', 0, meanline = False, showmeans = True)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='.')
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
    ax.set_axisbelow(True)
    # ax.set_title('Comparison of IID Bootstrap Resampling Across Five Distributions')
    ax.set_xlabel('Dice coefficient')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    plt.xlim(0.2, 1)
    plt.xticks([0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1], ['0.2', '0.4', '0.6', '0.8', '', '0.9', '', '1.0'])
    plt.yticks(range(1, 6), ['multiple edits', 'one edit', 'simulated edits', 'no edit', 'benchmark'])
    # ax.set_ylabel('Value')
    plt.show()

if __name__ == '__main__':
    plotBoxplot('../../brats_naive124_half1/dice_plot.csv')
