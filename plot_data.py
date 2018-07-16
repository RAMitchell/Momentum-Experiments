import pickle
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib
import re
import os

data_dir = "data"
figures_dir = "figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

all_data = {}
for x in filenames:
    data = pickle.load(open(data_dir + "/" + x, "rb"))
    all_data.update(data)

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],  # use latex default serif font
})

for dataset, data in all_data.items():
    plt.clf()
    snake_name = re.sub(' ', '_', dataset).lower()
    # Plot
    for optimiser, series in data["results"].items():
        plt.plot(series, marker='x', markevery=[len(series) - 1],
                 label=optimiser)

    plt.xlabel('iterations')
    plt.ylabel(data["metric"])
    plt.legend()
    plt.title(dataset)
    plt.savefig(figures_dir + '/' + snake_name + '.pgf', bbox_inches='tight')
    plt.savefig(figures_dir + '/' + snake_name + '.png', bbox_inches='tight')

    # Zoom
    plt.xlim(1e15, 0)
    for name, series in data["results"].items():
        xmin, xmax = plt.xlim()
        plt.xlim(min(xmin, len(series) - 1), max(xmax, len(series) - 1))

    plt.ylim(1e15, 0)
    for name, series in data["results"].items():
        ymin, ymax = plt.ylim()
        plt.ylim(min(ymin, series[-1]), max(ymax, series[-1]))

    # Add padding
    padding = 0.35
    xmin, xmax = plt.xlim()
    xpad = abs(xmin - xmax) * padding
    plt.xlim(xmin - xpad, xmax + xpad)
    ymin, ymax = plt.ylim()
    ypad = abs(ymin - ymax) * padding
    plt.ylim(ymin - ypad, ymax + ypad)

    plt.savefig(figures_dir + '/' + snake_name + '_zoomed.pgf', bbox_inches='tight')
    plt.savefig(figures_dir + '/' + snake_name + '_zoomed.png', bbox_inches='tight')
