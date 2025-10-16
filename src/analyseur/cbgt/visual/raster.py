# ~/analyseur/cbgt/visual/raster.py
#
# Documentation by Lungsi 6 Oct 2025
#
# This contains function for loading the files
#

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import re

# from ..loader import get_desired_spiketimes_superset
from analyseur.cbgt.loader import get_desired_spiketimes_subset

def _get_line_colors(colors=False, no_neurons=None):
    if colors:
        return [f'C{i}' for i in range(no_neurons)]  # set different colors for each set of positions
    else:
        return "black"

def rasterplot(spiketimes_superset, colors=False, neurons="all", nucleus=None):
    """
    Displays the rasterplot of the given spike times and returns the plot figure (to save if necessary).

    :param spiketimes_superset: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    :param colors: `[OPTIONAL] False` [default] or True
    :param neurons: [OPTIONAL] "all" [default] or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]
    :param nucleus: [OPTIONAL] None or name of the nucleus (string)
    :return: object `matplotlib.pyplot.eventplot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.eventplot.html>`_

    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spiketimes_superset = loadST.get_spiketimes_superset()

      from analyseur.cbgt.visual.raster import rasterplot

    2. Raster for all the neurons

    ::

      rasterplot(spiketimes_superset)

    3. Raster for first 50 neurons

    ::

      rasterplot(spiketimes_superset, neurons=range(50))

    4. Raster for second 50 neurons

    ::

      rasterplot(spiketimes_superset, neurons=range(50, 100))

    """
    [desired_spiketimes_subset, yticks] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)
    n_neurons = len(desired_spiketimes_subset)

    linecolors = _get_line_colors(colors=colors, no_neurons=n_neurons)

    # ====== PLOT PARAMETERS ======
    n_yticks = 20
    linelengths = 0.5  # default: 1
    linewidths = 0.5  # default: 1.5
    # ytick_trigger = 50
    # ytick_interval = 10
    ytick_interval = int(n_neurons / n_yticks)

    # lineoffsets = 0.5  # default: 1
    lineoffsets = np.arange(1, n_neurons + 1)

    plt.eventplot(desired_spiketimes_subset, colors=linecolors,
                  linelengths=linelengths, linewidths=linewidths,
                  lineoffsets=lineoffsets,
                  orientation="horizontal", alpha=None)

    # if n_neurons > ytick_trigger:
    #     # plt.yticks([])
    #     plt.yticks(lineoffsets[::ytick_interval], yticks[::ytick_interval])
    # else:
    #     plt.yticks(lineoffsets, yticks)

    plt.yticks(lineoffsets[::ytick_interval], yticks[::ytick_interval])

    plt.ylabel("neurons")
    plt.xlabel("Time (ms)")

    nucname = "" if nucleus is None else " in "+nucleus
    allno = str(n_neurons)
    if neurons=="all":
        plt.title("Raster of all (" + allno + ") the neurons" + nucname)
    else:
        plt.title("Raster of the selected neurons" + nucname)

    plt.show()

    return plt