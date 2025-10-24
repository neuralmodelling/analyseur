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

# from ..curate import get_desired_spiketimes_superset
from analyseur.cbgt.curate import get_desired_spiketimes_subset

def _get_line_colors(colors=False, no_neurons=None):
    if colors:
        return [f'C{i}' for i in range(no_neurons)]  # set different colors for each set of positions
    else:
        return "black"

def rasterplot(spiketimes_superset, colors=False, neurons="all", nucleus=None, show=True):
    """
    Displays the rasterplot of the given spike times (seconds) and returns the plot figure (to save if necessary).

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
    if n_neurons > 50:
        ytick_interval = int(n_neurons / n_yticks)
    else:
        ytick_interval = 1

    # lineoffsets = 0.5  # default: 1
    # lineoffsets = np.arange(1, n_neurons + 1)
    lineoffsets = np.arange(n_neurons) * 0.8 + 0.5  # minimal spacing between neurons

    # Plot
    plt.clf()

    fig, ax = plt.subplots(figsize=(16, 14))
    ax.eventplot(desired_spiketimes_subset, colors=linecolors,
                 linelengths=linelengths, linewidths=linewidths,
                 lineoffsets=lineoffsets,orientation="horizontal", alpha=None)

    # if n_neurons > ytick_trigger:
    #     # plt.yticks([])
    #     plt.yticks(lineoffsets[::ytick_interval], yticks[::ytick_interval])
    # else:
    #     plt.yticks(lineoffsets, yticks)

    ax.set_yticks(lineoffsets[::ytick_interval], yticks[::ytick_interval])

    ax.set_ylabel("neurons")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, lineoffsets[-1] + 0.5)  # visibile y-range to eliminate white space

    nucname = "" if nucleus is None else " in "+nucleus
    allno = str(n_neurons)
    if neurons=="all":
        ax.set_title("Raster of all (" + allno + ") the neurons" + nucname)
    else:
        ax.set_title("Raster of " + str(neurons[0]) + " to " + str(neurons[-1]) + " neurons" + nucname)

    plt.grid(True, axis="x", alpha=0.3)

    if show:
        plt.show()

    return fig, ax