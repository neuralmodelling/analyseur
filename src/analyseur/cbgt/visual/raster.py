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

def _extract_neuron_no(neuron_id):
    match = re.search(r'n(\d+)', neuron_id)
    return int(match.group(1))

def _get_desired_spiketrains(spiketrains, neurons="all"):
    desired_spiketrains = []
    yticks = []

    if neurons=="all":
        for nX, data in spiketrains.items():
            desired_spiketrains.append( list(data) )
            # yticks.append( _extract_neuron_no(nX) )
            yticks.append(nX)
    else: # neurons = range(a, b) or neurons = [1, 4, 5, 9]
        for i in neurons:
            neuron_id = "n" + str(i)
            desired_spiketrains.append( list(spiketrains[neuron_id]) )
            # yticks.append( _extract_neuron_no(neuron_id) )
            yticks.append(neuron_id)
    return desired_spiketrains, yticks


def _get_line_colors(colors=False, no_neurons=None):
    if colors:
        return [f'C{i}' for i in range(no_neurons)]  # set different colors for each set of positions
    else:
        return "black"

def rasterplot(spiketrains, colors=False, neurons="all", nucleus=None):
    """
    Displays the rasterplot of the given spike times and returns the plot figure (to save if necessary).

    :param spiketrains: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    :param colors: `[OPTIONAL] False` [default] or True
    :param neurons: [OPTIONAL] "all" [default] or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]
    :param nucleus: [OPTIONAL] None or name of the nucleus (string)
    :return: object `matplotlib.pyplot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_

    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketrains()

      from analyseur.cbgt.visual.raster import rasterplot

    2. Raster for all the neurons

    ::

      rasterplot(spike_trains)

    3. Raster for first 50 neurons

    ::

      rasterplot(spike_trains, neurons=range(50))

    4. Raster for second 50 neurons

    ::

      rasterplot(spike_trains, neurons=range(50, 100))

    """
    # ====== PLOT PARAMETERS ======
    linelengths = 0.5  # default: 1
    linewidths = 0.5  # default: 1.5
    ytick_trigger = 50
    ytick_interval = 10

    [desired_spiketrains, yticks] = _get_desired_spiketrains(spiketrains, neurons=neurons)
    linecolors = _get_line_colors(colors=colors, no_neurons=len(desired_spiketrains))

    # lineoffsets = 0.5  # default: 1
    lineoffsets = np.arange(1, len(desired_spiketrains) + 1)

    plt.eventplot(desired_spiketrains, colors=linecolors,
                  linelengths=linelengths, linewidths=linewidths,
                  lineoffsets=lineoffsets,
                  orientation="horizontal", alpha=None)

    if len(desired_spiketrains) > ytick_trigger:
        # plt.yticks([])
        plt.yticks(lineoffsets[::ytick_interval], yticks[::ytick_interval])
    else:
        plt.yticks(lineoffsets, yticks)

    plt.ylabel("neurons")
    plt.xlabel("Time (ms)")

    nucname = "" if nucleus is None else " in "+nucleus
    allno = str(len(desired_spiketrains))
    if neurons=="all":
        plt.title("Raster of all (" + allno + ") the neurons" + nucname)
    else:
        plt.title("Raster of the selected neurons" + nucname)

    plt.show()

    return plt