# ~/analyseur/cbgt/visual/variation.py
#
# Documentation by Lungsi 30 Oct 2025
#
# This contains function for SpikingStats
#

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, alpha

import re

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.stats.isi import InterSpikeInterval
from analyseur.cbgt.stats.variation import Variations
from analyseur.cbgt.parameters import SpikeAnalysisParams, SimulationParams

spikeanal = SpikeAnalysisParams()
simparams = SimulationParams()

"""
Plots
"""


##########################################################################
#    CV PLOT
##########################################################################

def plotCV_in_ax(ax, spiketimes_superset, nucleus=None, orient=None):
    """
    Draws the Coefficient of Variation on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param neurons: "all" [default] or list: range(a, b) or [1, 4, 5, 9]
    - :param nucleus: string; name of the nucleus
    - :param orient: "horizontal" or None [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """

    n_neurons = len(spiketimes_superset)

    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    all_isi = InterSpikeInterval.compute(spiketimes_superset)
    CVarr = Variations.computeCV(all_isi)
    vec_CV = CVarr.values()

    window = spikeanal.window
    binsz = spikeanal.binsz_100perbin
    n_bins = round((window[1] - window[0]) / binsz)

    if orient=="horizontal":
        ax.barh(range(len(vec_CV)), vec_CV, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
        ax.set_xlabel("CV")
    else:
        ax.bar(range(len(vec_CV)), vec_CV, color="steelblue", edgecolor="black")
        # ax.hist(vec_CV, bins=n_bins, alpha=0.7, color="green", edgecolor="black", )
        ax.set_ylabel("CV")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("CV Distribution of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plotCV(spiketimes_superset, nucleus=None, orient=None):
    """
    Visualize Coefficient of Variation of the given neuron population.

    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param neurons: "all" or list: range(a, b) or [1, 4, 5, 9]
    - :param nucleus: string; name of the nucleus
    - :param orient: "horizontal" or None [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    if orient=="horizontal":
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = plotCV_in_ax(ax, spiketimes_superset, nucleus=nucleus, orient=orient)

    plt.show()

    return fig, ax