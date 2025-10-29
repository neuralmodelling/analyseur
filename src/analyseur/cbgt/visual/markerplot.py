# ~/analyseur/cbgt/visual/markerplot.py
#
# Documentation by Lungsi 29 Oct 2025
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


##########################################################################
#    Rate Change SCATTER
##########################################################################

def plot_ratechange_in_ax(ax, spiketimes_superset, stimulus_onset=None,
                           window=None, neurons=None, nucleus=None, orient=None):
    """
    Draws the Population Rate Change Scatter on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param stimulus_onset: float; 0 [default]
    - :param window: 2-tuple; (0, 10) [default]
    - :param neurons: "all" [default] or list: range(a, b) or [1, 4, 5, 9]
    - :param nucleus: string; name of the nucleus
    - :param orient: "horizontal" or None [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    # ============== DEFAULT Parameters ==============
    if neurons is None:
        neurons = "all"

    if window is None:
        window = spikeanal.window

    if stimulus_onset is None:
        stimulus_onset = 0

    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)

    n_neurons = len(desired_spiketimes_subset)

    # Compute Rate Change
    baseline_rates = []
    response_rates = []
    for indiv_spiketimes in desired_spiketimes_subset:
        indiv_spiketimes = np.array(indiv_spiketimes)
        baseline_spikes = indiv_spiketimes[(indiv_spiketimes >= window[0]) & (indiv_spiketimes < stimulus_onset)]
        response_spikes = indiv_spiketimes[indiv_spiketimes >= stimulus_onset]

        baseline_rate = len(baseline_spikes) / ((stimulus_onset - window[0]) + 1e-8)
        response_rate = len(response_spikes) / ((window[1] - stimulus_onset) + 1e-8)

        baseline_rates.append(baseline_rate)
        response_rates.append(response_rate)

    # Plot
    if orient=="horizontal":
        ax.scatter(response_rates, baseline_rates, alpha=0.6, color="orange")
        ax.plot([0, max(baseline_rates)], [0, max(baseline_rates)],
                "k--", alpha=0.5, label="No Change")

        ax.set_ylabel("Baseline Rate (Hz)")
        ax.set_xlabel("Response Rate (Hz)")
    else:
        ax.scatter(baseline_rates, response_rates, alpha=0.6, color="orange")
        ax.plot([0, max(baseline_rates)], [0, max(baseline_rates)],
                "k--", alpha=0.5, label="No Change")

        ax.set_ylabel("Response Rate (Hz)")
        ax.set_xlabel("Baseline Rate (Hz)")

    ax.grid(True, alpha=0.3)

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Rate Change: Baseline vs. Response of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plot_ratechange(spiketimes_superset, stimulus_onset=None,
                     window=None, neurons=None, nucleus=None, orient=None):
    """
    Visualize Rate Change Scatter of the given neuron population.

    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param stimulus_onset: float
    - :param window: 2-tuple; defines upper and lower range of the bins
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

    ax = plot_ratechange_in_ax(ax, spiketimes_superset, stimulus_onset=stimulus_onset,
                                window=window, neurons=neurons, nucleus=nucleus, orient=orient)

    if ax is None:
        print("There are no latencies to plot.")
    else:
        plt.show()

    return fig, ax
