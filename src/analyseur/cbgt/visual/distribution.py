# ~/analyseur/cbgt/visual/distribution.py
#
# Documentation by Lungsi 17 Oct 2025
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
from analyseur.cbgt.parameters import SignalAnalysisParams, SimulationParams

siganal = SignalAnalysisParams()
simparams = SimulationParams()


##########################################################################
#    Rate Distribution PLOT
##########################################################################

def plot_ratedist_in_ax(ax, spiketimes_superset, binsz=None, window=None,
                        neurons=None, nucleus=None, orient=None):
    """
    Draws the Population Rate Distribution on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param binsz: integer or float; 0.01 [default]
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
        window = siganal.window

    if binsz is None:
        binsz = siganal.binsz_100perbin

    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)

    # Compute Firing Rate
    firing_rates = [len(indiv_spiketimes) / (window[1] - window[0])
                    for indiv_spiketimes in desired_spiketimes_subset]
    avg_firerate = np.mean(firing_rates)

    n_bins = round((window[1] - window[0]) / binsz)
    n_neurons = len(desired_spiketimes_subset)

    # Plot
    if orient=="horizontal":
        ax.hist(firing_rates, bins=n_bins, alpha=0.7, color="green",
                edgecolor="black", orientation='horizontal')
        ax.axhline(avg_firerate, color="red", linestyle="--", label=f"Mean: {avg_firerate:.1f} Hz")

        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_xlabel("Number of Neurons")
    else:
        ax.hist(firing_rates, bins=n_bins, alpha=0.7, color="green", edgecolor="black",)
        ax.axvline(avg_firerate, color="red", linestyle="--", label=f"Mean: {avg_firerate:.1f} Hz")

        ax.set_ylabel("Number of Neurons")
        ax.set_xlabel("Firing Rate (Hz)")

    ax.grid(True, alpha=0.3)

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Population Rate Distribution of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plot_ratedist(spiketimes_superset, binsz=None, window=None,
                  neurons=None, nucleus=None, orient=None):
    """
    Visualize Rate Distribution of the given neuron population.

    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param binsz: integer or float; defines the number of equal-width bins in the range
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

    ax = plot_ratedist_in_ax(ax, spiketimes_superset, binsz=binsz, window=window,
                             neurons=neurons, nucleus=nucleus, orient=orient)

    plt.show()

    return fig, ax


##########################################################################
#    Latency Distribution PLOT
##########################################################################

def plot_latencydist_in_ax(ax, spiketimes_superset, stimulus_onset=None, binsz=None,
                           window=None, neurons=None, nucleus=None, orient=None):
    """
    Draws the Population Latency Distribution on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param stimulus_onset: float; 0 [default]
    - :param binsz: integer or float; 0.01 [default]
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
        window = siganal.window

    if binsz is None:
        binsz = siganal.binsz_100perbin

    if stimulus_onset is None:
        stimulus_onset = 0

    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)

    n_bins = round((window[1] - window[0]) / binsz)
    n_neurons = len(desired_spiketimes_subset)

    # Compute Latencies
    latencies = []
    for indiv_spiketimes in desired_spiketimes_subset:
        indiv_spiketimes = np.array(indiv_spiketimes)
        response_spikes = indiv_spiketimes[indiv_spiketimes >= stimulus_onset]
        if len(response_spikes) > 0:
            latency = np.min(response_spikes) - stimulus_onset
            latencies.append(latency)

    if len(latencies) > 0:
        avg_latency = np.mean(latencies)
        # Plot
        if orient=="horizontal":
            ax.hist(latencies, bins=n_bins, alpha=0.7, color="green",
                    edgecolor="black", orientation='horizontal')
            ax.axhline(avg_latency, color="red", linestyle="--", label=f"Mean: {avg_latency:.1f} s")

            ax.set_ylabel("Response Latency (s)")
            ax.set_xlabel("Number of Neurons")
        else:
            ax.hist(latencies, bins=n_bins, alpha=0.7, color="green", edgecolor="black",)
            ax.axvline(avg_latency, color="red", linestyle="--", label=f"Mean: {avg_latency:.1f} s")

            ax.set_ylabel("Number of Neurons")
            ax.set_xlabel("Response Latency (s)")

        ax.grid(True, alpha=0.3)

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Response Latency Distribution of " + str(n_neurons) + " neurons" + nucname)

        return ax
    else:
        return None

def plot_latencydist(spiketimes_superset, stimulus_onset=None, binsz=None,
                     window=None, neurons=None, nucleus=None, orient=None):
    """
    Visualize Latency Distribution of the given neuron population.

    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param stimulus_onset: float
    - :param binsz: integer or float; defines the number of equal-width bins in the range
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

    ax = plot_latencydist_in_ax(ax, spiketimes_superset, stimulus_onset=stimulus_onset, binsz=binsz,
                                window=window, neurons=neurons, nucleus=nucleus, orient=orient)

    if ax is None:
        print("There are no latencies to plot.")
    else:
        plt.show()

    return fig, ax


##########################################################################
#    CUMULATIVE SPIKE COUNTS PLOT
##########################################################################

def _get_neuron_count(spiketimes_1neuron):
    spiketimes = np.array(spiketimes_1neuron)
    sorted_spikes = np.sort(spiketimes)
    cumulative = np.arange(1, len(sorted_spikes) + 1)

    return sorted_spikes, cumulative

def _get_pop_count(desired_spiketimes_subset):
    all_spikes = np.sort(np.concatenate([spiketimes
                                         for spiketimes in desired_spiketimes_subset
                                         if len(spiketimes) > 0]))
    pop_cumulative = np.arange(1, len(all_spikes) + 1)

    return  all_spikes, pop_cumulative

def plot_spike_counts_distrib_in_ax(ax, spiketimes_superset, neurons=None, nucleus=None, orient=None):
    """
    Draws the Spike Count Distribution on the given
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
    if neurons is None:
        neurons = "all"

    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons="all")

    n_neurons = len(desired_spiketimes_subset)

    if orient=="horizontal":
        for i, indiv_spiketimes in enumerate(desired_spiketimes_subset):
            if len(indiv_spiketimes) > 0:
                [sorted_spikes, indiv_cumulative] = _get_neuron_count(indiv_spiketimes)

                ax.stairs(indiv_cumulative, sorted_spikes, orientation=orient, label="")
    else:
        for i, indiv_spiketimes in enumerate(desired_spiketimes_subset):
            if len(indiv_spiketimes) > 0:
                [sorted_spikes, indiv_cumulative] = _get_neuron_count(indiv_spiketimes)

                ax.step(sorted_spikes, indiv_cumulative, where="post", alpha=0.5, linewidth=1)

    [all_spikes, pop_cumulative] = _get_pop_count(desired_spiketimes_subset)

    if len(all_spikes) > 0:
        ax.step(all_spikes, pop_cumulative, where="post", color="red",
                 linewidth=2, label="Population Total")

    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Spike Count")

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Cumulative Spike Counts Distribution of " + str(n_neurons) + " neurons" + nucname)

    ax.legend()

    return ax

def plot_spike_counts_distrib(spiketimes_superset, neurons=None, nucleus=None, orient=None):
    """
    Visualize Spike Count Distribution of the given neuron population.

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

    ax = plot_spike_counts_distrib_in_ax(ax, spiketimes_superset)

    plt.show()

    return fig, ax


##########################################################################
#    SPIKE DENSITY PLOT
##########################################################################

def _get_neuron_densities(spiketimes_1neuron, time_points, bandwidth):
    spiketimes = np.array(spiketimes_1neuron)
    kde = gaussian_kde(spiketimes, bw_method=bandwidth)

    return kde(time_points)

def _get_pop_densities(desired_spiketimes_subset, time_points, bandwidth):
    all_spiketimes = np.array(desired_spiketimes_subset)
    all_spikes = np.sort(np.concatenate([spiketimes for spiketimes in all_spiketimes if len(spiketimes) > 0]))
    kde = gaussian_kde(all_spikes, bw_method=bandwidth)

    return all_spikes, kde(time_points)

def plot_spike_density_distrib_in_ax(ax, spiketimes_superset, window=(0, 10), bandwidth=0.1):
    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons="all")
    time_points = np.linspace(window[0], window[1], 1000) # have to decide on the number 1000

    for i, indiv_spiketimes in enumerate(desired_spiketimes_subset):
        if len(indiv_spiketimes) > 0:
            neuron_density = _get_neuron_densities(indiv_spiketimes, time_points, bandwidth)

            ax.step(time_points, neuron_density + i*0.1, alpha=0.7, linewidth=1)

    [all_spikes, pop_density] = _get_pop_count(desired_spiketimes_subset)

    if len(all_spikes) > 0:
        ax.step(time_points, pop_density + len(desired_spiketimes_subset)*0.1, "r-",
                linewidth=2, label="Population Average")

    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Density + Offset")
    ax.set_title("Spike Density")
    ax.set_legend()

    return ax

def plot_spike_density_distrib(spiketimes_superset, window=(0, 10), bandwidth=0.1):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax = plot_spike_density_distrib_in_ax(ax, spiketimes_superset, window=window, bandwidth=bandwidth)

    plt.show()

    return fig, ax


##########################################################################
#    ISI PLOT
##########################################################################

def isi_distrib(spiketimes_superset, n_bins=50):
    fig, axes = plt.subplots(1, 2)

    all_isis = []
    for neuron_id, spiketimes in spiketimes_superset.items():
        if len(spiketimes) > 1:
            isis = np.diff(spiketimes)
            all_isis.extend(isis)

            axes[0].hist(isis, bins=n_bins, alpha=0.3, density=True)

    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("Interspike Interval (s)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Neuron ISI Distributions")

    if all_isis:
        axes[1].hist(all_isis, bins=n_bins, alpha=0.7, color="red", density=True)
        axes[1].set_grid(True, alpha=0.3)

        axes[1].set_xlabel("Interspike Interval (s)")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Population ISI Distributions")

    plt.tight_layout()

    plt.show()

    return fig, axes