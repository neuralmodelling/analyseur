# ~/analyseur/cbgt/visual/distribution.py
#
# Documentation by Lungsi 17 Oct 2025
#
# This contains function for SpikingStats
#
"""
=====================
Distribution Plotting
=====================

+--------------------------------------------------+
| Functions                                        |
+==================================================+
| :func:`plot_ratedist`                            |
+--------------------------------------------------+
| :func:`plot_ratedist_in_ax`                      |
+--------------------------------------------------+
| :func:`plot_latencydist`                         |
+--------------------------------------------------+
| :func:`plot_latencydist_in_ax`                   |
+--------------------------------------------------+
| :func:`plot_spike_counts_distrib`                |
+--------------------------------------------------+
| :func:`plot_spike_counts_distrib_in_ax`          |
+--------------------------------------------------+
| :func:`plot_spike_density_distrib`               |
+--------------------------------------------------+
| :func:`plot_spike_density_distrib_line_in_ax`    |
+--------------------------------------------------+
| :func:`plot_spike_density_distrib_stacked_in_ax` |
+--------------------------------------------------+
| :func:`plot_isi_distrib`                         |
+--------------------------------------------------+

1. Pre-requisites
=================

1.1. Import Modules
-------------------
::

    from analyseur.cbgt.loader import LoadSpikeTimes
    from analyseur.cbgt.visual.distribution import <desired_method>

1.2. Load file and get spike times
----------------------------------
::

    loadST = LoadSpikeTimes("spikes_GPi.csv")
    spiketimes_set = loadST.get_spiketimes_superset()

2. Cases
========

2.1. Standard plot
------------------
::

    <desired_method>(spiketimes_set)

2.2. Create the plot for customization
--------------------------------------
This is for power users who for instance want to insert the plot in their
collage of subplots.
::

    import matplotlib.pyplot as plt
    from analyseur.cbgt.visual.distribution import plot_ratedist_in_ax

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')

    ax1 = plot_ratedist_in_ax(ax1, spiketimes_set)
    ax2 = plot_ratedist_in_ax(ax2, spiketimes_set)

    plt.show()

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">
"""
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

def plot_ratedist_in_ax(ax, spiketimes_set, binsz=None, window=None,
                        neurons=None, nucleus=None, orient=None):
    """
    .. code-block:: text

        Population Rate Distribution

        Number of Neurons
        ^
        |                █
        |              ████
        |            ███████
        |          ███████████
        |        ███████████████
        |      ███████████████████
        |            │
        |            │  Mean
        +-------------------------------------------------> Firing Rate (Hz)
        65     70     75     80     85     90     95    100

        Histogram shows the distribution of firing rates
        across neurons. The vertical marker indicates the
        mean population firing rate.

    Draws the Population Rate Distribution on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

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

    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons=neurons)

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

def plot_ratedist(spiketimes_set, binsz=None, window=None,
                  neurons=None, nucleus=None, orient=None):
    """
    Visualize Rate Distribution of the given neuron population using :func:`plot_ratedist_in_ax`

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

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

    ax = plot_ratedist_in_ax(ax, spiketimes_set, binsz=binsz, window=window,
                             neurons=neurons, nucleus=nucleus, orient=orient)

    plt.show()

    return fig, ax


##########################################################################
#    Latency Distribution PLOT
##########################################################################

def plot_latencydist_in_ax(ax, spiketimes_set, stimulus_onset=None, binsz=None,
                           window=None, neurons=None, nucleus=None, orient=None):
    """
    .. code-block:: text

        Response Latency Distribution

        Number of Neurons
        ^
        |            █
        |          ████
        |        ███████
        |      ███████████
        |    ███████████████
        |      ███████████
        |            │
        |            │  Mean
        +-------------------------------------------------> Response Latency (s)
        0.000   0.005   0.010   0.015   0.020   0.025

        Histogram shows the distribution of response latencies
        across neurons. The vertical marker indicates the
        mean response latency.

    Draws the Population Latency Distribution on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

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

    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons=neurons)

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

def plot_latencydist(spiketimes_set, stimulus_onset=None, binsz=None,
                     window=None, neurons=None, nucleus=None, orient=None):
    """
    Visualize Latency Distribution of the given neuron population using :func:`plot_latencydist_in_ax`

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

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

    ax = plot_latencydist_in_ax(ax, spiketimes_set, stimulus_onset=stimulus_onset, binsz=binsz,
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

def plot_spike_counts_distrib_in_ax(ax, spiketimes_set, neurons=None, nucleus=None, orient=None):
    """
    .. code-block:: text

        Cumulative Spike Count
        ^
        |                             ███████████████████
        |                        █████
        |                   █████
        |              █████
        |         █████
        |    █████
        | █████
        |
        +-------------------------------------------------> Time (s)
        0          2          4          6          8         10

        Thin step curves represent cumulative spike counts
        for individual neurons.

        The thicker curve represents the total cumulative
        spike count of the entire neuron population.

    Draws the Spike Count Distribution on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

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

    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons="all")

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

def plot_spike_counts_distrib(spiketimes_set, neurons=None, nucleus=None, orient=None):
    """
    Visualize Spike Count Distribution of the given neuron population using :func:`plot_spike_counts_distrib_in_ax`

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

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

    ax = plot_spike_counts_distrib_in_ax(ax, spiketimes_set)

    plt.show()

    return fig, ax


##########################################################################
#    SPIKE DENSITY PLOT
##########################################################################
def _get_neuron_densities(spiketimes_1neuron, time_points, bandwidth):
    spiketimes = np.array(spiketimes_1neuron)

    if len(spiketimes) < 2:
        return np.zeros_like(time_points)

    kde = gaussian_kde(spiketimes, bw_method=bandwidth)
    return kde(time_points)

def _get_pop_densities(desired_spiketimes_subset, time_points, bandwidth):

    all_spikes = np.sort(
        np.concatenate([
            spiketimes for spiketimes in desired_spiketimes_subset
            if len(spiketimes) > 0
        ])
    )

    kde = gaussian_kde(all_spikes, bw_method=bandwidth)

    return all_spikes, kde(time_points)

def plot_spike_density_distrib_stacked_in_ax(ax, spiketimes_set,
                                             window=(0, 10), bandwidth=0.1,
                                             max_neurons=20):
    """
    .. code-block:: text

        Density + Offset
        ^

        |      ───────────── population density
        |
        |      ~~~~~~~~ neuron 10
        |      ~~~~~~~~ neuron 9
        |         :       :    :
        |      ~~~~~~~~ neuron 3
        |      ~~~~~~~~ neuron 2
        |      ~~~~~~~~ neuron 1
        |
        +------------------------------------> Time

    Visualize Spike Density Distribution of the given neuron population and
    plot `max_neurons` such that plot of one neuron is stacked over another.

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    OPTIONAL parameters

    - :param window: 2-tuple; defines upper and lower range of the bins
    - :param bandwidth: `0.1` [default]
    - :param max_neurons: `20` [default]
    - :return: object `ax` with Spike Density Distribution plot done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons="all")
    time_points = np.linspace(window[0], window[1], 1000) # have to decide on the number 1000

    if len(desired_spiketimes_subset) > max_neurons:
        desired_spiketimes_subset = desired_spiketimes_subset[:max_neurons]

    densities = []
    for spks in desired_spiketimes_subset:
        if len(spks) > 1:
            densities.append(_get_neuron_densities(spks, time_points, bandwidth))

    if not densities:
        return ax

    max_density = max(np.max(d) for d in densities)
    offset_step = max_density * 2

    for i, neuron_density in enumerate(densities):
        ax.step(time_points, neuron_density + i * offset_step,
                linewidth=1, alpha=0.7)

    all_spikes, pop_density = _get_pop_densities(desired_spiketimes_subset,
                                                 time_points, bandwidth)

    ax.step(time_points, pop_density + len(densities) * offset_step,
            "r-", linewidth=2, label="Population Average")

    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Density + Offset")
    ax.set_title("Spike Density")
    ax.legend()

    return ax

def plot_spike_density_distrib_line_in_ax(ax, spiketimes_set,
                                          window=(0, 10), bandwidth=0.1):
    """
    .. code-block:: text

        Population Spike Density

        Density
        ^
        |                ________
        |             __--      --__
        |           _--            --_
        |         _--                --_
        |       _--                    --_
        |______--                        --______
        |
        +-------------------------------------------------> Time (s)
        0          2          4          6          8         10

        Smooth red curve represents the population spike
        density estimated using kernel density estimation
        (KDE) over all spikes in the neuron population.

    Visualize Spike Density Distribution of the given neuron population.

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    OPTIONAL parameters

    - :param window: 2-tuple; defines upper and lower range of the bins
    - :param bandwidth: `0.1` [default]
    - :return: object `ax` with Spike Density Distribution plot done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons="all")
    time_points = np.linspace(window[0], window[1], 1000) # have to decide on the number 1000

    _, pop_density = _get_pop_densities(desired_spiketimes_subset, time_points, bandwidth)

    ax.plot(time_points, pop_density, color="red", linewidth=2)

    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Density")
    ax.set_title("Population Spike Density")
    ax.legend()

    return ax

def plot_spike_density_distrib(spiketimes_set, window=(0, 10), bandwidth=0.1, plot_type="line"):
    """
    Visualize Spike Density Distribution of the given neuron population using :func:`plot_spike_density_distrib_line_in_ax` or :func:`plot_spike_density_distrib_stacked_in_ax`
    (depending on the `plot_type`).

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    OPTIONAL parameters

    - :param window: 2-tuple; defines upper and lower range of the bins
    - :param bandwidth: `0.1` [default]
    - :param plot_type: `"line"` [default] or `"stacked"`
    - :return: object `ax` with spike density distribution plot done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type=="line":
        ax = plot_spike_density_distrib_line_in_ax(ax, spiketimes_set,
                                                   window=window, bandwidth=bandwidth)
    else:
        ax = plot_spike_density_distrib_in_ax(ax, spiketimes_set,
                                              window=window, bandwidth=bandwidth)

    plt.show()

    return fig, ax


##########################################################################
#    ISI PLOT
##########################################################################

def plot_isi_distrib(spiketimes_set, n_bins=50):
    """
    .. code-block:: text

        Interspike Interval Distributions

        ┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
        │ Neuron ISI Distributions            │   │ Population ISI Distributions        │
        │                                     │   │                                     │
        │ Density                             │   │ Density                             │
        │ ^                                   │   │ ^                                   │
        │ |      ████                         │   │ |       ████                        │
        │ |    ████████                       │   │ |     ████████                      │
        │ |   ███████████                     │   │ |   ███████████                     │
        │ |    █████████                      │   │ |     █████████                     │
        │ |      █████                        │   │ |       █████                       │
        │ |                                   │   │ |                                   │
        │ +---------------------------------> │   │ +---------------------------------> │
        │     Interspike Interval (s)         │   │     Interspike Interval (s)         │
        └─────────────────────────────────────┘   └─────────────────────────────────────┘

        Left: overlapping ISI histograms for individual neurons.
        Right: histogram of ISIs pooled across the entire population.

    Visualize Distribution of Inter-Spike Interval of the given neuron population.

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    OPTIONAL parameters

    - :param n_bins: `50` [default]
    - :return: object `ax` with ISI Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    fig, axes = plt.subplots(1, 2)

    all_isis = []
    per_neuron_isis = []

    for neuron_id, spiketimes in spiketimes_set.items():
        if len(spiketimes) > 1:
            isis = np.diff(spiketimes)
            per_neuron_isis.append(isis)
            all_isis.extend(isis)

    if not all_isis:
        return fig, axes

    bins = np.linspace(min(all_isis), max(all_isis), n_bins + 1)

    for isis in per_neuron_isis:
        axes[0].hist(isis, bins=bins, alpha=0.3, density=True)

    axes[1].hist(all_isis, bins=bins, alpha=0.7, color="red", density=True)

    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    axes[0].set_xlabel("Interspike Interval (s)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Neuron ISI Distributions")

    axes[1].set_xlabel("Interspike Interval (s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Population ISI Distributions")

    plt.tight_layout()
    plt.show()

    return fig, axes
