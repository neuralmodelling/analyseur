# ~/analyseur/cbgt/visual/rate.py
#
# Documentation by Lungsi 30 Oct 2025
#
# This contains function for SpikingStats
#

"""
+------------------------------+--------------------------------------------------------------------------------------+
| Functions                    | Purpose                                                                              |
+==============================+======================================================================================+
| :func:`plot_mean_rate`       | plots Mean Rate (1/s) of all the neurons in a population                             |
+------------------------------+--------------------------------------------------------------------------------------+
| :func:`plot_mean_rate_in_ax` | draws the Mean Rate (1/s) of all the neurons into a given `matplotlib.pyplot.axis`   |
+------------------------------+--------------------------------------------------------------------------------------+

===============
Plot Mean Rate
===============

-----------------
1. Pre-requisites
-----------------

1.1. Import Modules
````````````````````
::

    from analyseur.cbgt.loader import LoadSpikeTimes
    from analyseur.cbgt.visual.rate import plot_mean_rate

1.2. Load file and get spike times
```````````````````````````````````
::

    loadST = LoadSpikeTimes("spikes_GPi.csv")
    spiketimes_superset = loadST.get_spiketimes_superset()

---------
2. Cases
---------

2.1. Visualize Mean Rate with default setting
``````````````````````````````````````````````
::

    [fig, ax] = plot_mean_rate(spiketimes_superset)

2.2. Visualize Mean Rate in portrait mode
``````````````````````````````````````````
::

    [fig, ax] = plot_mean_rate(spiketimes_superset, mode="portrait")

2.3. Visualize Mean Rate in portrait mode with nucleus name in title
````````````````````````````````````````````````````````````````````
::

    [fig, ax] = plot_mean_rate(spiketimes_superset, mode="portrait", nucleus="GPi")

2.4. Create the plot for customization
``````````````````````````````````````
This is for power users who for instance want to insert the Mean Rate plot in their
collage of subplots.
::

    import matplotlib.pyplot as plt
    from analyseur.cbgt.visual.rate import plot_mean_rate_in_ax

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')

    ax1 = plot_mean_rate_in_ax(ax1, spiketimes_superset)
    ax2 = plot_mean_rate_in_ax(ax2, spiketimes_superset)

    plt.show()

NOTE: This example shows :func:`plot_mean_rate_in_ax` in default setting but this function works like
:func:`plot_mean_rate` therefore all the cases 2.1, 2.2 and 2.3 are applicable for :func:`plot_mean_rate_in_ax`.

===============================
Plot Average Instantaneous Rate
===============================

Similar as documented above for plotting Mean Rate but using the function
:func:`plot_avg_inst_rate` and :func:`plot_avg_inst_rate_in_ax` with the
additional OPTIONAL argument for `binsz` (otherwise it picks a default value).
This is imported as
::

    from analyseur.cbgt.visual.rate import plot_avg_inst_rate, plot_avg_inst_rate_in_ax

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">

"""

import numbers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, alpha

import re

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.stats.rate import Rate
from analyseur.cbgt.stats.isi import InterSpikeInterval
from analyseur.cbgt.parameters import SignalAnalysisParams, SimulationParams

__siganal = SignalAnalysisParams()
__simparams = SimulationParams()


##########################################################################
#    PLOT Mean Rate Based on Spike Counts
##########################################################################

def plot_mean_rate_spikecounts_in_ax(ax, spiketimes_set, window=None, binsz=None,
                                     nucleus=None, mode=None):
    """
    Draws the Mean Rate (1/s) on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    OPTIONAL parameters

    - :param nucleus: string; name of the nucleus
    - :param mode: "portrait" or None/landscape [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = __siganal.window

    if binsz is None:
        binsz = __siganal.binsz_100perbin

    # if neurons is None:
    #     neurons = "all"
    # elif isinstance(neurons, numbers.Number):
    #
    # if neurons == "all":
    #     spiketimes_set = spiketimes_superset
    # elif isinstance(neurons, numbers.Number):
    #     spiketimes_set = dict(list(spiketimes_set.items())[:neurons])  # first N = neurons
    # else:
    #     keys_to_remove = ["n" + str(i) for i in neurons]
    #
    #     # Convert to set for faster lookup
    #     remove_set = set(keys_to_remove)
    #
    #     spiketimes_set = {k: v for k, v in spiketimes_superset.items() if k not in remove_set}

    n_neurons = len(spiketimes_set)

    match mode:
        case "portrait":
            orient = "horizontal"
        case _:
            orient = "landscape"

    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    _, rate_matrix, _ = Rate.get_count_rate_matrix(spiketimes_set=spiketimes_set, window=window, binsz=binsz,)
    mu_rate_vec = rate_matrix.mean(axis=1)

    print(len(mu_rate_vec))

    if orient == "horizontal":
        ax.barh(range(len(mu_rate_vec)), mu_rate_vec, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
        ax.set_xlabel("Mean Rate (1/s)")
        ax.margins(y=0)
    else:
        ax.bar(range(len(mu_rate_vec)), mu_rate_vec, color="steelblue", edgecolor="black")
        ax.set_ylabel("Mean Rate (1/s)")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Mean Rate Distribution of " + str(n_neurons) + " neurons" + nucname)

    return ax


##########################################################################
#    PLOT Mean Rate
##########################################################################

def plot_mean_rate_in_ax(ax, spiketimes_set, nucleus=None, mode=None):
    """
    Draws the Mean Rate (1/s) on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    OPTIONAL parameters

    - :param nucleus: string; name of the nucleus
    - :param mode: "portrait" or None/landscape [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    n_neurons = len(spiketimes_set)

    match mode:
        case "portrait":
            orient = "horizontal"
        case _:
            orient = "landscape"

    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    [all_isi, _] = InterSpikeInterval.compute(spiketimes_set)
    mu_arr = InterSpikeInterval.mean_freqs(all_isi)
    vec_mu = mu_arr.values()

    if orient == "horizontal":
        ax.barh(range(len(vec_mu)), vec_mu, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
        ax.set_xlabel("Mean Rate (1/s)")
        ax.margins(y=0)
    else:
        ax.bar(range(len(vec_mu)), vec_mu, color="steelblue", edgecolor="black")
        ax.set_ylabel("Mean Rate (1/s)")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Mean Rate Distribution of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plot_mean_rate(spiketimes_superset, nucleus=None, mode=None):
    """
    Visualize Mean Rate (1/s) of the given neuron population.

    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param nucleus: string; name of the nucleus
    - :param mode: "portrait" or None/landscape [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    if mode=="portrait":
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = plot_mean_rate_in_ax(ax, spiketimes_superset, nucleus=nucleus, mode=mode)

    plt.show()

    return fig, ax


##########################################################################
#    PLOT Average Instantaneous Rate
##########################################################################

def plot_pool_avg_inst_rates_in_ax(ax, spiketimes_superset, binsz=None, nucleus=None, mode=None):
    """
    Draws the Mean Rate (1/s) on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param binsz: 0.01 [default]
    - :param nucleus: string; name of the nucleus
    - :param mode: "portrait" or None/landscape [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    if binsz is None:
        binsz = __siganal.binsz_100perbin

    n_neurons = len(spiketimes_superset)

    match mode:
        case "portrait":
            orient = "horizontal"
        case _:
            orient = "landscape"

    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    [all_isi, all_times] = InterSpikeInterval.compute(spiketimes_superset)
    all_inst = InterSpikeInterval.inst_rates(all_isi)
    [avg_rates, bin_centers, bin_counts] = InterSpikeInterval.pool_avg_inst_rates(inst_rates_set=all_inst,
                                                                                  tbins_set=all_times,
                                                                                  binsz=binsz)
    if orient == "horizontal":
        ax.barh(bin_centers, avg_rates, height=binsz*0.8, linewidth=0.5,
                alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_ylabel("Time (s)")
        ax.set_xlabel("Average Inst. Rate (Hz)")
    else:
        # Base bar
        ax.bar(bin_centers, avg_rates, width=binsz*0.8, linewidth=0.5,
               alpha=0.7, color="steelblue", edgecolor="black")
        # ax.plot(bin_centers, avg_rates, "o-", linewidth=2, markersize=6)  # Plot
        ax.set_ylabel("Average Inst. Rate (Hz)")
        ax.set_xlabel("Time (s)")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Pooled Average Inst. Rates of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plot_pool_avg_inst_rates(spiketimes_superset, binsz=None, nucleus=None, mode=None):
    """
    Visualize Mean Rate (1/s) of the given neuron population.

    :param spiketimes_superset: Dictionary returned using :class:`~analyseur.cbgt.loader.LoadSpikeTimes`

    OPTIONAL parameters

    - :param binsz: 0.01 [default]
    - :param nucleus: string; name of the nucleus
    - :param mode: "portrait" or None/landscape [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    if mode=="portrait":
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = plot_pool_avg_inst_rates_in_ax(ax, spiketimes_superset, binsz=binsz, nucleus=nucleus, mode=mode)

    plt.show()

    return fig, ax


##########################################################################
#    PLOT Instantaneous Rate
##########################################################################

def plot_true_avg_inst_rate_in_ax(ax, spiketimes_set, nucleus=None, mode=None):
    """
    Draws the Instantaneuous Rate (1/s) on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    OPTIONAL parameters

    :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    n_neurons = len(spiketimes_set)

    match mode:
        case "portrait":
            orient = "horizontal"
        case _:
            orient = "landscape"

    get_axis = lambda orient: "x" if orient == "horizontal" else "y"

    [all_isi, _] = InterSpikeInterval.compute(spiketimes_set)
    all_inst_rates = InterSpikeInterval.inst_rates(isi_set=all_isi)
    Instarr = InterSpikeInterval.true_avg_inst_rates(inst_rates_set=all_inst_rates)
    vec_IR = Instarr.values()

    #print(f"Inst. Rate Array {Instarr}")

    if orient == "horizontal":
        ax.barh(range(len(vec_IR)), vec_IR, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
        ax.set_xlabel("Avg. Ins. Rate (1/s)")
        ax.margins(y=0)
    else:
        ax.bar(range(len(vec_IR)), vec_IR, color="steelblue", edgecolor="black")
        ax.set_ylabel("Avg. Ins. Rate (1/s)")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Avg. Inst. Rate Distribution of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plot_true_avg_inst_rate(spiketimes_set, nucleus=None, mode=None):
    """
    Visualize Instantaneuous Rate (1/s) of the given neuron population.

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    OPTIONAL parameters

    :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    if mode=="portrait":
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = plot_true_avg_inst_rate_in_ax(ax, spiketimes_set, nucleus=nucleus, mode=mode)

    plt.show()

    return fig, ax

def plot_mean_rate_all_neurons_across_time_in_ax(ax, spiketimes_set, window=None,
                                                 binsz=None, nucleus=None,):
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = __siganal.window

    if binsz is None:
        binsz = __siganal.binsz_100perbin

    n_neurons = len(spiketimes_set)

    mu_rate_vec, time_bins = Rate.mean_rate(spiketimes_set=spiketimes_set, window=window,
                                            binsz=binsz, neurons="all")

    # Plot
    ax.plot(np.arange(len(mu_rate_vec)) / (window[1] - window[0]),
            mu_rate_vec, "b-", linewidth=1)
    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Firing Rate (Hz)")

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Firing Rate Over Time of " + str(n_neurons) + " neurons" + nucname)

    return ax