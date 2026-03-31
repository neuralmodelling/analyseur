# ~/analyseur/rbcbg/visual/rate.py
#
# Documentation by Lungsi 18 Nov 2025
#
# This contains function for Visualizing Rates
#

"""
+-------------------------------------+----------------------------------------------------------------------------------------+
| Functions                           | Purpose                                                                                |
+=====================================+========================================================================================+
| :func:`plot_rate_all_neurons_in_ax` | plots Rate (1/s) of each neurons in a population into a given `matplotlib.pyplot.axis` |
+-------------------------------------+----------------------------------------------------------------------------------------+
| :func:`plot_mean_rate_in_ax`        | draws the Mean Rate (1/s) of all the neurons into a given `matplotlib.pyplot.axis`     |
+-------------------------------------+----------------------------------------------------------------------------------------+

===============
Plot Mean Rate
===============

-----------------
1. Pre-requisites
-----------------

1.1. Import Modules
````````````````````
::

    from analyseur.rbcbg.loader import LoadRates
    from analyseur.rbcbg.visual.rate import plot_rate_all_neurons_in_ax, plot_mean_rate_in_ax


1.2. Load file and get firing rates
```````````````````````````````````
::

    loadFR = LoadRates("GPiSNr_model_9_percent_0.csv")
    t_sec, rates_Hz = loadFR.get_rates()

---------
2. Cases
---------

2.1. Visualize All Firing Rates with default setting
````````````````````````````````````````````````````
::

    fig, ax = plt.subplots(figsize=(6, 10))

    ax = plot_rate_all_neurons_in_ax(ax, t_sec, rates_Hz, nucleus="GPiSNr")

    plt.show()

2.2. Visualize All Firing Rates within a desired window
```````````````````````````````````````````````````````
::

    window = (0,1)  # first second

    fig, ax = plt.subplots(figsize=(6, 10))

    ax = plot_rate_all_neurons_in_ax(ax, t_sec, rates_Hz, nucleus="GPiSNr", window=window)

    plt.show()

2.3. Visualize Mean Firing rate
```````````````````````````````
::

    fig, ax = plt.subplots(figsize=(6, 10))

    ax = plot_mean_rate_in_ax(ax, t_sec, rates_Hz, nucleus="GPiSNr")

    plt.show()


.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">

"""

from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, alpha

import re

from analyseur.rbcbg.curate import filter_rates
from analyseur.rbcbg.parameters import SignalAnalysisParams, SimulationParams

__siganal = SignalAnalysisParams()
__simparams = SimulationParams()


##########################################################################
#    PLOT Instantaneous Rate
##########################################################################

def plot_rate_all_neurons_in_ax(ax, times_array, rates_array,
                                nucleus=None, window=None):
    """
    .. code-block:: text

        Firing Rate (Hz)
        │
        │ 0.4 ┤            /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
        │     │           /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
        │     │          /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
        │ 0.3 ┤   ~~~~~~/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/
        │     │  ~~~~~~/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
        │     │ ~~~~~~/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/
        │ 0.2 ┤~~~~~~/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
        │     │~~~~~/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/
        │     │~~~~/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
        │ 0.1 ┤~~~/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/
        │     │  /
        │     │ /
        │ 0.0 ┼─┴──────────────────────────────────────────────────→ Time (s)
                0        2        4        6        8        10

        (many neurons → overlapping oscillatory traces with slight offsets;
        initial transient → synchronized rhythmic activity)

    Given a `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_ this draws
    the firing rates (1/s) for each neurons in a given population

    :param ax: object `matplotlib.pyplot.axis`
    :param times_array: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
    :param mu_rate_array: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`

    [OPTIONAL]

    :param nucleus: string; name of the nucleus
    :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
    :return: ax with respective plotting

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = __siganal.window

    n_neurons = rates_array.shape[1]

    filtered_t, filtered_r = filter_rates(times_sec=times_array,
                                          rates_Hz=rates_array,
                                          window=window)

    # ---- Colors & styles ----
    colors = plt.cm.tab20(np.linspace(0, 1, n_neurons))
    line_styles = cycle(["-", "--", "-.", ":"])

    # ---- Plot each channel ----
    for i in range(n_neurons):
        style = next(line_styles)
        color = colors[i % len(colors)]

        ax.plot(filtered_t, filtered_r[:, i], linestyle=style,
                color=color, linewidth=1.5, label=f"Neuron {i}")

    # ---- Formatting ----
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Firing Rate (Hz)")

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Firing Rate Over Time (All Neurons)" + nucname)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return ax

def plot_mean_rate_in_ax(ax, times_array, rates_array,
                         nucleus=None, window=None):
    """
    .. code-block:: text

        Firing Rate (Hz)
        │
        │ 4.0 ┤        ▲
        │     │       / \\
        │ 3.0 ┤      /   \\        /\\    /\\    /\\    /\\    /\
        │     │     /     \\      /  \\  /  \\  /  \\  /  \\  /  \
        │ 2.0 ┤    /       \\    /    \\/    \\/    \\/    \\/    \
        │     │   /         \\__/
        │ 1.0 ┤__/
        │
        └────────────────────────────────────────────────────→ Time (s)
        0        2        4        6        8        10


    Given a `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_ this draws
    the firing rates (1/s) averaged across all neurons in a given population

    :param ax: object `matplotlib.pyplot.axis`
    :param times_array: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
    :param mu_rate_array: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`

    [OPTIONAL]

    :param nucleus: string; name of the nucleus
    :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
    :return: ax with respective plotting

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = __siganal.window

    filtered_t, filtered_r = filter_rates(times_sec=times_array,
                                          rates_Hz=rates_array,
                                          window=window)

    filtered_r_mean = filtered_r.mean(axis=1)

    # ---- Plot the mean rate ----
    ax.plot(filtered_t, filtered_r_mean, linewidth=1.5)

    # ---- Formatting ----
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Firing Rate (Hz)")

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Mean Firing Rate Over Time" + nucname)

    return ax

