# ~/analyseur/cbgt/visual/markerplot.py
#
# Documentation by Lungsi 29 Oct 2025
#
# This contains function for SpikingStats
#
"""
===============
Marker Plotting
===============

+------------------------------+-----------------------------------------------------------------------------------------------------+
| Functions                    | Purpose                                                                                             |
+==============================+=====================================================================================================+
| :func:`plot_raster`          | plots Coefficient of Variations of all the neurons in a population                                  |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| :func:`plot_ratechange`      | draws the Coefficient of Variations of all the neurons into a given `matplotlib.pyplot.axis`        |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| :func:`plot_ratechange_in_ax`| plots Local Coefficient of Variations of all the neurons in a population                            |
+------------------------------+-----------------------------------------------------------------------------------------------------+

--------------------------
Raster Plot of Spike Times
--------------------------

1. Pre-requisites
=================

1.1. Import Modules
-------------------
::

    from analyseur.cbgt.loader import LoadSpikeTimes
    from analyseur.cbgt.visual.markerplot import plot_raster

1.2. Load file and get spike times
----------------------------------
::

    loadST = LoadSpikeTimes("spikes_GPi.csv")
    spiketimes_superset = loadST.get_spiketimes_superset()

2. Cases
========

2.1. Raster for all the neurons
-------------------------------
::

    plot_raster(spiketimes_superset)

2.2. Raster for first 50 neurons
--------------------------------
::

    plot_raster(spiketimes_superset, neurons=range(50))

2.3. Raster for second 50 neurons
---------------------------------
::

    plot_raster(spiketimes_superset, neurons=range(50, 100))

2.4. Create the plot for customization
``````````````````````````````````````
This is for power users who for instance want to insert the raster plot in their
collage of subplots.
::

    import matplotlib.pyplot as plt
    from analyseur.cbgt.visual.markerplot import plot_raster_in_ax

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')

    ax1 = plot_raster_in_ax(ax1, spiketimes_superset)
    ax2 = plot_raster_in_ax(ax2, spiketimes_superset)

    plt.show()

NOTE: This example shows :func:`plot_raster_in_ax` in default setting but this function works like
:func:`plot_raster` therefore all the cases 2.1, 2.2 and 2.3 are applicable for :func:`plot_raster_in_ax`.

------------------------
Plot Rate Change Scatter
------------------------

1. Pre-requisites
=================

1.1. Import Modules
-------------------
::

    from analyseur.cbgt.loader import LoadSpikeTimes
    from analyseur.cbgt.visual.markerplot import plot_ratechange

1.2. Load file and get spike times
----------------------------------
::

    loadST = LoadSpikeTimes("spikes_GPi.csv")
    spiketimes_superset = loadST.get_spiketimes_superset()

2. Cases
========

2.1. Plot Rate Change Scatter for all the neurons
-------------------------------------------------
::

    plot_ratechange(spiketimes_superset)

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">

"""

import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.parameters import SignalAnalysisParams, SimulationParams

__siganal = SignalAnalysisParams()
__simparams = SimulationParams()


##########################################################################
#    Raster Plot
##########################################################################

def _get_line_colors(colors=False, no_neurons=None):
    if colors:
        return [f'C{i}' for i in range(no_neurons)]  # set different colors for each set of positions
    else:
        return "black"

def plot_raster_in_ax(ax, spiketimes_superset, colors=False, neurons=None, nucleus=None,):
    """
    Draws the Rasterplot (`matplotlib.pyplot.eventplot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.eventplot.html>`_)
    on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_superset: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`

    OPTIONAL parameters

    :param colors: `False` [default] or True
    :param neurons: `"all"` [default] or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]
    :param nucleus: string; name of the nucleus
    :return: object `ax` with Raster plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    # ============== DEFAULT Parameters ==============
    if neurons is None:
        neurons = "all"

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

    return ax

def plot_raster(spiketimes_superset, colors=False, neurons=None, nucleus=None,):
    """
    Visualize Raster plot for the given neuron population.

    :param spiketimes_superset: Dictionary returned using :class:`~analyseur.cbgt.loader.LoadSpikeTimes`

    OPTIONAL parameters

    :param colors: `False` [default] or True
    :param neurons: `"all"` [default] or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]
    :param nucleus: string; name of the nucleus
    :return: object `ax` with Raster plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    fig, ax = plt.subplots(figsize=(18, 12))

    ax = plot_raster_in_ax(ax, spiketimes_superset, colors=colors, neurons=neurons, nucleus=nucleus,)

    plt.show()

    return fig, ax


##########################################################################
#    Rate Change SCATTER
##########################################################################

def plot_ratechange_in_ax(ax, spiketimes_superset, stimulus_onset=None,
                           window=None, neurons=None, nucleus=None, mode=None):
    """
    Draws the Population Rate Change Scatter on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    :param stimulus_onset: float; 0 [default]
    :param window: 2-tuple; (0, 10) [default]
    :param neurons: "all" [default] or list: range(a, b) or [1, 4, 5, 9]
    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    # ============== DEFAULT Parameters ==============
    if neurons is None:
        neurons = "all"

    if window is None:
        window = __siganal.window

    if stimulus_onset is None:
        stimulus_onset = 0

    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)

    n_neurons = len(desired_spiketimes_subset)

    match mode:
        case "portrait":
            orient = "horizontal"
        case _:
            orient = "landscape"

    get_axis = lambda orient: "x" if orient == "horizontal" else "y"

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

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Rate Change: Baseline vs. Response of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plot_ratechange(spiketimes_superset, stimulus_onset=None,
                     window=None, neurons=None, nucleus=None, mode=None):
    """
    Visualize Rate Change Scatter of the given neuron population.

    :param spiketimes_superset: Dictionary returned using :class:`~analyseur.cbgt.loader.LoadSpikeTimes`

    OPTIONAL parameters

    :param stimulus_onset: float
    :param window: 2-tuple; defines upper and lower range of the bins
    :param neurons: "all" or list: range(a, b) or [1, 4, 5, 9]
    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    if mode == "portrait":
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = plot_ratechange_in_ax(ax, spiketimes_superset, stimulus_onset=stimulus_onset,
                                window=window, neurons=neurons, nucleus=nucleus, mode=mode)

    if ax is None:
        print("There are no latencies to plot.")
    else:
        plt.show()

    return fig, ax