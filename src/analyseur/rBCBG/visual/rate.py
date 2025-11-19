# ~/analyseur/rbcbg/visual/rate.py
#
# Documentation by Lungsi 18 Nov 2025
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

from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, alpha

import re

from analyseur.rbcbg.curate import filter_rates, filter_rates_set
from analyseur.rbcbg.parameters import SignalAnalysisParams, SimulationParams

__siganal = SignalAnalysisParams()
__simparams = SimulationParams()


##########################################################################
#    PLOT Instantaneous Rate
##########################################################################

def plot_rate_all_channels_across_time_in_ax(ax, rates_set, window=None,
                                                 nucleus=None,):
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = __siganal.window

    filtered_set = filter_rates_set(rates_set=rates_set, window=window)

    # Generate colors from colormap
    num_colors = len(filtered_set)
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
    line_styles = cycle(["-", "--", "-.", ":"])
    # Plot
    for i, (key, array) in enumerate(rates_set.items()):
        style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        ax.plot(array, label=key, linestyle=style, color=color, linewidth=2)
    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Firing Rate (Hz)")

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Firing Rate Over Time for all channels of " + " neurons" + nucname)

    return ax

def plot_mean_rate_all_channels_across_time_in_ax(ax, mu_rate_arr, window=None, nucleus=None,):
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = __siganal.window

    t_end_ms = window[1] * __siganal._1000ms
    t_axis = np.arange(t_end_ms - 1) / __siganal._1000ms

    mu_rate_vec = filter_rates(rates_array=mu_rate_arr, window=window)

    # Plot
    ax.plot(t_axis, mu_rate_vec, "b-", linewidth=1)
    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Firing Rate (Hz)")

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("Firing Rate Over Time of " + " neurons" + nucname)

    return ax
