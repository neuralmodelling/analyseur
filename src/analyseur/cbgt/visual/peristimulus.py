# ~/analyseur/cbgt/visual/peristimulus.py
#
# Documentation by Lungsi 8 Oct 2025
#
# This contains function for Peri-Stimulus Time Histogram (PSTH)
#

import numpy as np
import matplotlib.pyplot as plt

# from ..curate import get_desired_spiketimes_subset
from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.stats.psth import PSTH

class VizPSTH(object):
    """
    The Peri-Stimulus Time Histogram (PSTH) Class is instantiated by passing

    :param spiketimes_superset: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    
    +--------------------------------+--------------------------------------------------------------------+
    | Methods                        | Return                                                             |
    +================================+====================================================================+
    | :py:meth:`.plot`               | - `matplotlib.pyplot.hist` object                                  |
    +--------------------------------+--------------------------------------------------------------------+
    | :py:meth:`.plot_in_ax`         | - `matplotlib.pyplot.axis` object                                  |
    +--------------------------------+--------------------------------------------------------------------+

    * PSTH gives an overall temporal pattern of population activity with a picture in both temporal and rate
    * The computation is done by :class:`~analyseur.cbgt.stats.psth.PSTH`
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spiketimes_superset = loadST.get_spiketrains()

      from analyseur.cbgt.visual.peristimulus import vizPSTH

      my_psth = vizPSTH(spiketimes_superset)

    2. Peri-Stimulus Time Histogram for the whole simulation window

    ::

      my_psth.plot()

    3. PSTH for desired window and bin size

    ::

      my_psth.plot(window=(0,5), binsz=1)  # time unit in seconds
      my_psth.plot(window=(0,5), binsz=0.05)

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """

    def __init__(self, spiketimes_superset):
        self.spiketimes_superset = spiketimes_superset

    @staticmethod
    def plot_in_ax(ax, spiketimes_superset, binsz=None, window=(), neurons=None, nucleus=None):
        """
        Draws the Peri-Stimulus Time Histogram (PSTH) on the given
        `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

        :param ax: object `matplotlib.pyplot.axis``
        :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :param binsz: integer or float; defines the number of equal-width bins in the range
        :param window: 2-tuple; defines upper and lower range of the bins
        :param neurons: "all" or list: range(a, b) or [1, 4, 5, 9]
        :param nucleus: string; name of the nucleus
        :return: object `ax` with PSTH plotting done into it

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # Compute PSTH
        [counts, bin_centers, popfirerates, true_avg_rate] = PSTH.compute_poolPSTH(spiketimes_superset, neurons=neurons,
                                                                                    binsz=binsz, window=window)
        n_neurons = len(true_avg_rate["firing_rates"])

        # Plot
        ax.bar(bin_centers, counts, width=binsz, alpha=0.7, color="blue", edgecolor="black")
        ax.grid(True, alpha=0.3)

        ax.set_ylabel("Spike Count")
        ax.set_xlabel("Time (s)")

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("PSTH - Population Activity of " + str(n_neurons) + " neurons" + nucname +
                     "\n (mean firing rate within the window = "
                     + str(true_avg_rate["mean_firing_rate"]) + " Hz)")

        return ax


    def plot(self, binsz=0.01, window=(0, 10), neurons="all", nucleus=None, show=True):
        """
        Displays the Peri-Stimulus Time Histogram (PSTH) of the given spike times (seconds)
        and returns the plot figure (to save if necessary).
        
        :param binsz: integer or float; defines the number of equal-width bins in the range
        :param window: 2-tuple; defines upper and lower range of the bins
        :param neurons: "all" or list: range(a, b) or [1, 4, 5, 9]
        :param nucleus: string; name of the nucleus
        :param show: boolean [default: True]
        :return: object `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_
        containing `matplotlib.pyplot.bar <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_
        
        * `window` controls the binning range as well as the spike counting window
        * CBGT simulation was done in seconds so window `(0, 10)` signifies time 0 s to 10 s

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        
        """
        # Set binsz and window as the instance attributes
        self.binsz = binsz
        self.window = window

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = self.plot_in_ax(ax, self.spiketimes_superset, binsz=binsz, window=window, neurons=neurons, nucleus=nucleus)

        if show:
            plt.show()
        
        return fig, ax

