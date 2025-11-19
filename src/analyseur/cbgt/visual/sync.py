# ~/analyseur/cbgt/visual/sync.py
#
# Documentation by Lungsi 12 Nov 2025
#

import numpy as np
import matplotlib.pyplot as plt

from analyseur.cbgt.parameters import SignalAnalysisParams
from analyseur.cbgt.stats.sync import Synchrony

class VizSynchrony(object):
    __siganal = SignalAnalysisParams()

    @classmethod
    def plot_ci_in_ax(cls, ax, spiketimes_set, binsz=None, window=None, neurons=None, nucleus=None):
        pass

    @classmethod
    def plot_ci(cls, spiketimes_set, binsz=0.01, window=(0, 10), neurons="all", nucleus=None, show=True):
        """
        Displays the Pooled Peri-Stimulus Time Histogram (PSTH) of the given spike times (seconds)
        and returns the plot figure (to save if necessary).

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: integer or float; `0.01` [default]
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

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
        cls.binsz = binsz
        cls.window = window

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = cls.plot_ci_in_ax(ax, spiketimes_set, binsz=binsz, window=window,
                               neurons=neurons, nucleus=nucleus)

        if show:
            plt.show()

        return fig, ax


    @classmethod
    def plot_spike_corr_in_ax(cls, ax, spiketimes_set, binsz=None, window=None, nucleus=None):
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        n_neurons = len(spiketimes_set)

        correlations, pairs, spike_matrix, time_bins_center = Synchrony.compute_pairwise_corr(spiketimes_set,
                                                                                              binsz=binsz,
                                                                                              window=window)
        # Create Histogram
        bins = np.arange(-1, 1 + binsz, binsz)
        n, bins, patches = ax.hist(correlations, bins=bins, color="black", alpha=0.7,
                                   edgecolor="white", linewidth=0.5)
        # statistics
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)

        # Style plot
        ax.set_xlabel("Spike count correlations (r)")
        ax.set_ylabel("Number of pairs")

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Distribution of Spike Count Correlations " + str(n_neurons) + " neurons" + nucname)

        # Add vertical line at r=0
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)

        # Set limits
        ax.set_xlim(-1, 1)
        ax.grid(True, alpha=0.3)

        # Add Statistics text
        ax.text(0.05, 0.95, f"Mean r = {mean_corr:.3f}\nStd = {std_corr:.3f}",
                transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        return ax


    @classmethod
    def plot_spike_corr(cls, spiketimes_set, binsz=0.01, window=(0, 10), neurons="all", nucleus=None, show=True):
        """
        Displays the histogram of spike count correlations of the given spike times (seconds)
        and returns the plot figure (to save if necessary).

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: integer or float; `0.01` [default]
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

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
        cls.binsz = binsz
        cls.window = window

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = cls.plot_spike_corr_in_ax(ax, spiketimes_set, binsz=binsz, window=window,
                                       nucleus=nucleus)

        if show:
            plt.show()

        return fig, ax