# ~/analyseur/cbgt/visual/popurate.py
#
# Documentation by Lungsi 9 Oct 2025
#
# This contains function for Population Spike Rate Histogram (PSRH)
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# from ..curate import get_desired_spiketimes_subset
from analyseur.cbgt.curate import get_desired_spiketimes_subset

class PSRH(object):
    """
    The Population Spike Rate Histogram (PSRH) Class is instantiated by passing

    :param spiketimes_superset: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    
    +--------------------------------+--------------------------------------------------------------------+
    | Methods                        | Return                                                             |
    +================================+====================================================================+
    | :py:meth:`.plot`               | - `matplotlib.pyplot.plot` object                                  |
    +--------------------------------+--------------------------------------------------------------------+
    | :py:meth:`.analytics`          | - dictionary of population dynamics from the population rates      |
    +--------------------------------+--------------------------------------------------------------------+
    | :py:meth:`.plot_ratevar`       | - `matplotlib.pyplot.plot` object                                  |
    +--------------------------------+--------------------------------------------------------------------+

    * The instance must first invoke :py:meth:`.plot` before calling :py:meth:`.analytics` or :py:meth:`.plot_ratevar`
    * `psrh` gives a collective dynamics of the population ensemble
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spiketimes_superset = loadST.get_spiketimes_superset()

      from analyseur.cbgt.visual.popurate import PSRH

      my_psrh = PSRH(spiketimes_superset)

    2. Population Spike Rate Histogram for the whole simulation window

    ::

      my_psrh.plot()

    3. PSRH for desired window and bin size

    ::

      my_psrh.plot(window=(0,5), binsz=1)    # time unit in seconds
      my_psrh.plot(window=(0,5), binsz=0.05)

    4. Get the analytics

    ::

      my_psrh.analytics()
      
    5. View Firing Rate Variability
    
    ::

      my_psrh.plot_ratevar()

    """

    def __init__(self, spiketimes_superset):
        self.spiketimes_superset = spiketimes_superset

    def _compute_psrh(self, desired_spiketimes_subset, binsz=0.05, window=(0, 10)):
        bins = np.arange(window[0], window[1] + binsz, binsz)

        # Population Rate Histogram
        pop_rate = np.zeros(len(bins) - 1)
        # Firing rate variability
        firing_rates = np.zeros((len(desired_spiketimes_subset), len(bins)-1))
        # First get the counts
        for i, a_spiketrain in enumerate(desired_spiketimes_subset):
            counts, _ = np.histogram(a_spiketrain, bins=bins)
            pop_rate += counts
            firing_rates[i] = counts / binsz # Per bin compute firing rate per neuron
        # Then compute the rates
        pop_rate = pop_rate / (binsz * len(desired_spiketimes_subset))  # kHz: spikes per milliseconds neurons

        return pop_rate, firing_rates, bins

    def plot(self, binsz=0.05, window=(0, 10), nucleus=None, show=True):
        """
        Displays the Population Spike Rate Histogram (PSRH) of the given spike times (seconds)
        and returns the plot figure (to save if necessary).
        
        :param binsz: integer or float; defines the number of equal-width bins in the range [default: 50]
        :param window: 2-tuple; defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: string; [OPTIONAL] None or name of the nucleus
        :param show: boolean [default: True]
        :return: object `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_
        
        * `window` controls the binning range as well as the spike counting window
        * CBGT simulation was done in seconds so window `(0, 10)` signifies time 0 s to 10 s
        
        """
        # Set binsz and window as the instance attributes
        self.binsz = binsz
        self.window = window
        self.nucleus = nucleus

        # Get and set desired_spiketimes_subset as instance attribute
        [self.desired_spiketimes_subset, _] = get_desired_spiketimes_subset(self.spiketimes_superset, neurons="all")
        # NOTE: desired_spiketimes_subset as nested list and not numpy array because
        # each neuron may have variable length of spike times
        self.n_neurons = len(self.desired_spiketimes_subset)

        # Compute PSRH and set the results as instance attributes
        [self.pop_rate, self.firing_rates, self.bins] = \
            self._compute_psrh(self.desired_spiketimes_subset, binsz=binsz, window=window)

        t_axis = self.bins[:-1] + binsz / 2

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_axis, self.pop_rate, linewidth=2)
        ax.fill_between(t_axis, self.pop_rate, alpha=0.3)
        ax.grid(True, alpha=0.3)

        ax.set_ylabel("Pop. firing rate (Hz)")
        ax.set_xlabel("Time (s)")

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Population Spiking Rate Histogram of " + str(self.n_neurons) + " neurons" + nucname)

        if show:
            plt.show()

        return fig, ax

    def analytics(self, stimulus_onset=0):
        """
        Extracts population dynamics from the population rates
        :param stimulus_onset: [OPTIONAL] default: 0
        :return: dictionary
        
        +---------------------------+------------------------------+
        | Dictionary key            | Meaning                      |
        +===========================+==============================+
        | `"mean_rate"`          | average firing rate across the window            |
        +---------------------------+------------------------------+
        | `"peak_rate"`    | maximum population activity level        |
        +---------------------------+------------------------------+
        | `"peak_time"`     | time taken to maximum response from reference point (stimulus onset) |
        +---------------------------+------------------------------+
        | `"response_latency"`     | time until response exceeds threshold |
        +---------------------------+------------------------------+
        | `"fano_factor"`       | measure of dispersion   |
        +---------------------------+------------------------------+
        | `"cv"` | coefficient of variation           |
        +---------------------------+------------------------------+
        | `"skewness"`       | asymmetry of rate distribution   |
        +---------------------------+------------------------------+
        | `"kurtosis"` | "tailedness" of distribution (peak or flat)   |
        +---------------------------+------------------------------+

        """
        mean_rate = np.mean(self.pop_rate)

        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        # Response timing w.r.t stimulus
        post_stimulus = bin_centers >= stimulus_onset
        if np.any(post_stimulus):
            stim_rates = self.pop_rate[post_stimulus]
            response_latency = bin_centers[post_stimulus][np.argmax(stim_rates > mean_rate)]
        else:
            response_latency = None

        return {
            "mean_rate": mean_rate.item(),
            "peak_rate": (np.max(self.pop_rate)).item(),
            "peak_time": (bin_centers[np.argmax(self.pop_rate)]).item(),
            "response_latency": response_latency.item(),
            "fano_factor": (np.var(self.pop_rate) / mean_rate).item(),
            "cv": (np.std(self.pop_rate) / mean_rate).item(),
            "skewness": (stats.skew(self.pop_rate)).item(),
            "kurtosis": (stats.kurtosis(self.pop_rate)).item(),
        }

    def plot_ratevar(self):
        """
        Displays the Population Spike Rate Variability in terms of:
        Mean ± STD Variability and Coefficient of Variation.

        :return: object `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_
        
        """
        t_axis = self.bins[:-1] + self.binsz / 2
        linewidth = 2

        mean_fr = np.mean(self.firing_rates, axis=0)
        std_fr = np.std(self.firing_rates, axis=0)

        cv = std_fr / (mean_fr + 1e-8)

        # Plot
        fig, axes = plt.subplots(1,2)

        axes[0].plot(t_axis, mean_fr, label="Mean", linewidth=linewidth)
        axes[0].fill_between(t_axis, mean_fr - std_fr, mean_fr + std_fr,
                         alpha=0.3, label="±1 STD")
        axes[0].grid(True, alpha=0.3)

        axes[0].set_ylabel("Firing Rate (Hz)")
        axes[0].set_xlabel("Time (s)")

        nucname = "" if self.nucleus is None else " in " + self.nucleus
        axes[0].set_title("Population Firing Rate (Mean ± STD) Variability of " + str(self.n_neurons) + " neurons" + nucname)

        axes[1].plot(t_axis, cv, linewidth=linewidth)
        axes[1].grid(True, alpha=0.3)

        axes[1].set_ylabel("Coefficient of Variation")
        axes[1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

        return fig, axes

