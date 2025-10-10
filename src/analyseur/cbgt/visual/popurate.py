# ~/analyseur/cbgt/visual/popurate.py
#
# Documentation by Lungsi 9 Oct 2025
#
# This contains function for Population Spike Rate Histogram (PSRH)
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ..loader import get_desired_spiketrains

class PSRH(object):
    """
    The Population Spike Rate Histogram (PSRH) Class is instantiated by passing

    :param spiketrains: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    
    +--------------------------------+--------------------------------------------------------------------+
    | Methods                        | Return                                                             |
    +================================+====================================================================+
    | :py:meth:`.plot`               | - `matplotlib.pyplot.hist` object                                  |
    +--------------------------------+--------------------------------------------------------------------+
    | :py:meth:`.analytics` | - dictionary of population dynamics from the population rates   |
    +--------------------------------+--------------------------------------------------------------------+

    * The instance must first invoke :py:meth:`.plot` before calling either :py:meth:`.analytics_temporal` or :py:meth:`.analytics_rate`
    * `psrh` gives a collective dynamics of the population ensemble
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketrains()

      from analyseur.cbgt.visual.popurate import PSRH

      my_psrh = PSRH(spike_trains)

    2. Population Spike Rate Histogram for the whole simulation window

    ::

      my_psrh.plot()

    3. PSRH for desired window and bin size

    ::

      my_psrh.plot(window=(0,50), binsz=1)
      my_psrh.plot(window=(0,50), binsz=0.05)

    4. Get the analytics

    ::

      my_psrh.analytics()

    """

    def __init__(self, spiketrains):
        self.spiketrains = spiketrains

    def _compute_psrh(self, desired_spiketrains, binsz=50, window=(0, 10000)):
        bins = np.arange(window[0], window[1] + binsz, binsz)

        # Population Rate Histogram
        pop_rate = np.zeros(len(bins) - 1)
        # First get the counts
        for a_spiketrain in desired_spiketrains:
            counts, _ = np.histogram(a_spiketrain, bins=bins)
            pop_rate += counts
        # Then compute the rates
        pop_rate = pop_rate / (binsz * len(desired_spiketrains))  # kHz: spikes per milliseconds neurons

        return pop_rate, bins

    def plot(self, binsz=50, window=(0, 10000), nucleus=None):
        """
        Displays the Population Spike Rate Histogram (PSRH) of the given spike times
        and returns the plot figure (to save if necessary).
        
        :param binsz: integer or float; defines the number of equal-width bins in the range [default: 50]
        :param window: 2-tuple; defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: string; [OPTIONAL] None or name of the nucleus
        :return: object `matplotlib.pyplot.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_
        
        * `window` controls the binning range as well as the spike counting window
        * CBGT simulation was done in milliseconds so window `(0, 10000)` signifies time 0 ms to 10,000 ms (or 10 s)
        
        """
        # Set binsz and window as the instance attributes
        self.binsz = binsz
        self.window = window

        # Get and set desired_spiketrains as instance attribute
        [self.desired_spiketrains, _] = get_desired_spiketrains(self.spiketrains)
        # NOTE: desired_spiketrains as nested list and not numpy array because
        # each neuron may have variable length of spike times
        self.n_neurons = len(self.desired_spiketrains)

        # Compute PSRH and set the results as instance attributes
        [self.pop_rate, self.bins] = \
            self._compute_psrh(self.desired_spiketrains, binsz=binsz, window=window)

        t_axis = self.bins[:-1] + binsz / 2

        # Plot
        plt.plot(t_axis, self.pop_rate, linewidth=2)
        plt.fill_between(t_axis, self.pop_rate, alpha=0.3)
        plt.grid(True, alpha=0.3)

        plt.ylabel("Pop. firing rate (kHz)")
        plt.xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        plt.title("Population Spiking Rate Histogram of " + str(self.n_neurons) + " neurons" + nucname)

        plt.show()

        return plt

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

