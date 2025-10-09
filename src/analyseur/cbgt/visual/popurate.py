# ~/analyseur/cbgt/visual/popurate.py
#
# Documentation by Lungsi 9 Oct 2025
#
# This contains function for Population Spike Rate Histogram (PSRH)
#

import numpy as np
import matplotlib.pyplot as plt

from ..loader import get_desired_spiketrains

class PSRH(object):

    def __init__(self, spiketrains):
        self.spiketrains = spiketrains

    def _compute_psrh(self, desired_spiketrains, binsz=50, window=(0, 10000)):
        allspikes = np.concatenate(desired_spiketrains)
        allspikes_in_window = allspikes[(allspikes >= window[0]) &
                                        (allspikes <= window[1])]  # Memory efficient

        bins = np.arange(window[0], window[1] + binsz, binsz)

        # Population Rate Histogram
        pop_rate = np.zeros(len(bins) - 1)
        for spikes in allspikes_in_window:
            counts, _ = np.histogram(spikes, bins=bins)
            pop_rate += counts
        # Histogram to firing rate
        pop_rate = 1000 * (pop_rate / (binsz * len(allspikes_in_window)))  # in seconds [default: spikes per milliseconds neurons]

        return pop_rate, bins, allspikes_in_window

    def plot(self, binsz=50, window=(0, 10000), nucleus=None):
        # Set binsz and window as the instance attributes
        self.binsz = binsz
        self.window = window

        # Get and set desired_spiketrains as instance attribute
        [self.desired_spiketrains, _] = get_desired_spiketrains(self.spiketrains)
        # NOTE: desired_spiketrains as nested list and not numpy array because
        # each neuron may have variable length of spike times

        # Compute PSRH and set the results as instance attributes
        [self.pop_rate, self.bins, self.allspikes_in_window] = \
            self._compute_psrh(self.desired_spiketrains, binsz=binsz, window=window)

        t_axis = self.bins[:-1] + binsz / 2

        # Plot
        plt.plot(t_axis, self.pop_rate, linewidth=2)
        plt.fill_between(t_axis, self.pop_rate, alpha=0.3)
        plt.grid(True, alpha=0.3)

        plt.ylabel("Pop. firing rate (Hz)")
        plt.xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        plt.title("Population Spiking Rate Histogram of " +
                  str(len(self.allspikes_in_window)) + " neurons" + nucname)

        plt.show()

        return plt

