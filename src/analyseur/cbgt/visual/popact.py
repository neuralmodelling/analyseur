# ~/analyseur/cbgt/visual/popact.py
#
# Documentation by Lungsi 10 Oct 2025
#
# This contains function for Population Activity Heatmap
#

import numpy as np
import matplotlib.pyplot as plt

from ..loader import get_desired_spiketrains

class ActivityHeatmap(object):
    def __init__(self, spiketrains):
        self.spiketrains = spiketrains

    def _compute_activity(self, desired_spiketrains, binsz=50, window=(0, 10000)):
        bins = np.arange(window[0], window[1] + binsz, binsz)

        # Activity Matrix
        activity = np.zeros((len(desired_spiketrains), len(bins) - 1))
        for i, spikes in enumerate(desired_spiketrains):
            counts, _ = np.histogram(spikes, bins=bins)
            activity[i] = counts
        activity = activity[::-1, :]  # reverse it so that neuron 0 is at the bottom

        return activity, bins

    def plot(self, binsz=50, window=(0, 10000), nucleus=None):
        # Set binsz and window as the instance attributes
        self.binsz = binsz
        self.window = window

        # Get and set desired_spiketrains as instance attribute
        [self.desired_spiketrains, _] = get_desired_spiketrains(self.spiketrains)
        # NOTE: desired_spiketrains as nested list and not numpy array because
        # each neuron may have variable length of spike times
        self.n_neurons = len(self.desired_spiketrains)

        # Compute activities in activity matrix and set the results as instance attributes
        [self.activity_matrix, self.bins] = \
            self._compute_psrh(self.desired_spiketrains, binsz=binsz, window=window)

        t_axis = self.bins[:-1] + binsz / 2

        # Plot
        plt.imshow(self.activity_matrix, aspect="auto", cmap="hot",
                   # extent=[window[0], window[1], n_neurons, 0] # if neuron 0 is at the top by default
                   extent=[window[0], window[1], 0, self.n_neurons])
        plt.colorbar(label="Spike Count per Bin")

        plt.ylabel("neurons")
        plt.xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        plt.title("Population Activity Heatmap of " + str(self.n_neurons) + " neurons" + nucname)

        plt.show()

        return plt