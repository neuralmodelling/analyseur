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
    """
    The ActivityHeatmap Class is instantiated by passing

    :param spiketrains: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    
    +--------------------------------+--------------------------------------------------------------------+
    | Methods                        | Return                                                             |
    +================================+====================================================================+
    | :py:meth:`.plot`               | - `matplotlib.pyplot.imshow` object                                |
    +--------------------------------+--------------------------------------------------------------------+

    * `popactivity` gives a spatio-temporal pattern across neurons
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketrains()

      from analyseur.cbgt.visual.popact import ActivityHeatmap

      my_pact = ActivityHeatmap(spike_trains)

    2. Population Activity Heatmap for the entire simulation window

    ::

      my_pact.plot()

    3. Population Activity Heatmap for desired window and bin size

    ::

      my_pact.plot(spike_trains, window=(0,50), binsz=1)
      my_pact.plot(spike_trains, window=(0,50), binsz=0.05)

    """
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
        """

        Displays the Population Activity Heatmap of the given spike times and returns the plot figure (to save if necessary).

        :param spiketrains: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
        :param binsz: defines the number of equal-width bins in the range [default: 50]
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)
        :return: object `matplotlib.pyplot.imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_
    
        * `window` controls the binning range as well as the spike counting window
        * CBGT simulation was done in milliseconds so window `(0, 10000)` signifies time 0 ms to 10,000 ms (or 10 s)
        * `popactivity` gives a spatio-temporal pattern across neurons

        """
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
            self._compute_activity(self.desired_spiketrains, binsz=binsz, window=window)

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