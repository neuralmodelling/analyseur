# ~/analyseur/cbgt/stat/psth.py
#
# Documentation by Lungsi 28 Oct 2025
#

import numpy as np

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.parameters import SpikeAnalysisParams

spikeanal = SpikeAnalysisParams()


class PSTH(object):
    """
    Computes for the Peri-Stimulus Time Histogram (PSTH using
    `numpy.histogram <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`_)
    of spiking times from all neurons.

    +----------------------------+----------------------------------------------------------------------------------------------------------+
    | Methods                    | Argument                                                                                                 |
    +============================+==========================================================================================================+
    | :py:meth:`.compute`        | - `spiketimes_superset`: see :class:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`      |
    |                            | - `window` [OPTIONAL]: Tuple `(0, 10) seconds` [default]                                                 |
    |                            | - `binsz` [OPTIONAL]: 0.01 (= 100 per bin) [default]                                                     |
    |                            | - `neurons` [OPTIONAL]: "all" [default] or list: range(a, b) or [1, 4, 5, 9]                             |
    +----------------------------+----------------------------------------------------------------------------------------------------------+

    =========
    Use Cases
    =========

    ------------------
    1. Pre-requisites
    ------------------

    1.1. Import Modules
    ````````````````````
    ::

        from analyseur.cbgt.loader import LoadSpikeTimes
        from analyseur.cbgt.stats.psth import PSTH

    1.2. Load file and get spike times
    ```````````````````````````````````
    ::

        loadST = LoadSpikeTimes("spikes_GPi.csv")
        spiketimes_superset = loadST.get_spiketimes_superset()

    ---------
    2. Cases
    ---------

    2.1. Compute PSTH (for all neurons)
    ````````````````````````````````````
    ::

        B = PSTH.compute(spiketimes_superset)

    2.2. Compute PSTH for chosen neurons with desired bin size
    ``````````````````````````````````````````````````````````
    ::

        B = PSTH.compute(spiketimes_superset, neurons=range(30, 120), binsz=0.1)

    PSTH for neurons 30 to 120 with the bin size of 0.1 seconds.

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    @classmethod
    def _compute_true_avg_firing_rate(cls, window, spiketimes_superset):
        """
        Computes the average of each neuron's firing rate over the entire period

        :param window:
        :param spiketimes_superset:
        :return: dictionary with keys: firing_rates, mean_firing_rate, std_firing_rate
        """
        firing_rates = []
        total_duration = window[1] - window[0]

        for indiv_spiketimes in spiketimes_superset:
            spiketimes = np.array(indiv_spiketimes)
            spikes_in_window = spiketimes[(spiketimes >= window[0]) & (spiketimes <= window[1])]
            indiv_rate = len(spikes_in_window) / total_duration  # kHz
            firing_rates.append(indiv_rate)

        return {
            "firing_rates": np.array(firing_rates),
            "mean_firing_rate": np.mean(firing_rates),
            "std_firing_rate": np.std(firing_rates),
        }

    @classmethod
    def _compute_pop_firing_rate(cls, n_neurons, binsz, pop_counts):
        """Computes the firing rate of the whole population."""
        return pop_counts / (n_neurons * binsz)  # in Hz

    @classmethod
    def compute(cls, spiketimes_superset, neurons=None, binsz=None, window=None):
        """
        Computation for Peri-Stimulus Time Histogram (PSTH) of all individual neurons.

        :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :param window: Tuple in the form `(start_time, end_time)`; (0, 10) [default]
        :param binsz: 0.01 (= 100 per bin) [default]
        :return: a tuple in the following order
        - array of the values (counts) of the histogram
        - array of bin centers
        - population firing rate
        - dictionary of firing rates
            - "firing_rates": array of firing rates for each neuron
            - "mean_firing_rate": mean of the array of firing rates
            - "std_firing_rate": standard deviation of the array of firing rates

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = spikeanal.window

        if binsz is None:
            binsz = spikeanal.binsz_100perbin

        if neurons is None:
            neurons = "all"

        [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)

        allspikes = np.concatenate(desired_spiketimes_subset)
        allspikes_in_window = allspikes[(allspikes >= window[0]) &
                                        (allspikes <= window[1])]  # Memory efficient

        bins = np.arange(window[0], window[1] + binsz, binsz)
        # bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute
        counts, bin_edges = np.histogram(allspikes_in_window, bins=bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # should be = (bins[:-1] + bins[1:]) / 2

        # firerate = self._compute_firing_rate_in_window(window, allspikes_in_window)
        popfirerates = cls._compute_pop_firing_rate(len(desired_spiketimes_subset), binsz, counts)
        true_avg_rate = cls._compute_true_avg_firing_rate(window, desired_spiketimes_subset)

        return counts, bin_centers, popfirerates, true_avg_rate #, allspikes_in_window