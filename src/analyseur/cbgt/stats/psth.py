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

    2.3. Analytics: Get temporal features from the PSTH
    ```````````````````````````````````````````````````
    ::

        [counts, bin_info, popfirerates, true_avg_rate, desired_spiketimes_subset] \
              = PSTH.compute(spiketimes_superset)

        temporal_features = PSTH.analytics_temporal(desired_spiketimes_subset,
                                                    popfirerates=popfirerates,
                                                    bin_centers=bin_info["bin_centers"],
                                                    binsz=bin_info["binsz"],)

    2.4. Analytics: Get rate-based features from the PSTH
    `````````````````````````````````````````````````````
    ::

        rate_features = PSTH.analytics_rate(desired_spiketimes_subset,
                                            true_avg_rate=true_avg_rate,
                                            popfirerates=popfirerates,
                                            window=bin_info["window"],)

    2.5. Analytics: Get energetics from the PSTH
    ````````````````````````````````````````````
    ::

        energetics = PSTH.analytics_energy(true_avg_rate)

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
        """
        Computes the firing rate of the whole population.
        """
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
        - dictionary of bin information
            - "window": window used for computing the PSTH
            - "binsz": binsz used for computing the PSTH
            - "bin_centers": array of bin centers
        - array of population firing rate (at each bin)
        - dictionary of firing rates
            - "firing_rates": array of firing rates for each neuron
            - "mean_firing_rate": mean of the array of firing rates
            - "std_firing_rate": standard deviation of the array of firing rates
        - a nested list of spike times used for computing the PSTH

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

        bin_info = {
            "window": window,
            "binsz": binsz,
            "bin_centers": (bin_edges[:-1] + bin_edges[1:]) / 2  # should be = (bins[:-1] + bins[1:]) / 2
        }

        # firerate = self._compute_firing_rate_in_window(window, allspikes_in_window)
        popfirerates = cls._compute_pop_firing_rate(len(desired_spiketimes_subset), binsz, counts)
        true_avg_rate = cls._compute_true_avg_firing_rate(window, desired_spiketimes_subset)

        return counts, bin_info, popfirerates, true_avg_rate, desired_spiketimes_subset #, allspikes_in_window

    @staticmethod
    def analytics_temporal(desired_spiketimes_subset, popfirerates=[], bin_centers=[], binsz=None, stimulus_onset=0):
        """
        Extracts temporal features from the PSTH counts

        :param desired_spiketimes_subset: a nested list of spike times used for computing the PSTH
        :param popfirerates: array of population firing rate (at each bin)
        :param bin_centers: array of bin centers
        :param stimulus_onset: [OPTIONAL] default: 0
        :return: dictionary

        +---------------------------+------------------------------+
        | Dictionary key            | Meaning                      |
        +===========================+==============================+
        | `"peak_latency"`          | when peak occurs             |
        +---------------------------+------------------------------+
        | `"response_latencies"`    | when responses begins        |
        +---------------------------+------------------------------+
        | `"response_duration"`     | how long it lasts            |
        +---------------------------+------------------------------+
        | `"response_sequence"`     | time span of first responses |
        +---------------------------+------------------------------+
        | `"response_spread"`       | how synced are the neurons   |
        +---------------------------+------------------------------+
        | `"temporal_coordination"` | coordination index           |
        +---------------------------+------------------------------+

        * the temporal profile (full time course) is the value of the attribute :ivar popfirerates:

        """
        pre_stimulus_rates = popfirerates[bin_centers < stimulus_onset]
        post_stimulus_rates = popfirerates[bin_centers >= stimulus_onset]
        post_stimulus_times = bin_centers[bin_centers >= stimulus_onset]

        # When does the neuron population respond
        if not pre_stimulus_rates.size:
            baseline = 0
            response_threshold = 0
        else:
            baseline = np.mean(pre_stimulus_rates)
            response_threshold = baseline + 2 * np.std(pre_stimulus_rates)

        if len(post_stimulus_rates) > 0:
            supra_thresholds = post_stimulus_rates > response_threshold
            if np.any(supra_thresholds):
                response_latencies = post_stimulus_times[np.where(supra_thresholds)[0][0]] - stimulus_onset
            else:
                response_latencies = None
        else:
            response_latencies = None

        # Population response
        peak_time = bin_centers[np.argmax(popfirerates)]

        if response_latencies is not None:
            response_periods = post_stimulus_rates > baseline
            if np.any(response_periods):
                response_duration = np.sum(response_periods) * binsz
            else:
                response_duration = np.array(0)
        else:
            response_duration = np.array(0)

        # Response dynamics: Temporal pattern (response evolution) and
        # temporal coordination among neurons
        first_spike_times = [np.min(np.array(indiv_spiketimes)[np.array(indiv_spiketimes) >= stimulus_onset])
                             if np.any(np.array(indiv_spiketimes) >= stimulus_onset)
                             else np.inf
                             for indiv_spiketimes in desired_spiketimes_subset]
        first_spike_times = [t for t in first_spike_times if t != np.inf]

        if len(first_spike_times) > 0:
            response_spread = np.std(first_spike_times)  # spread of first spike times
            response_sequence = np.max(first_spike_times) - np.min(first_spike_times)  # total span
        else:
            response_spread = np.array(0)
            response_sequence = np.array(0)

        return {
            "peak_latency": peak_time - stimulus_onset,
            "response_latencies": response_latencies,
            "response_duration": response_duration,
            "response_sequence": response_sequence,
            "response_spread": response_spread,
            "temporal_coordination": 1 / (response_spread + 1e-8),
            # "response_profile": popfirerates,
        }

    @staticmethod
    def analytics_rate(desired_spiketimes_subset, true_avg_rate=[], popfirerates=[], window=(), stimulus_onset=0):
        """
        Extracts rate-based features from the PSTH counts

        :param desired_spiketimes_subset:
        :param true_avg_rate:
        :param popfirerates:
        :param window:
        :param stimulus_onset: [OPTIONAL] default: 0
        :return: dictionary

        +--------------------------+------------------------------------------------------------+
        | Dictionary key           | Meaning                                                    |
        +==========================+============================================================+
        | `"mean_firing_rate"`     | average of each neuron's rate                              |
        +--------------------------+------------------------------------------------------------+
        | `"std_firing_rate"`      | spread of the firing rates                                 |
        +--------------------------+------------------------------------------------------------+
        | `"avg_time_vary_rate"`   | average of the time varying population signal              |
        +--------------------------+------------------------------------------------------------+
        | `"mean_baseline_rate"`   | average spontaneous activity levels                        |
        +--------------------------+------------------------------------------------------------+
        | `"mean_response_rate"`   | average response strengths                                 |
        +--------------------------+------------------------------------------------------------+
        | `"mean_rate_increase"`   | average absolute response magnitudes                       |
        +--------------------------+------------------------------------------------------------+
        | `"mean_fold_change"`     | average relative response magnitudes                       |
        +--------------------------+------------------------------------------------------------+
        | `"active_fraction"`      | active neurons out of total neurons                        |
        +--------------------------+------------------------------------------------------------+
        | `"population_sparsity"`  | measure of concentration of firing rates across population |
        +--------------------------+------------------------------------------------------------+
        | `"rate_heterogeneity"`   | measure of response heterogeneity                          |
        +--------------------------+------------------------------------------------------------+
        | `"response_reliability"` | measure of response consistency                            |
        +--------------------------+------------------------------------------------------------+

        """
        mean_firing_rate = true_avg_rate["mean_firing_rate"]
        std_firing_rate = true_avg_rate["std_firing_rate"]
        n_neurons = len(true_avg_rate["firing_rates"])

        baseline_rates = []
        response_rates = []

        for indiv_spiketimes in desired_spiketimes_subset:
            spiketimes = np.array(indiv_spiketimes)

            # Rates: Baseline vs Response
            baseline_spikes = spiketimes[(spiketimes >= window[0]) & (spiketimes < stimulus_onset)]
            response_spikes = spiketimes[spiketimes >= stimulus_onset]

            baseline_duration = stimulus_onset - window[0]
            response_duration = window[1] - stimulus_onset

            baseline_rates.append(len(baseline_spikes) / baseline_duration
                                  if baseline_duration > 0 else 0)
            response_rates.append(len(response_spikes) / response_duration
                                  if response_duration > 0 else 0)

        baseline_rates = np.array(baseline_rates)
        response_rates = np.array(response_rates)

        # Rate change metrics
        rate_changes = response_rates - baseline_rates
        fold_changes = response_rates / (baseline_rates + 1e-8)

        # Population coding properties
        active_neurons = np.sum(response_rates > baseline_rates + np.std(baseline_rates))
        sparsity_index = 1 - (mean_firing_rate ** 2 / np.mean(true_avg_rate["firing_rates"] ** 2))

        return {
            "mean_firing_rate": mean_firing_rate.item(),
            "std_firing_rate": std_firing_rate.item(),
            "avg_time_vary_rate": np.mean(popfirerates).item(),
            "mean_baseline_rate": np.mean(baseline_rates).item(),
            "mean_response_rate": np.mean(response_rates).item(),
            "mean_rate_change": np.mean(rate_changes).item(),
            "mean_fold_change": np.mean(fold_changes).item(),
            "active_fraction": (active_neurons / n_neurons).item(),
            "population_sparsity": sparsity_index.item(),
            "rate_heterogeneity": (std_firing_rate / mean_firing_rate).item(),
            "response_reliability": (np.sum(rate_changes > 0) / n_neurons).item(),
        }

    @staticmethod
    def analytics_energy(true_avg_rate):
        """
        Extracts energy features from the PSTH counts

        :param true_avg_rate:
        :return: dictionary

        +--------------------------+------------------------------------------------------------+
        | Dictionary key           | Meaning                                                    |
        +==========================+============================================================+
        | `"total_population_activity"`     | average of each neuron's rate                              |
        +--------------------------+------------------------------------------------------------+
        | `"max_entropy"`      | spread of the firing rates                                 |
        +--------------------------+------------------------------------------------------------+
        | `"entropy"`   | average of the time varying population signal              |
        +--------------------------+------------------------------------------------------------+
        | `"efficiency"`   | average spontaneous activity levels                        |
        +--------------------------+------------------------------------------------------------+
        | `"energy_per_bit"`   | average response strengths                                 |
        +--------------------------+------------------------------------------------------------+
        | `"dynamic_range"`   | average absolute response magnitudes                       |
        +--------------------------+------------------------------------------------------------+

        """
        firing_rates = true_avg_rate["firing_rates"]
        total_activity = np.sum(firing_rates)

        max_entropy = np.log(len(firing_rates))

        entropy = -np.sum([(r/total_activity) * np.log(r/total_activity)
                           for r in firing_rates if r > 0])

        dynamic_range = np.max(firing_rates) / (np.max(firing_rates[firing_rates > 0]) + 1e-8)

        return {
            "total_population_activity": total_activity,
            "max_entropy": max_entropy,
            "entropy": entropy,
            "efficiency": entropy / max_entropy,
            "energy_per_bit": total_activity / (entropy + 1e-8),
            "dynamic_range": dynamic_range,
        }