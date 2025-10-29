# ~/analyseur/cbgt/stat/psth.py
#
# Documentation by Lungsi 28 Oct 2025
#

import numpy as np

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.parameters import SpikeAnalysisParams
from analyseur.cbgt.analytics.ratesparsity import Sparsity

spikeanal = SpikeAnalysisParams()


class PSTH(object):
    """
    Computes for the Peri-Stimulus Time Histogram (PSTH using
    `numpy.histogram <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`_)
    of spiking times from all neurons.
    This gives an overall temporal pattern of population activity with a picture in both temporal and rate.

    +--------------------------------+----------------------------------------------------------------------------------------------------------+
    | Methods                        | Argument                                                                                                 |
    +================================+==========================================================================================================+
    | :py:meth:`.compute_poolPSTH`   | - `spiketimes_superset`: see :class:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`      |
    |                                | - `window` [OPTIONAL]: Tuple `(0, 10) seconds` [default]                                                 |
    |                                | - `binsz` [OPTIONAL]: 0.01 (= 100 per bin) [default]                                                     |
    |                                | - `neurons` [OPTIONAL]: "all" [default] or list: range(a, b) or [1, 4, 5, 9]                             |
    +--------------------------------+----------------------------------------------------------------------------------------------------------+
    | :py:meth:`.compute_avgPSTH`    | - `spiketimes_superset`: see :class:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`      |
    |                                | - `window` [OPTIONAL]: Tuple `(0, 10) seconds` [default]                                                 |
    |                                | - `binsz` [OPTIONAL]: 0.01 (= 100 per bin) [default]                                                     |
    |                                | - `neurons` [OPTIONAL]: "all" [default] or list: range(a, b) or [1, 4, 5, 9]                             |
    +--------------------------------+----------------------------------------------------------------------------------------------------------+
    | :py:meth:`.analyze_temporal`   | - `desired_spiketimes_subset`: a nested list of spike times used for computing the PSTH                  |
    |                                | - `popfirerates`: array of population firing rate (at each bin)                                          |
    |                                | - `bin_centers`: array of bin centers                                                                    |
    |                                | - `binsz`: bin size used for the PSTH                                                                    |
    |                                | - `stimulus_onset` [OPTIONAL]: 0 [default]                                                               |
    +--------------------------------+----------------------------------------------------------------------------------------------------------+
    | :py:meth:`.analyze_rate`       | - `desired_spiketimes_subset`: a nested list of spike times used for computing the PSTH                  |
    |                                | - `true_avg_rate`: dictionary of firing rates; see return value of :py:meth:`.compute`                   |
    |                                | - `popfirerates`: array of population firing rate (at each bin)                                          |
    |                                | - `window`: window used for the PSTH                                                                     |
    |                                | - `stimulus_onset` [OPTIONAL]: 0 [default]                                                               |
    +--------------------------------+----------------------------------------------------------------------------------------------------------+
    | :py:meth:`.analyze_energy`     | - `true_avg_rate`: dictionary of firing rates; see return value of :py:meth:`.compute_poolPSTH`          |
    +--------------------------------+----------------------------------------------------------------------------------------------------------+

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

        B = PSTH.compute_poolPSTH(spiketimes_superset)
        C = PSTH.compute_avgPSTH(spiketimes_superset)

    2.2. Compute PSTH for chosen neurons with desired bin size
    ``````````````````````````````````````````````````````````
    ::

        B = PSTH.compute_poolPSTH(spiketimes_superset, neurons=range(30, 120), binsz=0.1)

    PSTH for neurons 30 to 120 with the bin size of 0.1 seconds.

    2.3. Analytics: Get temporal features from the PSTH
    ```````````````````````````````````````````````````
    ::

        [counts, bin_info, popfirerates,
        true_avg_rate, desired_spiketimes_subset] = PSTH.compute_poolPSTH(spiketimes_superset)

        temporal_features = PSTH.analyze_temporal(desired_spiketimes_subset,
                                                  popfirerates=popfirerates,
                                                  bin_centers=bin_info["bin_centers"],
                                                  binsz=bin_info["binsz"],)

    2.4. Analytics: Get rate-based features from the PSTH
    `````````````````````````````````````````````````````
    ::

        rate_features = PSTH.analyze_rate(desired_spiketimes_subset,
                                          true_avg_rate=true_avg_rate,
                                          popfirerates=popfirerates,
                                          window=bin_info["window"],)

    2.5. Analytics: Get energetics from the PSTH
    ````````````````````````````````````````````
    ::

        energetics = PSTH.analyze_energy(true_avg_rate)

    ======================================
    Possible Insights From Population PSTH
    ======================================

    +-------------------------+-------------------------------------------------------------+
    |   Insight               | Meaning                                                     |
    +=========================+=============================================================+
    | `"collective_timing"`   | when does the population (as whole) respond?                |
    +-------------------------+-------------------------------------------------------------+
    | `"pop_reliability"`     | how robust is the population spiking?                       |
    +-------------------------+-------------------------------------------------------------+
    | `"info_redundancy"`     | how similarly/diversely do neurons (in population) respond? |
    +-------------------------+-------------------------------------------------------------+
    | `"coding_strategy"`     | sparse or dense population spiking?                         |
    +-------------------------+-------------------------------------------------------------+
    | `"population_dynamics"` | how activity propagated through the network?                |
    +-------------------------+-------------------------------------------------------------+

    - sparse coding is a measure of whether a few neurons do most of the firing
    - dense coding is a measure of whether the neurons firing in the population is evenly distributed

    ======================================================================
    Mean of Individual Firing Rate vs Mean of Time-Varying Population Rate
    ======================================================================

    True Average Firing Rate (or Mean of Individual Firing Rate) is

    - count each neuron's spikes over entire time window
    - average across neurons (Hz)

    Mean of Time-Varying Population Rate is

    - count population spikes per time (bin)
    - take each count per population size per bin (i.e instantaneous rate)
    - average cross time bins (Hz)

    +-------------------------------------------+----------------------------------------------------+---------------------------------------------------+
    |  True Average Firing Rate                 |  Population Rate (Time-Varying)                    | Mean of Population Rate                           |
    +===========================================+====================================================+===================================================+
    | - actual average firing rate of neurons   | - analyze temporal dynamics of population response | - average level of time-varying population signal |
    | - calculate population statistics         |     - response latency, peak time, duration        | - comparing temporal patterns (normalization)     |
    | - compare firing rates across populations | - study population activity evolving over time     | - is NOT neuron's average firing rate             |
    +-------------------------------------------+----------------------------------------------------+---------------------------------------------------+

    * When firing rate is constant over time
        - Mean of Individual Firing Rate = Mean of Time-Varying Population Rate

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
    def _compute_avg_firing_rate_from_PSTH(cls, window, n_neurons, pop_counts):
        """
        Computes the average of each neuron's firing rate from PSTH data
        """
        total_duration = window[1] - window[0]
        return np.sum(pop_counts) / (n_neurons * total_duration) # Hz

    @classmethod
    def _compute_pop_firing_rate(cls, n_neurons, binsz, pop_counts):
        """
        Computes the  firing rate of the whole population at each bin.
        Therefore, this is the TIME-VARYING population rate (since its at each bin).
        Hence, mean of the population rates across the bins is NOT average firing rate.
        It is the AVERAGE of the TIME-VARYING rates across all bins.
        """
        return pop_counts / (n_neurons * binsz)  # in Hz

    @classmethod
    def compute_poolPSTH(cls, spiketimes_superset, neurons=None, binsz=None, window=None):
        """
        Computation of Pooled Population Peri-Stimulus Time Histogram (PSTH) of all individual neurons.

        :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :param window: Tuple in the form `(start_time, end_time)`; e.g (0, 10)
        :param binsz: e.g 0.01 (= 100 per bin)
        :param neurons: "all" or list: range(a, b) or [1, 4, 5, 9]
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

        # Poole spikes from ALL neurons
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

    @classmethod
    def compute_avgPSTH(cls, spiketimes_superset, neurons=None, binsz=None, window=None):
        """
        Computation of Average of Individual Peri-Stimulus Time Histogram (Population PSTH).

        :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :param window: Tuple in the form `(start_time, end_time)`; e.g (0, 10)
        :param binsz: e.g 0.01 (= 100 per bin)
        :param neurons: "all" or list: range(a, b) or [1, 4, 5, 9]
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

        bins = np.arange(window[0], window[1] + binsz, binsz)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # PSTH for EACH neuron
        n_neurons = len(desired_spiketimes_subset)
        neuron_psths = np.zeros(n_neurons)
        for i, indiv_spiketimes in enumerate(desired_spiketimes_subset):
            counts, _ = np.histogram(indiv_spiketimes, bins=bins)
            rates = counts / binsz # Hz
            neuron_psths[i] = rates
        # Average the PSTH across neurons
        average_psth = np.mean(neuron_psths, axis=0)
        std_err_psth = np.std(neuron_psths, axis=0) / np.sqrt(n_neurons)

        bin_info = {
            "window": window,
            "binsz": binsz,
            "bin_centers": (bins[:-1] + bins[1:]) / 2
        }

        # firerate = self._compute_firing_rate_in_window(window, allspikes_in_window)
        popfirerates = cls._compute_pop_firing_rate(len(desired_spiketimes_subset), binsz, counts)
        true_avg_rate = cls._compute_true_avg_firing_rate(window, desired_spiketimes_subset)

        return average_psth, std_err_psth, bin_info, popfirerates, true_avg_rate, desired_spiketimes_subset

    @staticmethod
    def analyze_temporal(desired_spiketimes_subset, popfirerates=[], bin_centers=[], binsz=None, stimulus_onset=0):
        """
        Extracts temporal features from the PSTH counts

        :param desired_spiketimes_subset: a nested list of spike times used for computing the PSTH
        :param popfirerates: array of population firing rate (at each bin)
        :param bin_centers: array of bin centers
        :param binsz: bin size used for the PSTH
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

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

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
    def analyze_rate(desired_spiketimes_subset, true_avg_rate=[], popfirerates=[], window=(), stimulus_onset=0):
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
        | `"rate_heterogeneity"`   | measure of response heterogeneity                          |
        +--------------------------+------------------------------------------------------------+
        | `"response_reliability"` | measure of response consistency                            |
        +--------------------------+------------------------------------------------------------+
        | `"active_fraction"`      | active neurons out of total neurons                        |
        +--------------------------+------------------------------------------------------------+
        | `"population_sparsity"`  | measure of concentration of firing rates across population |
        +--------------------------+------------------------------------------------------------+

        - sparsity index (`"population_sparsity"`) is a measure of how concentrated or distributed the firing is across the neurons in the population

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        mean_firing_rate = true_avg_rate["mean_firing_rate"]
        std_firing_rate = true_avg_rate["std_firing_rate"]

        n_neurons = len(desired_spiketimes_subset)
        total_duration = window[1] - window[0]

        # For each neuron in the population
        firing_rates = []
        baseline_rates = []
        response_rates = []

        for indiv_spiketimes in desired_spiketimes_subset:
            total_spikes = len(indiv_spiketimes)
            spiketimes = np.array(indiv_spiketimes)

            # Overall firing rate
            firing_rate = total_spikes / total_duration if total_duration > 0 else 0
            firing_rates.append(firing_rate)

            # Rates: Baseline vs Response
            baseline_spikes = spiketimes[(spiketimes >= window[0]) & (spiketimes < stimulus_onset)]
            response_spikes = spiketimes[spiketimes >= stimulus_onset]

            baseline_duration = stimulus_onset - window[0]
            response_duration = window[1] - stimulus_onset

            baseline_rates.append(len(baseline_spikes) / baseline_duration
                                  if baseline_duration > 0 else 0)
            response_rates.append(len(response_spikes) / response_duration
                                  if response_duration > 0 else 0)

        firing_rates = np.array(firing_rates)
        baseline_rates = np.array(baseline_rates)
        response_rates = np.array(response_rates)

        # Rate change metrics
        rate_changes = response_rates - baseline_rates
        fold_changes = response_rates / (baseline_rates + 1e-8)

        # Population coding properties
        active_neurons = np.sum(response_rates > baseline_rates + np.std(baseline_rates))
        sparsity_analytics = Sparsity.analyze(true_avg_rate["firing_rates"],
                                              baseline_rates, response_rates)

        return {
            "mean_firing_rate": mean_firing_rate.item(),
            "std_firing_rate": std_firing_rate.item(),
            "avg_time_vary_rate": np.mean(popfirerates).item(),
            "mean_baseline_rate": np.mean(baseline_rates).item(),
            "mean_response_rate": np.mean(response_rates).item(),
            "mean_rate_change": np.mean(rate_changes).item(),
            "mean_fold_change": np.mean(fold_changes).item(),
            "rate_heterogeneity": (std_firing_rate / mean_firing_rate).item(),
            "response_reliability": (np.sum(rate_changes > 0) / n_neurons).item(),
            "active_fraction": (active_neurons / n_neurons).item(),
            "population_sparsity": sparsity_analytics,
        }

    @staticmethod
    def analyze_energy(true_avg_rate):
        """
        Extracts energy features from the PSTH counts, i.e, analytics for metabolic efficiency of rate distribution.

        :param true_avg_rate:
        :return: dictionary

        +-------------------------------+------------------------------------------------------------+
        | Dictionary key                | Meaning                                                    |
        +===============================+============================================================+
        | `"total_population_activity"` | -to do-                                                    |
        +-------------------------------+------------------------------------------------------------+
        | `"max_entropy"`               | -to do-                                                    |
        +-------------------------------+------------------------------------------------------------+
        | `"entropy"`                   | -to do                                                     |
        +-------------------------------+------------------------------------------------------------+
        | `"efficiency"`                | how efficiently are the rates distributed                  |
        +-------------------------------+------------------------------------------------------------+
        | `"energy_per_bit"`            | metabolic cost per information bit                         |
        +-------------------------------+------------------------------------------------------------+
        | `"dynamic_range"`             | -to do-                                                    |
        +-------------------------------+------------------------------------------------------------+

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

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