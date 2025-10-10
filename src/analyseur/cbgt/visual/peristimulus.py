# ~/analyseur/cbgt/visual/peristimulus.py
#
# Documentation by Lungsi 8 Oct 2025
#
# This contains function for Peri-Stimulus Time Histogram (PSTH)
#

import numpy as np
import matplotlib.pyplot as plt

from ..loader import get_desired_spiketrains

class PSTH(object):
    """
    The Peri-Stimulus Time Histogram (PSTH) Class is instantiated by passing

    :param spiketrains: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    
    +--------------------------------+--------------------------------------------------------------------+
    | Methods                        | Return                                                             |
    +================================+====================================================================+
    | :py:meth:`.plot`               | - `matplotlib.pyplot.hist` object                                  |
    +--------------------------------+--------------------------------------------------------------------+
    | :py:meth:`.analytics_temporal` | - dictionary of temporal features extracted from the PSTH counts   |
    +--------------------------------+--------------------------------------------------------------------+
    | :py:meth:`.analytics_rate`     | - dictionary of rate-based features extracted from the PSTH counts |
    +--------------------------------+--------------------------------------------------------------------+

    * The instance must first invoke :py:meth:`.plot` before calling either :py:meth:`.analytics_temporal` or :py:meth:`.analytics_rate`
    * `psth` gives an overall temporal pattern of population activity with a picture in both temporal and rate
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketrains()

      from analyseur.cbgt.visual.peristimulus import PSTH

      my_psth = PSTH(spike_trains)

    2. Peri-Stimulus Time Histogram for the whole simulation window

    ::

      my_psth.plot()

    3. PSTH for desired window and bin size

    ::

      my_psth.plot(window=(0,50), binsz=1)
      my_psth.plot(window=(0,50), binsz=0.05)

    4. Get the analytics for respective (immediately preceeding plotted psth)

    ::

      my_psth.analytics_temporal()
      my_psth.analytics_rate()

    """

    def __init__(self, spiketrains):
        self.spiketrains = spiketrains

    def _compute_true_avg_firing_rate(self, window, desired_spiketrains):
        """
        Computes the average of each neuron's firing rate over the entire period

        :param window:
        :param desired_spiketrains:
        :return: dictionary with keys: firing_rates, mean_firing_rate, std_firing_rate
        """
        firing_rates = []
        total_duration = window[1] - window[0]

        for indiv_spiketimes in desired_spiketrains:
            spiketimes = np.array(indiv_spiketimes)
            spikes_in_window = spiketimes[(spiketimes >= window[0]) & (spiketimes <= window[1])]
            indiv_rate = len(spikes_in_window) / total_duration # kHz
            firing_rates.append(indiv_rate)

        return {
            "firing_rates": np.array(firing_rates),
            "mean_firing_rate": np.mean(firing_rates),
            "std_firing_rate": np.std(firing_rates),
        }

    def _compute_pop_firing_rate(self, n_neurons, binsz, pop_counts):
        return pop_counts / (n_neurons * binsz)  # in kHz

    def _compute_psth(self, desired_spiketrains, binsz=50, window=(0, 10000)):
        allspikes = np.concatenate(desired_spiketrains)
        allspikes_in_window = allspikes[(allspikes >= window[0]) &
                                        (allspikes <= window[1])]  # Memory efficient

        bins = np.arange(window[0], window[1] + binsz, binsz)
        # bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute
        counts, bin_edges = np.histogram(allspikes_in_window, bins=bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # should be = (bins[:-1] + bins[1:]) / 2

        # firerate = self._compute_firing_rate_in_window(window, allspikes_in_window)
        popfirerates = self._compute_pop_firing_rate(len(desired_spiketrains), binsz, counts)
        true_avg_rate = self._compute_true_avg_firing_rate(window, desired_spiketrains)

        return counts, bin_centers, popfirerates, true_avg_rate, allspikes_in_window

    def plot(self, binsz=50, window=(0, 10000), nucleus=None):
        """
        Displays the Peri-Stimulus Time Histogram (PSTH) of the given spike times
        and returns the plot figure (to save if necessary).
        
        :param binsz: integer or float; defines the number of equal-width bins in the range [default: 50]
        :param window: 2-tuple; defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: string; [OPTIONAL] None or name of the nucleus
        :return: object `matplotlib.pyplot.bar <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html>`_
        
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

        # Compute PSTH and set the results as instance attributes
        [self.counts, self.bin_centers, self.popfirerates,
         self.true_avg_rate, self.allspikes_in_window] = \
            self._compute_psth(self.desired_spiketrains, binsz=binsz, window=window)

        # Plot
        plt.bar(self.bin_centers, self.counts, width=binsz,
                alpha=0.7, color="blue", edgecolor="black")
        plt.grid(True, alpha=0.3)

        plt.ylabel("Spike Count")
        plt.xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        plt.title("PSTH - Population Activity of " + str(self.n_neurons) + " neurons" + nucname +
                  "\n (mean firing rate within the window = "
                  + str(self.true_avg_rate["mean_firing_rate"]) + " kHz)")

        plt.show()
        
        return plt

    def analytics_temporal(self, stimulus_onset=0):
        """
        Extracts temporal features from the PSTH counts
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
        pre_stimulus_rates = self.popfirerates[self.bin_centers < stimulus_onset]
        post_stimulus_rates = self.popfirerates[self.bin_centers >= stimulus_onset]
        post_stimulus_times = self.bin_centers[self.bin_centers >= stimulus_onset]

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
        peak_time = self.bin_centers[np.argmax(self.popfirerates)]

        if response_latencies is not None:
            response_periods = post_stimulus_rates > baseline
            if np.any(response_periods):
                response_duration = np.sum(response_periods) * self.binsz
            else:
                response_duration = np.array(0)
        else:
            response_duration = np.array(0)

        # Response dynamics: Temporal pattern (response evolution) and
        # temporal coordination among neurons
        first_spike_times = [np.min(np.array(indiv_spiketimes)[np.array(indiv_spiketimes) >= stimulus_onset])
                             if np.any(np.array(indiv_spiketimes) >= stimulus_onset)
                             else np.inf
                             for indiv_spiketimes in self.desired_spiketrains]
        first_spike_times = [t for t in first_spike_times if t != np.inf]

        if len(first_spike_times) > 0:
            response_spread = np.std(first_spike_times) # spread of first spike times
            response_sequence = np.max(first_spike_times) - np.min(first_spike_times) # total span
        else:
            response_spread = np.array(0)
            response_sequence = np.array(0)
        
        return {
            "peak_latency": peak_time.item() - stimulus_onset,
            "response_latencies": response_latencies,
            "response_duration": response_duration.item(),
            "response_sequence": response_sequence.item(),
            "response_spread": response_spread.item(),
            "temporal_coordination": 1 / (response_spread.item() + 1e-8),
            # "response_profile": self.popfirerates,
        }

    def analytics_rate(self, stimulus_onset=0):
        """
        Extracts rate-based features from the PSTH counts
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
        mean_firing_rate = self.true_avg_rate["mean_firing_rate"]
        std_firing_rate = self.true_avg_rate["std_firing_rate"]

        baseline_rates = []
        response_rates = []

        for indiv_spiketimes in self.desired_spiketrains:
            spiketimes = np.array(indiv_spiketimes)

            # Rates: Baseline vs Response
            baseline_spikes = spiketimes[(spiketimes >= self.window[0]) & (spiketimes < stimulus_onset)]
            response_spikes = spiketimes[spiketimes >= stimulus_onset]

            baseline_duration = stimulus_onset - self.window[0]
            response_duration = self.window[1] - stimulus_onset

            baseline_rates.append( len(baseline_spikes) / baseline_duration
                                   if baseline_duration > 0 else 0 )
            response_rates.append( len(response_spikes) / response_duration
                                   if response_duration > 0 else 0 )

        baseline_rates = np.array(baseline_rates)
        response_rates = np.array(response_rates)
        
        # Rate change metrics
        rate_changes = response_rates - baseline_rates
        fold_changes = response_rates / (baseline_rates + 1e-8)

        # Population coding properties
        active_neurons = np.sum(response_rates > baseline_rates + np.std(baseline_rates))
        sparsity_index = 1 - (mean_firing_rate**2 / np.mean(self.true_avg_rate["firing_rates"]**2))

        return {
            "mean_firing_rate": mean_firing_rate.item(),
            "std_firing_rate": std_firing_rate.item(),
            "avg_time_vary_rate": np.mean(self.popfirerates).item(),
            "mean_baseline_rate": np.mean(baseline_rates).item(),
            "mean_response_rate": np.mean(response_rates).item(),
            "mean_rate_change": np.mean(rate_changes).item(),
            "mean_fold_change": np.mean(fold_changes).item(),
            "active_fraction": (active_neurons / self.n_neurons).item(),
            "population_sparsity": sparsity_index.item(),
            "rate_heterogeneity": (std_firing_rate / mean_firing_rate).item(),
            "response_reliability": (np.sum(rate_changes > 0) / self.n_neurons).item(),
        }

    def analytics_energy(self):
        firing_rates = self.true_avg_rate["firing_rates"]
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