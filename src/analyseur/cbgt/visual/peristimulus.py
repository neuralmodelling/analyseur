# ~/analyseur/cbgt/visual/peristimulus.py
#
# Documentation by Lungsi 8 Oct 2025
#
# This contains function for Peri-Stimulus Time Histogram (PSTH)
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

    def _compute_firing_rate_in_window(self, window, allspikes_in_window):
        total_time = window[1] - window[0]
        total_spikes = len(allspikes_in_window)
        # return 1000 * (total_spikes / total_time)  # in hertz
        return (total_spikes / total_time)  # in kHz

    def _compute_psth(self, binsz=50, window=(0, 10000)):
        [self.desired_spiketrains, _] = get_desired_spiketrains(self.spiketrains)

        allspikes = np.concatenate(self.desired_spiketrains)
        allspikes_in_window = allspikes[(allspikes >= window[0]) &
                                        (allspikes <= window[1])]  # Memory efficient

        bins = np.arange(window[0], window[1] + binsz, binsz)

        # Compute
        counts, bin_edges = np.histogram(allspikes_in_window, bins=bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fr = self._compute_firing_rate_in_window(window, allspikes_in_window)

        return counts, bin_centers, binsz, fr

    def plot(self, binsz=50, window=(0, 10000), nucleus=None):
        """
        Displays the Peri-Stimulus Time Histogram (PSTH) of the given spike times
        and returns the plot figure (to save if necessary).
        
        :param binsz: defines the number of equal-width bins in the range [default: 50]
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)] 
        :param nucleus: [OPTIONAL] None or name of the nucleus (string) 
        :return: object `matplotlib.pyplot.hist <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html>`_
        
        * `window` controls the binning range as well as the spike counting window
        * CBGT simulation was done in milliseconds so window `(0, 10000)` signifies time 0 ms to 10,000 ms (or 10 s)
        
        """
        # Compute PSTH and set the results as class attributes
        [self.counts, self.bin_centers, self.binsz, self.fr] = \
            self._compute_psth(binsz=binsz, window=window)

        # Plot
        plt.bar(self.bin_centers, self.counts, width=binsz,
                alpha=0.7, color="blue", edgecolor="black")
        plt.grid(True, alpha=0.3)

        plt.ylabel("Spike Count")
        plt.xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        allno = str(len(self.desired_spiketrains))
        plt.title("PSTH - Population Activity of " + allno + " neurons" + nucname +
                  "\n (firing rate within the window = " + str(self.fr) + " kHz)")

        plt.show()
        
        return plt

    def analytics_temporal(self, stimulus_onset=0):
        """
        Extracts temporal features from the PSTH counts
        :param stimulus_onset: [OPTIONAL] default: 0
        :return: dictionary
        
        +------------------------+----------------------+
        | Dictionary key         | Meaning              |
        +========================+======================+
        | `"response_latency"`   | when response begins |
        +------------------------+----------------------+
        | `"response_magnitude"` | peak amount          |
        +------------------------+----------------------+
        | `"peak_latency"`       | when peak occurs     |
        +------------------------+----------------------+
        | `"response_duration"`  | how long it lasts    |
        +------------------------+----------------------+
        | `"temporal_profile"`   | complete time course |
        +------------------------+----------------------+
        | `"adaptation_ratio"`   |                      |
        +------------------------+----------------------+
        
        """
        # When does the neuron respond
        response_threshold = \
            np.mean(self.counts[self.bin_centers < stimulus_onset]) + 2 * np.std( self.counts)

        supra_threshold = self.counts[self.bin_centers >= stimulus_onset] > response_threshold

        if np.any(supra_threshold):
            response_latency = \
                self.bin_centers[self.bin_centers >= stimulus_onset][np.where(supra_threshold)[0][0]] \
                - stimulus_onset
        else:
            response_latency = None
        
        # Temporal pattern or response evolution
        peak_time = self.bin_centers[np.argmax(self.counts)]
        response_duration = len(self.counts[self.bin_centers >= stimulus_onset]) \
                            * (self.bin_centers[1] - self.bin_centers[0])
        
        # Response dynamics
        early_response = np.mean(self.counts[(self.bin_centers >= stimulus_onset) &
                                             (self.bin_centers < stimulus_onset + 0.1)])
        late_response = np.mean(self.counts[(self.bin_centers >= stimulus_onset + 0.1) &
                                            (self.bin_centers < stimulus_onset + 0.5)])
        
        return {
            "response_latency": response_latency,
            "response_magnitude": np.max(self.counts) -
                                  np.mean(self.counts[self.bin_centers < stimulus_onset]),
            "peak_latency": peak_time - stimulus_onset,
            "response_duration": response_duration,
            "temporal_profile": self.counts,
            "adaptation_ratio": late_response /early_response if early_response > 0 else 0,
        }

    def analytics_rate(self, stimulus_onset=0):
        """
        Extracts rate-based features from the PSTH counts
        :param stimulus_onset: [OPTIONAL] default: 0
        :return: dictionary
        
        +--------------------------+---------------------------------+
        | Dictionary key           | Meaning                         |
        +==========================+=================================+
        | `"baseline_rate"`        | spontaneous activity level      |
        +--------------------------+---------------------------------+
        | `"peak_rate"`            | maximum response magnitude      |
        +--------------------------+---------------------------------+
        | `"mean_response_rate"`   | average response strength       |
        +--------------------------+---------------------------------+
        | `"rate_increase"`        | absolute response magnitude     |
        +--------------------------+---------------------------------+
        | `"fold_change"`          | relative response magnitude     |
        +--------------------------+---------------------------------+
        | `"response_reliability"` | measure of response consistency |
        +--------------------------+---------------------------------+
        | `"response_entropy"`     | complexity of rate pattern      |
        +--------------------------+---------------------------------+
        
        """
        pre_stimulus = self.counts[self.bin_centers < stimulus_onset]
        post_stimulus = self.counts[self.bin_centers >= stimulus_onset]
        
        baseline_rate = np.mean(pre_stimulus)
        response_peak = np.max(post_stimulus)
        mean_response = np.mean(post_stimulus)
        
        # Rate change metrics
        rate_increase = response_peak - baseline_rate
        fold_change = response_peak / (baseline_rate + 1e-8)
        
        # Response consistency
        response_consistency = 1 - (np.std(post_stimulus) / (mean_response + 1e-8))
        
        # Information metric
        response_entropy = stats.entropy(post_stimulus + 1e-8)
        
        return {
            "baseline_rate": baseline_rate,
            "peak_rate": response_peak,
            "mean_response_rate": mean_response,
            "rate_increase": rate_increase,
            "fold_change": fold_change,
            "response_reliability": response_consistency,
            "response_entropy": response_entropy,
        }