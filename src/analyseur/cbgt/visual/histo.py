# ~/analyseur/cbgt/visual/histo.py
#
# Documentation by Lungsi 7 Oct 2025
#
# This contains function for loading the files
#

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from ..loader import get_desired_spiketrains

def _compute_firing_rate_in_window(window, allspikes_in_window):
    total_time = window[1] - window[0]
    total_spikes = len(allspikes_in_window)
    # return 1000 * (total_spikes / total_time)  # in hertz
    return (total_spikes / total_time) # in kHz

def psth(spiketrains, binsz=50, window=(0,10000), nucleus=None):
    """
    Displays the Peri-Stimulus Time Histogram (PSTH) of the given spike times and returns the plot figure (to save if necessary).

    :param spiketrains: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    :param binsz: defines the number of equal-width bins in the range [default: 50]
    :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
    :param nucleus: [OPTIONAL] None or name of the nucleus (string)
    :return: object `matplotlib.pyplot.hist <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html>`_
    
    * `window` controls the binning range as well as the spike counting window
    * CBGT simulation was done in milliseconds so window `(0, 10000)` signifies time 0 ms to 10,000 ms (or 10 s)
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketrains()

      from analyseur.cbgt.visual.histo import psth

    2. Peri-Stimulus Time Histogram for the whole simulation window

    ::

      psth(spike_trains)

    3. PSTH for desired window and bin size

    ::

      psth(spike_trains, window=(0,50), binsz=1)
      psth(spike_trains, window=(0,50), binsz=0.05)
    
    """
    [desired_spiketrains, yticks] = get_desired_spiketrains(spiketrains)

    bins = np.arange(window[0], window[1] + binsz, binsz)
    allspikes = np.concatenate(desired_spiketrains)
    allspikes_in_window = allspikes[(allspikes >= window[0]) & (allspikes <= window[1])] # Memory efficient

    # plt.hist(allspikes, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    plt.hist(allspikes_in_window, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    plt.grid(True, alpha=0.3)

    plt.ylabel("Spike Count")
    plt.xlabel("Time (ms)")

    nucname = "" if nucleus is None else " in " + nucleus
    allno = str(len(desired_spiketrains))
    plt.title("PSTH - Population Activity of " + allno + " neurons" + nucname +
              "\n (firing rate within the window = " +
              str(_compute_firing_rate_in_window(window, allspikes_in_window)) + " kHz)")

    plt.show()

    return plt

def psrh(spiketrains, binsz=10, window=(0,10000), nucleus=None):
    """
    Displays the Population Spike Rate Histogram (PSRH) of the given spike times and returns the plot figure (to save if necessary).

    :param spiketrains: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    :param binsz: defines the number of equal-width bins in the range [default: 50]
    :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
    :param nucleus: [OPTIONAL] None or name of the nucleus (string)
    :return: object `matplotlib.pyplot <https://matplotlib.org/stable/api/pyplot_summary.html>`_
    
    * `window` controls the binning range as well as the spike counting window
    * CBGT simulation was done in milliseconds so window `(0, 10000)` signifies time 0 ms to 10,000 ms (or 10 s)
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketrains()

      from analyseur.cbgt.visual.histo import psrh

    2. Population Spike Rate Histogram for the entire simulation window

    ::

      psrh(spike_trains)

    3. Population Spike Rate Histogram for desired window and bin size

    ::

      psrh(spike_trains, window=(0,50), binsz=1)
      psrh(spike_trains, window=(0,50), binsz=0.05)

    """
    [desired_spiketrains, yticks] = get_desired_spiketrains(spiketrains)
    n_neurons = len(desired_spiketrains)

    bins = np.arange(window[0], window[1] + binsz, binsz)
    t_axis = bins[:-1] + binsz / 2

    pop_rate = np.zeros(len(bins) - 1)
    for spikes in desired_spiketrains:
        counts, _ = np.histogram(spikes, bins=bins)
        pop_rate += counts
    pop_rate = 1000*(pop_rate / (binsz * n_neurons)) # spikes per milliseconds neurons

    plt.plot(t_axis, pop_rate, linewidth=2)
    plt.fill_between(t_axis, pop_rate, alpha=0.3)
    plt.grid(True, alpha=0.3)

    plt.ylabel("Pop. firing rate (Hz)")
    plt.xlabel("Time (ms)")

    nucname = "" if nucleus is None else " in " + nucleus
    plt.title("Population firing rate of " + str(n_neurons) + " neurons" + nucname)

    plt.show()

    return plt
