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
    * `psth` gives an overall temporal pattern of population activity with a picture in both temporal and rate
    
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
    [desired_spiketrains, _] = get_desired_spiketrains(spiketrains)

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
    * `psrh` gives a collective dynamics of the population ensemble
    
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
    [desired_spiketrains, _] = get_desired_spiketrains(spiketrains)
    n_neurons = len(desired_spiketrains)

    bins = np.arange(window[0], window[1] + binsz, binsz)
    t_axis = bins[:-1] + binsz / 2

    # Population Rate Histogram
    pop_rate = np.zeros(len(bins) - 1)
    for spikes in desired_spiketrains:
        counts, _ = np.histogram(spikes, bins=bins)
        pop_rate += counts
    # Histogram to firing rate
    pop_rate = 1000*(pop_rate / (binsz * n_neurons)) # in seconds [default: spikes per milliseconds neurons]

    plt.plot(t_axis, pop_rate, linewidth=2)
    plt.fill_between(t_axis, pop_rate, alpha=0.3)
    plt.grid(True, alpha=0.3)

    plt.ylabel("Pop. firing rate (Hz)")
    plt.xlabel("Time (ms)")

    nucname = "" if nucleus is None else " in " + nucleus
    plt.title("Population Spiking Rate Histogram of " + str(n_neurons) + " neurons" + nucname)

    plt.show()

    return plt

def popactivity(spiketrains, binsz=10, window=(0,10000), nucleus=None):
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
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketrains()

      from analyseur.cbgt.visual.histo import popactivity

    2. Population Activity Heatmap for the entire simulation window

    ::

      popactivity(spike_trains)

    3. Population Activity Heatmap for desired window and bin size

    ::

      popactivity(spike_trains, window=(0,50), binsz=1)
      popactivity(spike_trains, window=(0,50), binsz=0.05)


    """
    [desired_spiketrains, _] = get_desired_spiketrains(spiketrains)
    n_neurons = len(desired_spiketrains)

    bins = np.arange(window[0], window[1] + binsz, binsz)

    # Activity Matrix
    activity = np.zeros((n_neurons, len(bins) - 1))
    for i, spikes in enumerate(desired_spiketrains):
        counts, _ = np.histogram(spikes, bins=bins)
        activity[i] = counts
    activity = activity[::-1, :] # reverse it so that neuron 0 is at the bottom

    plt.imshow(activity, aspect="auto", cmap="hot",
               # extent=[window[0], window[1], n_neurons, 0] # if neuron 0 is at the top by default
               extent = [window[0], window[1], 0, n_neurons])
    plt.colorbar(label="Spike Count per Bin")

    plt.ylabel("neurons")
    plt.xlabel("Time (ms)")

    nucname = "" if nucleus is None else " in " + nucleus
    plt.title("Population Activity Heatmap of " + str(n_neurons) + " neurons" + nucname)

    plt.show()

def popfr_variability(spiketrains, binsz=10, window=(0,10000), nucleus=None):
    """
    Displays the Population Activity Heatmap of the given spike times and returns the plot figure (to save if necessary).

    :param spiketrains: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    :param binsz: defines the number of equal-width bins in the range [default: 50]
    :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
    :param nucleus: [OPTIONAL] None or name of the nucleus (string)
    :return: object `matplotlib.pyplot.imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_
    
    * `window` controls the binning range as well as the spike counting window
    * CBGT simulation was done in milliseconds so window `(0, 10000)` signifies time 0 ms to 10,000 ms (or 10 s)
    * `popfr_variability` gives heterogeneity in population responses
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketrains()

      from analyseur.cbgt.visual.histo import popactivity

    2. Population Activity Heatmap for the entire simulation window

    ::

      popactivity(spike_trains)

    3. Population Activity Heatmap for desired window and bin size

    ::

      popactivity(spike_trains, window=(0,50), binsz=1)
      popactivity(spike_trains, window=(0,50), binsz=0.05)


    """
    [desired_spiketrains, _] = get_desired_spiketrains(spiketrains)
    n_neurons = len(desired_spiketrains)

    bins = np.arange(window[0], window[1] + binsz, binsz)
    t_axis = bins[:-1] + binsz / 2

    # Per bin compute firing rate per neuron
    fr = np.zeros((n_neurons, len(bins)-1))
    for i, spikes in enumerate(desired_spiketrains):
        counts, _ = np.histogram(spikes, bins=bins)
        fr[i] = 1000*(counts / binsz) # to seconds [default: spikes / milliseconds]

    mean_fr = np.mean(fr, axis=0)
    std_fr = np.std(fr, axis=0)

    plt.plot(t_axis, mean_fr, label="Mean", linewidth=2)
    plt.fill_between(t_axis, mean_fr - std_fr, mean_fr + std_fr,
                     alpha=0.3, label="±1 STD")
    plt.grid(True, alpha=0.3)

    plt.ylabel("Firing Rate (Hz)")
    plt.xlabel("Time (ms)")

    nucname = "" if nucleus is None else " in " + nucleus
    plt.title("Population Firing Rate (Mean ± STD) Variability of " + str(n_neurons) + " neurons" + nucname)

    plt.show()