# ~/analyseur/cbgt/visual/correl.py
#
# Documentation by Lungsi 17 Oct 2025
#
# This contains function for SpikingStats
#

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from analyseur.cbgt.curate import get_desired_spiketimes_subset

def cross_correlations(spiketimes_superset, neuron_pairs=None):
    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons="all")

    if neuron_pairs is None:
        neuron_pairs = [(0,1), (0,2), (1,2)] # example

    n_pairs = len(neuron_pairs)

    # Plot
    fig, axes = plt.subplot(1, n_pairs, figsize=(4*n_pairs, 4))

    if n_pairs == 1:
        axes = [axes]

    for idx, (i, j) in enumerate(neuron_pairs):
        bins = np.arange(0, 5, 0.01)

        spikes_i, _ = np.histogram(desired_spiketimes_subset[i], bins=bins)
        spikes_j, _ = np.histogram(desired_spiketimes_subset[j], bins=bins)

        correlation = signal.correlate(spikes_i, spikes_j, mode="full")
        lags = signal.correlation_lags(len(spikes_i), len(spikes_j), mode="full")
        time_lags = lags * 0.01 # convert to seconds

        axes[idx].plot(time_lags, correlation)
        axes[idx].set_xlabel("Time Lag (s)")
        axes[idx].set_ylabel("Cross-correlation")
        axes[idx].set_title(f"Neurons {i}-{j}")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()