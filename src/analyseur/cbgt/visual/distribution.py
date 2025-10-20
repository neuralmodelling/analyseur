# ~/analyseur/cbgt/visual/distribution.py
#
# Documentation by Lungsi 17 Oct 2025
#
# This contains function for SpikingStats
#

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import re

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.visual.tabular import SpikingStats


##########################################################################
#    CUMULATIVE SPIKE COUNTS PLOT
##########################################################################

def _get_neuron_count(spiketimes_1neuron):
    spiketimes = np.array(spiketimes_1neuron)
    sorted_spikes = np.sort(spiketimes)
    cumulative = np.arange(1, len(sorted_spikes) + 1)

    return sorted_spikes, cumulative

def _get_pop_count(desired_spiketimes_subset):
    all_spiketimes = np.array(desired_spiketimes_subset)
    all_spikes = np.sort(np.concatenate([spiketimes for spiketimes in all_spiketimes if len(spiketimes) > 0]))
    pop_cumulative = np.arange(1, len(all_spikes) + 1)

    return  all_spikes, pop_cumulative

def spike_counts_distrib(spiketimes_superset):
    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons="all")
    fig, ax = plt.figure(figsize=(10, 6))

    for i, indiv_spiketimes in enumerate(desired_spiketimes_subset):
        if len(indiv_spiketimes) > 0:
            [sorted_spikes, indiv_cumulative] = _get_neuron_count(indiv_spiketimes)

            ax.step(sorted_spikes, indiv_cumulative, where="post", alpha=0.5, linewidth=1)

    [all_spikes, pop_cumulative] = _get_pop_count(desired_spiketimes_subset)

    if len(all_spikes) > 0:
        ax.step(all_spikes, pop_cumulative, where="post", color="red",
                 linewidth=2, label="Population Total")

    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Spike Count")
    ax.set_title("Cumulative Spike Counts")
    ax.set_legend()

    plt.show()

    return fig, ax


##########################################################################
#    SPIKE DENSITY PLOT
##########################################################################

def _get_neuron_densities(spiketimes_1neuron, time_points, bandwidth):
    spiketimes = np.array(spiketimes_1neuron)
    kde = gaussian_kde(spiketimes, bw_method=bandwidth)

    return kde(time_points)

def _get_pop_densities(desired_spiketimes_subset, time_points, bandwidth):
    all_spiketimes = np.array(desired_spiketimes_subset)
    all_spikes = np.sort(np.concatenate([spiketimes for spiketimes in all_spiketimes if len(spiketimes) > 0]))
    kde = gaussian_kde(all_spikes, bw_method=bandwidth)

    return all_spikes, kde(time_points)

def spike_densitites_distrib(spiketimes_superset, window=(0, 10), bandwidth=0.1):
    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons="all")
    time_points = np.linspace(window[0], window[1], 1000) # have to decide on the number 1000

    fig, ax = plt.figure(figsize=(10, 6))

    for i, indiv_spiketimes in enumerate(desired_spiketimes_subset):
        if len(indiv_spiketimes) > 0:
            neuron_density = _get_neuron_densities(indiv_spiketimes, time_points, bandwidth)

            ax.step(time_points, neuron_density + i*0.1, alpha=0.7, linewidth=1)

    [all_spikes, pop_density] = _get_pop_count(desired_spiketimes_subset, time_points, bandwidth)

    if len(all_spikes) > 0:
        ax.step(time_points, pop_density + len(desired_spiketimes_subset)*0.1, "r-",
                linewidth=2, label="Population Average")

    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Density + Offset")
    ax.set_title("Spike Density")
    ax.set_legend()

    plt.show()

    return fig, ax


##########################################################################
#    CV PLOT
##########################################################################

def cv_distrib(spiketimes_superset):
    sstat = SpikingStats(spiketimes_superset)
    compstat = sstat.compute_stats()

    CVarr = compstat["CV_array"].values()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(range(len(CVarr)), CVarr, color="steelblue", edgecolor="black")
    ax.grid(True, alpha=0.3, axis="x")

    # ax.set_ylable()
    # ax.set_xlable()

    ax.set_title("CV")

    plt.show()

    return ax


##########################################################################
#    ISI PLOT
##########################################################################

def isi_distrib(spiketimes_superset, n_bins=50):
    fig, axes = plt.subplots(1, 2)

    all_isis = []
    for neuron_id, spiketimes in spiketimes_superset.items():
        if len(spiketimes) > 1:
            isis = np.diff(spiketimes)
            all_isis.extend(isis)

            axes[0].hist(isis, bins=n_bins, alpha=0.3, density=True)

    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("Interspike Interval (ms)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Neuron ISI Distributions")

    if all_isis:
        axes[1].hist(all_isis, bins=n_bins, alpha=0.7, color="red", density=True)
        axes[1].set_grid(True, alpha=0.3)

        axes[1].set_xlabel("Interspike Interval (ms)")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Population ISI Distributions")

    plt.tight_layout()

    plt.show()

    return fig, axes