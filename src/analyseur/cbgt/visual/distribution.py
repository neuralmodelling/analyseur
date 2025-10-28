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
from analyseur.cbgt.stats.isi import InterSpikeInterval
from analyseur.cbgt.stats.variation import Variations


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

def spike_counts_distrib_in_ax(ax, spiketimes_superset):
    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons="all")

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

    return ax

def spike_counts_distrib(spiketimes_superset):
    fig, ax = plt.figure(figsize=(10, 6))

    ax = spike_counts_distrib_in_ax(ax, spiketimes_superset)

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

def spike_densitites_distrib_in_ax(ax, spiketimes_superset, window=(0, 10), bandwidth=0.1):
    [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons="all")
    time_points = np.linspace(window[0], window[1], 1000) # have to decide on the number 1000

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

    return ax

def spike_densitites_distrib(spiketimes_superset, window=(0, 10), bandwidth=0.1):
    fig, ax = plt.figure(figsize=(10, 6))

    ax = spike_counts_distrib_in_ax(ax, spiketimes_superset, window=(0, 10), bandwidth=0.1)

    plt.show()

    return fig, ax


##########################################################################
#    CV PLOT
##########################################################################

def cv_distrib(spiketimes_superset, orient="vertical", show=True):
    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    all_isi = InterSpikeInterval.compute(spiketimes_superset)
    CVarr = Variations.computeCV(all_isi)
    vec_CV = CVarr.values()

    if orient=="horizontal":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(vec_CV)), vec_CV, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(vec_CV)), vec_CV, color="steelblue", edgecolor="black")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    # ax.set_ylable()
    # ax.set_xlable()

    ax.set_title("CV")

    if show:
        plt.show()

    plt.close()

    return fig, ax


##########################################################################
#    Mean Freq PLOT
##########################################################################

def mean_freq_distrib(spiketimes_superset, orient="vertical", show=True):
    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    all_isi = InterSpikeInterval.compute(spiketimes_superset)
    mu_arr = InterSpikeInterval.mean_freqs(all_isi)
    vec_mu = mu_arr.values()

    if orient=="horizontal":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(vec_mu)), vec_mu, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(vec_mu)), vec_mu, color="steelblue", edgecolor="black")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    # ax.set_ylable()
    # ax.set_xlable()

    ax.set_title("Mean Freq (1/s)")

    if show:
        plt.show()

    plt.close()

    return fig, ax


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
    axes[0].set_xlabel("Interspike Interval (s)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Neuron ISI Distributions")

    if all_isis:
        axes[1].hist(all_isis, bins=n_bins, alpha=0.7, color="red", density=True)
        axes[1].set_grid(True, alpha=0.3)

        axes[1].set_xlabel("Interspike Interval (s)")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Population ISI Distributions")

    plt.tight_layout()

    plt.show()

    return fig, axes