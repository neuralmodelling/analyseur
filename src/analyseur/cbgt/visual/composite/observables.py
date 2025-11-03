# ~/analyseur/cbgt/visual/composite/observables.py
#
# Documentation by Lungsi 20 Oct 2025
#
# This contains function for Peri-Stimulus Time Histogram (PSTH)
#
import re

import matplotlib.pyplot as plt

import numpy as np

from analyseur.cbgt.loader import LoadSpikeTimes
from analyseur.cbgt.visual.tabular import SpikingStats
from analyseur.cbgt.visual.popact import PopAct
from analyseur.cbgt.stats.sync import Synchrony


def get_observables(rootpath, filename, decayfolderid):

    #nucleus_title = "PTN ("+str(np.round(decayfolderid[dirlist[frame]]*100, decimals=1))+"% decay)"

    stat_values = np.zeros((7, len(decayfolderid)))

    for i, dirname in enumerate(decayfolderid.keys()):
        filepath = rootpath + dirname + filename
        loadST = LoadSpikeTimes(filepath)
        spiketimes_superset = loadST.get_spiketimes_superset()
        n_neurons = len(spiketimes_superset)

        ss = SpikingStats(spiketimes_superset)
        pact = PopAct(spiketimes_superset)

        sstats = ss.compute_stats()
        dimstat = pact.analytics()

        stat_values[0, i] = sstats["grand_mean_freqs"]
        stat_values[1, i] = sstats["grand_CV"]
        stat_values[2, i] = sstats["grand_CV2"]
        stat_values[3, i] = sstats["grand_LV"]

        stat_values[4, i] = dimstat["pca"].n_components_  # Dimensionality
        stat_values[5, i] = stat_values[4, i] / n_neurons # Complexity

        stat_values[6, i] = Synchrony.compute_basic(spiketimes_superset)

        [fanofact, _, _] = Synchrony.compute_fano_factor(spiketimes_superset)
        stat_values[7, i] = fanofact

    return stat_values, loadST.extract_nucleus_name(filename)


def __create_multiple_y_axes(ax, num_axes, base_offset=50, right=True):
    axes = [ax]
    for i in range(1, num_axes):
        new_ax = ax.twinx()

        offset = i * base_offset

        if i == 1:
            new_ax.spines["right"].set_position(("outward", 0))
        else:
            if right:
                new_ax.spines["right"].set_position(("outward", offset))
            else:
                new_ax.spines["left"].set_position(("outward", offset))

        axes.append(new_ax)

    return axes

def __get_multiple_axes_legends(axes):
    lines, labels = axes[0].get_legend_handles_labels()
    for i in range(1, len(axes)):
        lines_, labels_ = axes[i].get_legend_handles_labels()
        lines += lines_
        labels += labels_

    return lines, labels


def plot_observables(rootpath, filename, decayfolderid, show=True, save=False):
    [stat_values, nucleus] = get_observables(rootpath, filename, decayfolderid)

    x_axis = np.round(np.array(list(decayfolderid.values())) * 100, decimals=1)

    labels = [r"$\overline{f}$", "CV", r"$CV_2$", "LV", "Dim", "Complex", r"$S_{sync}$", r"$F_{sync}$"]
    colors = ["blue", "red", "green", "violet", "sienna", "slategray", "teal", "darkorange"]
    num_axes = len(labels)
    suptitle = "Spiking statistics of "+ nucleus

    # Plot
    # plt.clf()
    fig = plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(2, 2, 4)

    axes1 = __create_multiple_y_axes(ax1, 3, base_offset=50)
    axes2 = __create_multiple_y_axes(ax2, 2, base_offset=40)
    axes3 = __create_multiple_y_axes(ax3, 2, base_offset=40)

    # Plot the data
    axes1[0].plot(x_axis, stat_values[1, :], color=colors[1], label=labels[1])
    axes1[0].set_xlabel("disinhibition (percentage)")
    axes1[0].set_ylabel(labels[1], color=colors[1])
    axes1[0].tick_params(axis="y", labelcolor=colors[1])

    axes1[1].plot(x_axis, stat_values[2, :], color=colors[2], label=labels[2])
    axes1[1].set_ylabel(labels[2], color=colors[2])
    axes1[1].tick_params(axis="y", labelcolor=colors[2])

    axes1[2].plot(x_axis, stat_values[3, :], color=colors[3], label=labels[3])
    axes1[2].set_ylabel(labels[3], color=colors[3])
    axes1[2].tick_params(axis="y", labelcolor=colors[3])

    lines1, labels1 = __get_multiple_axes_legends(axes1)
    axes1[0].legend(lines1, labels1, loc="upper left")

    axes2[0].plot(x_axis, stat_values[0, :], color=colors[0], label=labels[0])
    axes2[0].set_xlabel("disinhibition (percentage)")
    axes2[0].set_ylabel(labels[0], color=colors[0])
    axes2[0].tick_params(axis="y", labelcolor=colors[0])

    axes2[1].plot(x_axis, stat_values[4, :], color=colors[4], label=labels[4])
    axes2[1].set_ylabel(labels[4], color=colors[4])
    axes2[1].tick_params(axis="y", labelcolor=colors[4])

    lines2, labels2 = __get_multiple_axes_legends(axes2)
    axes2[0].legend(lines2, labels2, loc="upper left")

    axes3[0].plot(x_axis, stat_values[6, :], color=colors[6], label=labels[6])
    axes3[0].set_xlabel("disinhibition (percentage)")
    axes3[0].set_ylabel(labels[6], color=colors[6])
    axes3[0].tick_params(axis="y", labelcolor=colors[6])

    axes3[1].plot(x_axis, stat_values[7, :], color=colors[7], label=labels[7])
    axes3[1].set_ylabel(labels[7], color=colors[7])
    axes3[1].tick_params(axis="y", labelcolor=colors[7])

    lines3, labels3 = __get_multiple_axes_legends(axes3)
    axes3[0].legend(lines3, labels3, loc="upper left")

    plt.suptitle(suptitle)

    # plt.subplots_adjust(right=0.8)
    fig.tight_layout(pad=1.0)

    if show:
        plt.show()

    if save:
        fig.savefig( suptitle.replace(" ", "_") )

def __plot_observables2(rootpath, filename, decayfolderid):
    [stat_values, nucleus] = get_observables(rootpath, filename, decayfolderid)

    x_axis = np.round(np.array(list(decayfolderid.values())) * 100, decimals=1)

    labels = [r"$\overline{f}$", "CV", r"$CV_2$", "LV", "Dim", "Complex", r"$S_{sync}$", r"$F_{sync}$"]
    colors = ["blue", "red", "green", "violet", "sienna", "slategray", "teal", "darkorange"]
    num_axes = len(labels)

    total_offset = 300
    offsets = [i * (total_offset / (num_axes - 1)) for i in range(num_axes)]

    # Plot
    plt.clf()
    fig = plt.figure(1)

    # Grand means
    ax1a = fig.add_subplot(2, 1, 1)

    # Plot grand_mean_CV CV2 & LV on ax1a
    ax1a.plot(x_axis, stat_values[1, :], color=colors[1], label=labels[1])
    ax1a.set_xlabel("disinhibition (percentage)")
    ax1a.set_ylabel(labels[1], color=colors[1])
    ax1a.tick_params(axis="y", labelcolor=colors[1])

    ax1b = ax1a.twinx()
    ax1b.plot(x_axis, stat_values[2, :], color=colors[2], label=labels[2])
    ax1b.set_ylabel(labels[2], color=colors[2])
    ax1b.tick_params(axis="y", labelcolor=colors[2])

    ax1c = ax1a.twinx()
    ax1c.plot(x_axis, stat_values[3, :], color=colors[3], label=labels[3])
    ax1c.set_ylabel(labels[3], color=colors[3])
    ax1c.tick_params(axis="y", labelcolor=colors[3])

    ax1c.spines["right"].set_position(("outward", 60))

    lines_ax1a, labels_ax1a = ax1a.get_legend_handles_labels()
    lines_ax1b, labels_ax1b = ax1b.get_legend_handles_labels()
    lines_ax1c, labels_ax1c = ax1c.get_legend_handles_labels()
    ax1a.legend(lines_ax1a + lines_ax1b + lines_ax1c, labels_ax1a + labels_ax1b + labels_ax1c, loc="upper left")

    ax1a.set_title("Observables: CV, CV2, LV")

    # Plot Mean freq & Dimensionality
    ax2a = fig.add_subplot(2, 2, 3)

    # Plot grand_mean_freqs on ax1
    ax2a.plot(x_axis, stat_values[0, :], color=colors[0], label=labels[0])
    ax2a.set_xlabel("disinhibition (percentage)")
    ax2a.set_ylabel(labels[0], color=colors[0])
    ax2a.tick_params(axis="y", labelcolor=colors[0])

    ax2b = ax2a.twinx()
    ax2b.plot(x_axis, stat_values[4, :], color=colors[4], label=labels[4])
    ax2b.set_ylabel(labels[4], color=colors[4])
    ax2b.tick_params(axis="y", labelcolor=colors[4])

    lines_ax2a, labels_ax2a = ax2a.get_legend_handles_labels()
    lines_ax2b, labels_ax2b = ax2b.get_legend_handles_labels()
    ax2a.legend(lines_ax2a + lines_ax2b, labels_ax2a + labels_ax2b, loc="upper left")

    ax2a.set_title("Observables: Mean freq & Dimensionality")

    # Synchrony
    ax3a = fig.add_subplot(2, 2, 4)

    # Plot grand_mean_freqs on ax1
    ax3a.plot(x_axis, stat_values[6, :], color=colors[6], label=labels[6])
    ax3a.set_xlabel("disinhibition (percentage)")
    ax3a.set_ylabel(labels[6], color=colors[6])
    ax3a.tick_params(axis="y", labelcolor=colors[6])

    ax3b = ax1a.twinx()
    ax3b.plot(x_axis, stat_values[7, :], color=colors[7], label=labels[7])
    ax3b.set_ylabel(labels[7], color=colors[7])
    ax3b.tick_params(axis="y", labelcolor=colors[7])

    ax3b.spines["right"].set_position(("outward", 60))

    lines_ax3a, labels_ax3a = ax3a.get_legend_handles_labels()
    lines_ax3b, labels_ax3b = ax3b.get_legend_handles_labels()
    ax3a.legend(lines_ax3a + lines_ax3b, labels_ax3a + labels_ax3b, loc="upper left")

    ax3a.set_title("Observables: Synchrony")

    plt.show()


def __plot_observables3(rootpath, filename, decayfolderid):
    [stat_values, nucleus] = get_observables(rootpath, filename, decayfolderid)

    x_axis = np.round(np.array(list(decayfolderid.values())) * 100, decimals=1)

    labels = [r"$\overline{f}$", "CV", r"$CV_2$", "LV", "Dim", "Complex", r"$S_{sync}$", r"$F_{sync}$"]
    colors = ["blue", "red", "green", "violet", "sienna", "slategray", "teal", "darkorange"]
    num_axes = len(labels)

    total_offset = 300
    offsets = [i * (total_offset / (num_axes - 1)) for i in range(num_axes)]

    fig, ax1 = plt.subplots(figsize=(30, 8))

    # Plot grand_mean_freqs on ax1
    ax1.plot(x_axis, stat_values[0, :], color=colors[0], label=labels[0])
    ax1.set_xlabel("disinhibition (percentage)")
    ax1.set_ylabel(labels[0], color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    lines_axes, labels_axes = ax1.get_legend_handles_labels()

    # Create twin axes for remaining variables and store all axes
    axes = [ax1]

    for i in range(1, num_axes):
        ax = ax1.twinx()
        ax.plot(x_axis, stat_values[i, :], color=colors[i], label=labels[i])
        ax.set_ylabel(labels[i], color=colors[i])
        ax.tick_params(axis="y", labelcolor=colors[i])

        lines_ax, labels_ax = ax.get_legend_handles_labels()

        ax.spines["right"].set_position(("outward", offsets[i]))

        lines_axes += lines_ax
        labels_axes += labels_ax

        axes.append(ax)

    ax1.legend(lines_axes, labels_axes, loc="upper left")

    plt.title("Observables")

    plt.show()

# rootpath = "/home/lungsi/DockerShare/data/parameter_search/6aMar2025/CORTEX/"
# filename = "/spikes_PTN.csv"
# decayfolderid = {
#     "0": 0, "1": 0.10, "2": 0.15, "3": 0.20, "4": 0.25, "5": 0.30,
#     "6": 0.35, "7": 0.40, "8": 0.45, "9": 0.50, "10": 0.55, "11": 0.60,
#     "12": 0.65, "13": 0.70, "14": 0.75, "15": 0.80, "16": 0.85, "17": 0.90,
#     "18": 0.95, "19": 1.0,
#     }