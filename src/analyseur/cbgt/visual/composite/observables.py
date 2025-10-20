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

        ss = SpikingStats(spiketimes_superset)
        pact = PopAct(spiketimes_superset)

        sstats = ss.compute_stats()
        dimstat = pact.analytics()

        stat_values[0, i] = sstats["grand_mean_freqs"]
        stat_values[1, i] = sstats["grand_CV"]
        stat_values[2, i] = sstats["grand_CV2"]
        stat_values[3, i] = sstats["grand_LV"]

        stat_values[4, i] = dimstat["pca"].n_components_  # Dimensionality

        stat_values[5, i] = Synchrony.compute_basic(spiketimes_superset)

        [fanofact, _, _] = Synchrony.compute_fano_factor(spiketimes_superset)
        stat_values[6, i] = fanofact

    return stat_values


def plot_observables(rootpath, filename, decayfolderid):
    stat_values = get_observables(rootpath, filename, decayfolderid)

    x_axis = np.round(np.array(list(decayfolderid.values())) * 100, decimals=1)

    fig, ax1 = plt.subplots()

    # Plot grand_mean_freqs on ax1
    ax1.plot(x_axis, stat_values[0, :], color="blue", label=r"$\overline{f}$")
    ax1.set_xlabel("disinhibition (percentage)")
    ax1.set_ylabel(r"$\overline{f}$", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Twin the axis for grand_CV
    ax2 = ax1.twinx()
    ax2.plot(x_axis, stat_values[1, :], color="red", label="CV")
    ax2.set_ylabel("CV", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Twin again for CV2
    ax3 = ax1.twinx()
    ax3.plot(x_axis, stat_values[2, :], color="green", label=r"$CV_2$")
    ax3.set_ylabel(r"$CV_2$", color="green")
    ax3.tick_params(axis="y", labelcolor="green")

    # Twin again for LV
    ax4 = ax1.twinx()
    ax4.plot(x_axis, stat_values[3, :], color="violet", label="LV")
    ax4.set_ylabel("LV", color="violet")
    ax4.tick_params(axis="y", labelcolor="violet")

    # Twin again for Dimensionality
    ax5 = ax1.twinx()
    ax5.plot(x_axis, stat_values[4, :], color="sienna", label="Dim")
    ax5.set_ylabel("Dim", color="sienna")
    ax5.tick_params(axis="y", labelcolor="sienna")

    # Twin again for Dimensionality
    ax6 = ax1.twinx()
    ax6.plot(x_axis, stat_values[5, :], color="teal", label=r"$S_{sync}$")
    ax6.set_ylabel(r"$S_{sync}$", color="teal")
    ax6.tick_params(axis="y", labelcolor="teal")

    # Twin again for Dimensionality
    ax7 = ax1.twinx()
    ax7.plot(x_axis, stat_values[6, :], color="darkorange", label=r"$F_{sync}$")
    ax7.set_ylabel(r"$F_{sync}$", color="darkorange")
    ax7.tick_params(axis="y", labelcolor="darkorange")

    # Offset to avoid overlap
    ax7.spines["right"].set_position(("outward", 60))

    # Combine Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    lines5, labels5 = ax5.get_legend_handles_labels()
    lines6, labels6 = ax6.get_legend_handles_labels()
    lines7, labels7 = ax7.get_legend_handles_labels()

    ax1.legend(lines1 + lines2 + lines3 + lines4 + lines5 + lines6 + lines7,
               labels1 + labels2 + labels3 + labels4 + labels5 + labels6 + labels7,
               loc="upper left")

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