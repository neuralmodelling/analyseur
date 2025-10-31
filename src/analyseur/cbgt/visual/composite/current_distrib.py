# ~/analyseur/cbgt/visual/composite/current_distrib.py
#
# Documentation by Lungsi 21 Oct 2025
#
# This contains function for Peri-Stimulus Time Histogram (PSTH)
#
import matplotlib.pyplot as plt

import numpy as np

from analyseur.cbgt.loader import LoadChannelIorG
from analyseur.cbgt.parameters import SimulationParams, SignalAnalysisParams


def __get_region_name(simparams, nucleus):
    """Returns region name for respective nucleus name for which the spike times are for in the file."""
    if nucleus in simparams.nuclei_ctx:
        region = "cortex"
    elif nucleus in simparams.nuclei_bg:
        region = "bg"
    else:
        region = "thalamus"

    return region

def _get_mean_current(rootpath, dirname, nucleus, attriblist, simparams):
    # NOTE: Following my signal chat with Jeanne on 27 Oct 2025, although the filenames
    # have V in the filenames (V for voltage) Jeanne said the values are actually the
    # measures of mean I (current) across first 400 neurons.
    # Therefore, this function following loading of the files (for respective attribute/channel)
    # it does not take the product of the loaded V files (representing I's) and the respective
    # conductances to get the current. Once the respective files are loaded the function then
    # returns the mean of [mean I_400] across time.
    region = __get_region_name(simparams, nucleus)

    measurables = {}
    for attrib in attriblist:
        filepath = rootpath + dirname + "/" + nucleus + "_V_syn_" + \
                   attrib + "_1msgrouped_mean_preprocessed4Matlab_SHIFT.csv"
        loadIG = LoadChannelIorG(filepath)
        measurables[attrib] = loadIG.get_measurables()

    mean_current_across_t = {"L": np.mean(measurables["L"])}
    for attrib in attriblist:
        if attrib in simparams.neurotrans:
            mean_current_across_t[attrib] = np.mean(measurables[attrib])

    return mean_current_across_t

def get_observables(rootpath, nucleus, attriblist, decayfolderid):

    #nucleus_title = "PTN ("+str(np.round(decayfolderid[dirlist[frame]]*100, decimals=1))+"% decay)"
    simparams = SimulationParams()

    remove_list = ["g_AMPA", "g_NMDA", "g_GABAA", "g_GABAB"]
    remove_set = set(remove_list)

    filtered_attriblist = [item for item in attriblist if item not in remove_set]

    current_means = {}
    for i, dirname in enumerate(decayfolderid):
        current_means_ = _get_mean_current(rootpath, dirname, nucleus, attriblist, simparams)
        if i == 0:
            for attrib in filtered_attriblist:
                current_means[attrib] = [current_means_[attrib]]
        else:
            for attrib in filtered_attriblist:
                current_means[attrib].append(current_means_[attrib])

    return current_means, filtered_attriblist

def plot_current_distrib2(rootpath, nucleus, attriblist, decayfolderid, feedfwd=False, show=True, save=False):
    simparams = SimulationParams()

    [mean_I, filtered_attriblist] = get_observables(rootpath, nucleus, attriblist, decayfolderid)
    x_axis = np.round(np.array(list(decayfolderid.values())) * 100, decimals=1)
    n_experiments = len(x_axis)
    n_attrib = len(filtered_attriblist)
    suptitle = "Current Distribution of " + nucleus

    # Plot
    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set bar width and positions
    bar_width = 0.8 / n_attrib # width changes dynamically with number of attributes (channels)
    x_positions = np.arange(n_experiments)

    # Create bars for each attribute
    for i, attrib in enumerate(filtered_attriblist):
        offset = (i - n_attrib/2 + 0.5) * bar_width
        ax.bar(x_positions + offset, mean_I[attrib], bar_width, label=attrib)

    if feedfwd:
        plt.axhline(y=simparams.ff_currents[__get_region_name(simparams, nucleus)][nucleus],
                    color='b', linestyle='--', label=r"$I_{feedforward}$")

    ax.set_xlabel("Number of Experiments")
    ax.set_ylabel("Mean Current (µA⋅cm"+r"$^{-2}$)")
    ax.set_title(suptitle)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Exp {i+1}" for i in range(n_experiments)])
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()

    if show:
        plt.show()

    if save:
        plt.savefig(suptitle.replace(" ", "_"))


def plot_current_distrib(rootpath, nucleus, attriblist, decayfolderid, show=True, ):
    [mean_I, filtered_attriblist] = get_observables(rootpath, nucleus, attriblist, decayfolderid)

    plt.figure(figsize=(12, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(filtered_attriblist)))

    for i, attrib in enumerate(filtered_attriblist):
        data = mean_I[attrib]
        plt.hist(data, bins=8, alpha=0.7, color=colors[i],
                 label=f"{attrib} (mean: {np.mean(data):.3f}±{np.std(data):.3f} nA)",
                 edgecolor="black", linewidth=0.5)

    plt.xlabel("Mean Current (µA⋅cm"+r"$^{-2}$)")
    plt.ylabel("Number of Experiments")
    plt.title("Distribution of Mean Channel Currents across 10 Experiments")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if show:
        plt.show()


def plotH_current_distrib(rootpath, nucleus, attriblist, decayfolderid, feedfwd=False, show=True, save=False):
    simparams = SimulationParams()

    [mean_I, filtered_attriblist] = get_observables(rootpath, nucleus, attriblist, decayfolderid)
    x_axis = np.round(np.array(list(decayfolderid.values())) * 100, decimals=1)
    n_experiments = len(x_axis)
    suptitle = "Current Distribution of " + nucleus

    plt.figure(figsize=(14, 8))

    # Set bar positions
    x_pos = np.arange(n_experiments)
    width = 0.2
    multiplier = 0

    for attrib in filtered_attriblist:
        data = mean_I[attrib]
        offset = width * multiplier

        plt.bar(x_pos + offset, data, width, label=attrib)
        multiplier += 1

    if feedfwd:
        plt.axhline(y=simparams.ff_currents[__get_region_name(simparams, nucleus)][nucleus],
                    color='b', linestyle='--', label=r"$I_{feedforward}$")

    plt.xlabel("Number of Experiments")
    plt.ylabel("Mean Current (µA⋅cm"+r"$^{-2}$)")
    plt.title(suptitle)
    plt.xticks(x_pos + width * (len(filtered_attriblist) - 1) / 2,
               [f"{i}%" for i in x_axis])
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if show:
        plt.show()

    if save:
        plt.savefig(suptitle.replace(" ", "_"))