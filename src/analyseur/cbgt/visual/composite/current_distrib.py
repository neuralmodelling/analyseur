# ~/analyseur/cbgt/visual/composite/current_distrib.py
#
# Documentation by Lungsi 21 Oct 2025
#
# This contains function for Peri-Stimulus Time Histogram (PSTH)
#
import matplotlib.pyplot as plt

import numpy as np

from analyseur.cbgt.loader import LoadChannelVorG
from analyseur.cbgt.parameters import SimulationParams


def __get_region_name(simparams, nucleus):
    """Returns region name for respective nucleus name for which the spike times are for in the file."""
    if nucleus in simparams.nuclei_ctx:
        region = "cortex"
    elif nucleus in simparams.nuclei_bg:
        region = "bg"
    else:
        region = "thalamus"

    return region

def __get_mean_current(rootpath, dirname, nucleus, attriblist, simparams):
    region = __get_region_name(simparams, nucleus)

    measurables = {}
    for attrib in attriblist:
        filepath = rootpath + dirname + "/" + nucleus + "_V_syn_" + \
                   attrib + "_1msgrouped_mean_preprocessed4Matlab_SHIFT.csv"
        loadVG = LoadChannelVorG(filepath)
        measurables[attrib] = loadVG.get_measurables()

    current_across_400neurons = {"L": measurables["L"] * simparams.conductance[region]["g_L"]}
    for attrib in attriblist:
        for attrib in simparams.neurotrans:
            current_across_400neurons[attrib] = measurables[attrib] * measurables["g_"+attrib]

    # mean_current_across_t = [np.mean(current_across_400neurons["L"])]
    # for chnl in current_across_400neurons.keys():
    #     mean_current_across_t.append(np.mean(current_across_400neurons[chnl]))

    mean_current_across_t = {"L": np.mean(current_across_400neurons["L"])}
    for chnl, current400mean in current_across_400neurons.items():
        mean_current_across_t[chnl] = np.mean(current400mean)

    return mean_current_across_t

def get_observables(rootpath, nucleus, attriblist, decayfolderid):

    #nucleus_title = "PTN ("+str(np.round(decayfolderid[dirlist[frame]]*100, decimals=1))+"% decay)"
    simparams = SimulationParams()

    remove_list = ["g_AMPA", "g_NMDA", "g_GABAA", "g_GABAB"]
    remove_set = set(remove_list)

    filtered_attriblist = [item for item in attriblist if item not in remove_set]

    # current_means = np.zeros((len(filtered_attriblist), len(decayfolderid)))
    # for i, dirname in enumerate(decayfolderid.keys()):
    #     current_means[:,i] = __get_mean_current(rootpath, dirname, nucleus, attriblist, simparams)

    current_means = {}
    for i, dirname in enumerate(decayfolderid):
        current_means_ = __get_mean_current(rootpath, dirname, nucleus, attriblist, simparams)
        if i == 0:
            for attrib in filtered_attriblist:
                current_means[attrib] = [current_means_[attrib]]
        else:
            for attrib in filtered_attriblist:
                current_means[attrib].append(current_means_[attrib])

    return current_means, filtered_attriblist

def plot_current_distrib(rootpath, nucleus, attriblist, decayfolderid):
    [mean_I, filtered_attriblist] = get_observables(rootpath, nucleus, attriblist, decayfolderid)

    plt.figure(figsize=(12, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(filtered_attriblist)))

    for i, attrib in enumerate(filtered_attriblist):
        data = mean_I[attrib]
        plt.hist(data, bins=8, alpha=0.7, color=colors[i],
                 label=f"{attrib} (mean: {np.mean(data):.3f}±{np.std(data):.3f} nA)",
                 edgecolor="black", linewidth=0.5)

    plt.xlabel("Mean Current (nA)")
    plt.ylabel("Number of Experiments")
    plt.title("Distribution of Mean Channel Currents across 10 Experiments")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    plt.show()


def plotH_current_distrib(rootpath, nucleus, attriblist, decayfolderid, show=True, save=False):
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

    plt.xlabel("Number of Experiments")
    plt.ylabel("Mean Current (nA)")
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

# rootpath = "/home/lungsi/DockerShare/data/parameter_search/6aMar2025/CORTEX/"
# nucleus = "CSN"
# attriblist = ["L", "AMPA", "NMDA", "GABAA", "GABAB", "g_AMPA", "g_NMDA", "g_GABAA", "g_GABAB"]
# decayfolderid = {
#     "0": 0, "1": 0.10, "2": 0.15, "3": 0.20, "4": 0.25, "5": 0.30,
#     "6": 0.35, "7": 0.40, "8": 0.45, "9": 0.50, "10": 0.55, "11": 0.60,
#     "12": 0.65, "13": 0.70, "14": 0.75, "15": 0.80, "16": 0.85, "17": 0.90,
#     "18": 0.95, "19": 1.0,
#     }