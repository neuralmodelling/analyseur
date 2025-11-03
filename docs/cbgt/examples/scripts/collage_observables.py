import matplotlib.pyplot as plt

from analyseur.cbgt.loader import LoadSpikeTimes
from analyseur.cbgt.visual.markerplot import plot_raster_in_ax
from analyseur.cbgt.visual.variation import plotCV_in_ax
from analyseur.cbgt.visual.rate import plot_mean_rate_in_ax

from analyseur.cbgt.visual.composite.observables import get_observables

decayfolderid = {
    "0": 0, "1": 0.10, "2": 0.15, "3": 0.20, "4": 0.25, "5": 0.30,
    "6": 0.35, "7": 0.40, "8": 0.45, "9": 0.50, "10": 0.55, "11": 0.60,
    "12": 0.65, "13": 0.70, "14": 0.75, "15": 0.80, "16": 0.85, "17": 0.90,
    "18": 0.95, "19": 1.0,
}
rootpath = "/home/lungsi/DockerShare/data/17Oct2025/"
regionfolders = ["CORTEX/", "CORTEX/",
                 "BG/", "BG/", "BG/",
                 "BG/", "BG/",
                 "THALAMUS/", "THALAMUS/"]
filenames = ["/spikes_PTN.csv", "/spikes_CSN.csv",
             "/spikes_MSN.csv", "/spikes_STN.csv", "/spikes_FSI.csv",
             "/spikes_GPi.csv", "/spikes_GPe.csv",
             "/spikes_TRN.csv", "/spikes_MD.csv", ]

neurons = "all"
# neurons = range(1, 50)

labels = [r"$\overline{f}$", "CV", r"$CV_2$", "LV", "Dim", "Complex", r"$S_{sync}$", r"$F_{sync}$"]

for i, region in enumerate(regionfolders):
    use_rootpath = rootpath + region
    use_filename = filenames[i]

    for id in decayfolderid.keys():
        fullfilepath = use_rootpath + id + use_filename

        loadST = LoadSpikeTimes(fullfilepath)
        spiketimes_superset = loadST.get_spiketimes_superset()

        spiketimes_set = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=neurons)

        [stat_values, nucleus] = get_observables(use_rootpath, use_filename, id)
        x_axis = np.round(np.array(list(decayfolderid.values())) * 100, decimals=1)

        plt.clf()
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 12))

        # Plot CV
        axes[0, 0].plot(x_axis, stat_values[1, :])
        axes[0, 0].set_xlabel("disinhibition (%)")
        axes[0, 0].set_ylabel(labels[1])

        # Plot CV2
        axes[0, 1].plot(x_axis, stat_values[2, :])
        axes[0, 1].set_xlabel("disinhibition (%)")
        axes[0, 1].set_ylabel(labels[2])

        # Plot LV
        axes[0, 2].plot(x_axis, stat_values[3, :])
        axes[0, 2].set_xlabel("disinhibition (%)")
        axes[0, 2].set_ylabel(labels[3])

        # Plot mean freq
        axes[1, 0].plot(x_axis, stat_values[0, :])
        axes[1, 0].set_xlabel("disinhibition (%)")
        axes[1, 0].set_ylabel(labels[0])

        # Plot Dimensionality
        axes[1, 1].plot(x_axis, stat_values[4, :])
        axes[1, 1].set_xlabel("disinhibition (%)")
        axes[1, 1].set_ylabel(labels[4])

        # Plot Complexity
        axes[1, 2].plot(x_axis, stat_values[5, :])
        axes[1, 2].set_xlabel("disinhibition (%)")
        axes[1, 2].set_ylabel(labels[5])

        plt.tight_layout()
        plt.savefig("Observables_of_" + nucleus + "_" + id + ".png")

        fig2, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(x_axis, stat_values[6, :])
        ax1.set_xlabel("disinhibition (%)")
        ax1.set_ylabel(labels[6])

        ax2.plot(x_axis, stat_values[7, :])
        ax2.set_xlabel("disinhibition (%)")
        ax2.set_ylabel(labels[7])

        plt.tight_layout()
        plt.savefig("Synchrony_of_" + nucleus + "_" + id + ".png")