import matplotlib.pyplot as plt

from analyseur.cbgt.loader import LoadSpikeTimes
from analyseur.cbgt.visual.markerplot import plot_raster_in_ax
from analyseur.cbgt.visual.variation import plotCV_in_ax
from analyseur.cbgt.visual.rate import plot_mean_rate_in_ax

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

for i, region in enumerate(regionfolders):
    use_rootpath = rootpath + region
    use_filename = filenames[i]

    for id in decayfolderid.keys():
        fullfilepath = use_rootpath + id + use_filename

        loadST = LoadSpikeTimes(fullfilepath)
        spiketimes_superset = loadST.get_spiketimes_superset()

        spiketimes_set = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=neurons)

        nucleus = loadST.extract_nucleus_name(use_filename)

        plt.clf()
        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=1, ncols=3,
            gridspec_kw={"width_ratios": [3, 1, 1]},
            figsize=(25, 12))

        ax1 = plot_raster_in_ax(ax1, spiketimes_set, nucleus=nucleus, neurons=neurons)
        ax2 = plotCV_in_ax(ax2, spiketimes_set, mode="portrait")
        ax3 = plot_mean_rate_in_ax(ax3, spiketimes_set, mode="portrait")

        plt.tight_layout()
        plt.savefig("Raster_of_" + nucleus + "_" + id + ".png")