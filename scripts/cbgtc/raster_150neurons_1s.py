"""
========================================================
Population activity and membrane dynamics of 150 neurons
========================================================

The figure is invisibly generated and saved under the current working directory and
under the sub-directory `~/raster150_1s/`

.. code-block:: text

    +---------------------+-----------+-----------+
    |                     |           |           |
    |      subplot 1      | subplot 2 | subplot 3 |
    |                     |           |           |
    +---------------------+-----------+-----------+
    |                     |                       |
    |      subplot 4      |       subplot 5       |
    |                     |                       |
    +---------------------+-----------------------+

Figure contains five subplots such that for each disinhibition experiment it plots:

+-------+---------------------------------------+----------------------------------------------------------------------+
|Subplot| Content                               | Interpretation                                                       |
+=======+=======================================+======================================================================+
| 1     | raster of all the neurons             | :func:`analyseur.cbgtc.visual.markerplot.plot_raster_in_ax`          |
+-------+---------------------------------------+----------------------------------------------------------------------+
| 2     | CV distribution of all the neurons    | :func:`analyseur.cbgtc.visual.variation.plotCV_in_ax`                |
+-------+---------------------------------------+----------------------------------------------------------------------+
| 3     | mean rate of all the neurons          | :func:`analyseur.cbgtc.visual.rate.plot_mean_rate_spikecounts_in_ax` |
+-------+---------------------------------------+----------------------------------------------------------------------+
| 4     | mean membrane voltage                 | :meth:`analyseur.cbgtc.visual.measurable.VoltageTrace.plot_in_ax`    |
+-------+---------------------------------------+----------------------------------------------------------------------+
| 5     | pooled PSTH                           | :meth:`analyseur.cbgtc.visual.peristimulus.VizPSTH.plot_pool_in_ax`  |
+-------+---------------------------------------+----------------------------------------------------------------------+

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from analyseur.cbgtc.loader import LoadSpikeTimes
from analyseur.cbgtc.visual.markerplot import plot_raster_in_ax
from analyseur.cbgtc.visual.variation import plotCV_in_ax
from analyseur.cbgtc.visual.rate import plot_mean_rate_isi_in_ax, plot_true_avg_inst_rate_in_ax, plot_mean_rate_spikecounts_in_ax
from analyseur.cbgtc.visual.peristimulus import VizPSTH
from analyseur.cbgtc.visual.measurable import VoltageTrace
from analyseur.cbgtc.parameters import SimulationParams, SignalAnalysisParams

siganal = SignalAnalysisParams()

decayfolderid = {
    "0": 0, "1": 0.10, "2": 0.15, "3": 0.20, "4": 0.25, "5": 0.30,
    "6": 0.35, "7": 0.40, "8": 0.45, "9": 0.50, "10": 0.55, "11": 0.60,
    "12": 0.65, "13": 0.70, "14": 0.75, "15": 0.80, "16": 0.85, "17": 0.90,
    "18": 0.95, "19": 1.0,
}
rootpath = "/home/lungsi/DockerShare/data/09Feb2026/"
regionfolders = ["CORTEX/", "CORTEX/", "CORTEX/",
                 "BG/", "BG/", "BG/",
                 "BG/", "BG/",
                 "THALAMUS/", "THALAMUS/"
                 ]
filenames = ["/spikes_PTN.csv", "/spikes_CSN.csv", "/spikes_IN.csv",
             "/spikes_MSN.csv", "/spikes_STN.csv", "/spikes_FSI.csv",
             "/spikes_GPi.csv", "/spikes_GPe.csv",
             "/spikes_TRN.csv", "/spikes_MD.csv",
             ]

# neurons = "all"
# window = None
# binsz = None

neurons = 150
window = (0,1)
binsz = siganal.binsz_10perbin # binsz_100perbin [default]

xlim_cv = {"PTN": 1.3, "CSN": 1.5, "IN": 1.5, "MSN": 0.3, "STN": 1.2, "FSI": 1.75,
           "GPi": 0.5, "GPe": 0.4, "TRN": 0.225, "MD": 1.4}
# xlim_rate = {"PTN": 500, "CSN": 500, "IN": 1.5e8, "MSN": 250, "STN": 250, "FSI": 400,
#              "GPi": 400, "GPe": 300, "TRN": 210, "MD": 210}
xlim_rate = {"PTN": 50, "CSN": 35, "IN": 1.5e8, "MSN": 8, "STN": 55, "FSI": 80,
             "GPi": 160, "GPe": 130, "TRN": 75, "MD": 50}
[xlim_rate.update({e: 100}) for e in xlim_rate]

def main():

    for i, region in enumerate(regionfolders):
        use_rootpath = rootpath + region
        use_filename = filenames[i]

        for folder_id in decayfolderid.keys():
            fullfilepath = use_rootpath + folder_id + use_filename

            loadST = LoadSpikeTimes(fullfilepath)
            spiketimes_superset = loadST.get_spiketimes_superset()

            spiketimes_set = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, window=window, neurons=neurons)

            nucleus = loadST.extract_nucleus_name(use_filename)

            #plt.clf()
            fig = plt.figure(figsize=(25,12))
            gs = matplotlib.gridspec.GridSpec(2, 3, width_ratios=[3, 1, 1], height_ratios=[3,1])

            ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
            ax2 = fig.add_subplot(gs[0, 1])  # row 0, col 1
            ax3 = fig.add_subplot(gs[0, 2])  # row 0, col 2
            ax4 = fig.add_subplot(gs[1, 0])  # row 1, col 0
            ax5 = fig.add_subplot(gs[1, 1:])  # row 1, span cols 1 & 2


            ax1 = plot_raster_in_ax(ax1, spiketimes_set, nucleus=nucleus, neurons=neurons, alpha=False)
            ax2 = plotCV_in_ax(ax2, spiketimes_set, mode="portrait")

            ax3 = plot_mean_rate_spikecounts_in_ax(ax3, spiketimes_set, window=window, binsz=binsz, nucleus=nucleus, mode="portrait")
            # if neurons=="all":
            #     ax3 = plot_mean_rate_isi_in_ax(ax3, spiketimes_set, mode="portrait")
            # else:
            #     ax3 = plot_true_avg_inst_rate_in_ax(ax3, spiketimes_set, mode="portrait")

            try:
                ax2.set_xlim(0, xlim_cv[nucleus])
                ax3.set_xlim(0, xlim_rate[nucleus])
            except:
                pass

            ax4 = VoltageTrace.plot_in_ax(ax4, use_rootpath + folder_id + "/",
                                        nucleus, window=window)

            ax5 = VizPSTH.plot_pool_in_ax(ax5, spiketimes_set, window=window, binsz=binsz,
                                        nucleus=nucleus, neurons=neurons)

            plt.tight_layout()
            #
            Path("raster150_1s").mkdir(parents=True, exist_ok=True)
            #plt.savefig("raster150_1s_40pc/Raster_of_" + nucleus + "_" + f"{round(decayfolderid[folder_id]*100,2)}%" + ".png")
            plt.savefig("raster150_1s/Raster_of_" + nucleus + "_" + f"{round(decayfolderid[folder_id]*100,2)}%" + ".png")
            # plt.savefig("raster_all/Raster_of_" + nucleus + "_" + f"{round(decayfolderid[folder_id]*100,2)}%" + ".png")
            #
            plt.close()


# RUN THE SCRIPT
if __name__ == "__main__":
    main()
