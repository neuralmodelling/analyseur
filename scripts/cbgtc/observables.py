"""
===================================================
Two figures for viewing some essential observables
===================================================

The figures are invisibly generated and saved under the current working directory and
under the sub-directory `~/observables/`

1. Figure 1
===========

.. code-block:: text

    Figure 1 for each disinhibition experiment

    +-----------+-----------+-----------+
    |           |           |           |
    | subplot 1 | subplot 2 | subplot 3 |
    |           |           |           |
    +-----------+-----------+-----------+
    |           |           |           |
    | subplot 4 | subplot 5 | subplot 6 |
    |           |           |           |
    +-----------+-----------+-----------+

Figure 1 contains six subplots such that for each disinhibition experiment it plots:

+-------+------------------+------------------------------------------------------------------------------+
|Subplot| Content          | Interpretation                                                               |
+=======+==================+==============================================================================+
| 1     | CV               | :meth:`analyseur.cbgtc.stats.variation.Variations.computeCV`                 |
+-------+------------------+------------------------------------------------------------------------------+
| 2     | CV2              | :meth:`analyseur.cbgtc.stats.variation.Variations.computeCV2`                |
+-------+------------------+------------------------------------------------------------------------------+
| 3     | LV               | :meth:`analyseur.cbgtc.stats.variation.Variations.computeLV`                 |
+-------+------------------+------------------------------------------------------------------------------+
| 4     | mean firing rate | :meth:`analyseur.cbgtc.stats.isi.InterSpikeInterval.mean_freqs`              |
+-------+------------------+------------------------------------------------------------------------------+
| 5     | dimensionality   | `n_components_` of :meth:`analyseur.cbgtc.stats.pca.PCA.compute`             |
+-------+------------------+------------------------------------------------------------------------------+
| 6     | complexity       | `n_components_ / n_neurons` of :meth:`analyseur.cbgtc.stats.pca.PCA.compute` |
+-------+------------------+------------------------------------------------------------------------------+

2. Figure 2
===========

.. code-block:: text

    Figure 2 shows plots across all disinhibition experiments

    +---------------------+
    |                     |
    |      subplot 1      |
    |                     |
    +---------------------+
    |                     |
    |      subplot 2      |
    |                     |
    +---------------------+

Figure 2 contains two subplots such that across all disinhibition experiments it plots:

+-------+---------------------------------------+------------------------------------------------------------------+
|Subplot| Content                               | Interpretation                                                   |
+=======+=======================================+==================================================================+
| 1     | basic measure of synchrony            | :meth:`analyseur.cbgtc.stats.sync.Synchrony.compute_basic`       |
+-------+---------------------------------------+------------------------------------------------------------------+
| 2     | Fano factor as a measure of synchrony | :meth:`analyseur.cbgtc.stats.sync.Synchrony.compute_fano_factor` |
+-------+---------------------------------------+------------------------------------------------------------------+

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgtc.loader import LoadSpikeTimes
from analyseur.cbgtc.stats.isi import InterSpikeInterval
from analyseur.cbgtc.stats.variation import Variations
from analyseur.cbgtc.stats.pca import PCA
from analyseur.cbgtc.stats.sync import Synchrony

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
                 "THALAMUS/", "THALAMUS/"]
filenames = ["/spikes_PTN.csv", "/spikes_CSN.csv", "/spikes_IN.csv",
             "/spikes_MSN.csv", "/spikes_STN.csv", "/spikes_FSI.csv",
             "/spikes_GPi.csv", "/spikes_GPe.csv",
             "/spikes_TRN.csv", "/spikes_MD.csv", ]

neurons = "all"

labels = [r"$\overline{f}$", "CV", r"$CV_2$", "LV", "Dim", "Complex",
          r"$S_{sync}$", r"$F_{sync}$", r"$Corr_{pair}$",]

def safe_plot(ax, x, y, xlabel, ylabel, ylim=None):
    if np.all(np.isnan(y)):
        ax.set_visible(False)
        return

    mask = ~np.isnan(y)
    ax.plot(x[mask], y[mask])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)

def safe_plot_spectra(ax, spectra, ylabel, labels=None, max_comp=None, logy=True):

    valid_spectra = [s for s in spectra if s is not None and len(s) > 0]

    if len(valid_spectra) == 0:
        ax.set_visible(False)
        return

    for i, spec in enumerate(spectra):

        if spec is None or len(spec) == 0:
            continue

        if max_comp is not None:
            spec = spec[:max_comp]

        x = np.arange(1, len(spec)+1)

        label = None
        if labels is not None:
            label = labels[i]

        ax.plot(x, spec, marker="o", label=label)

    ax.set_xlabel("PCA component")
    ax.set_ylabel(ylabel) # "Explained variance"

    if logy:
        ax.set_yscale("log")

    ax.grid(True)

    if labels is not None:
        ax.legend(title="disinhibition")

def main():

    for r_idx, region in enumerate(regionfolders):
        use_rootpath = rootpath + region
        use_filename = filenames[r_idx]

        n_test = len(decayfolderid.keys())
        stat_values = np.full((8, n_test), np.nan)

        for t_idx, folder_id in enumerate(decayfolderid.keys()):
            fullfilepath = use_rootpath + folder_id + use_filename

            loadST = LoadSpikeTimes(fullfilepath)
            spiketimes_superset = loadST.get_spiketimes_superset()

            spiketimes_set = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=neurons)
            spiketimes_set = {
                k: v for k, v in spiketimes_set.items()
                if len(v) >= 3
            }
            n_neurons = len(spiketimes_set)
            n_neurons_test = np.zeros(n_test)

            nucleus = loadST.extract_nucleus_name(use_filename)

            try:
                I, _ = InterSpikeInterval.compute(spiketimes_set)
            except Exception:
                continue  # corrupted or empty structure

            I_valid = {k: v for k, v in I.items() if len(v) >= 2}

            if len(I_valid) == 0:
                continue

            mu = InterSpikeInterval.mean_freqs(I_valid)

            cv = Variations.computeCV(I_valid)
            cv2 = Variations.computeCV2(I_valid)
            lv = Variations.computeLV(I_valid)

            with np.errstate(all="ignore"):
                stat_values[0,t_idx] = np.nanmean(list(mu.values()))
                stat_values[1,t_idx] = np.nanmean(list(cv.values()))
                stat_values[2,t_idx] = np.nanmean(list(cv2.values()))
                stat_values[3,t_idx] = np.nanmean(list(lv.values()))

            if len(I_valid) >= 2:
                try:
                    pca_stat, _, _, _ = PCA.compute(spiketimes_set)
                    stat_values[4, t_idx] = pca_stat.n_components_
                    stat_values[5, t_idx] = pca_stat.n_components_ / n_neurons
                except Exception:
                    stat_values[4, t_idx] = np.nan
                    stat_values[5, t_idx] = np.nan

            if n_neurons >= 2:
                try:
                    s_sync, _, _ = Synchrony.compute_basic(spiketimes_set)
                    f_sync, _, _ = Synchrony.compute_fano_factor(spiketimes_set)

                    stat_values[6, t_idx] = s_sync
                    stat_values[7, t_idx] = f_sync

                except Exception:
                    stat_values[6, t_idx] = np.nan
                    stat_values[7, t_idx] = np.nan

            n_neurons_test[t_idx] = n_neurons


        x_axis = np.round(np.array(list(decayfolderid.values())) * 100, decimals=1)

        plt.clf()
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 12))

        ########## FIGURE-1 ##########
        safe_plot(axes[0, 0], x_axis, stat_values[1, :],        # Plot CV
            "disinhibition (%)", labels[1])

        safe_plot(axes[0, 1], x_axis, stat_values[2, :],        # Plot CV2
                "disinhibition (%)", labels[2])

        safe_plot(axes[0, 2], x_axis, stat_values[3, :],        # Plot LV
                "disinhibition (%)", labels[3])

        safe_plot(axes[1, 0], x_axis, stat_values[0, :],        # Plot mean freq
                "disinhibition (%)", labels[0], ylim=(0, 200))

        safe_plot(axes[1, 1], x_axis, stat_values[4, :],        # Plot Dimensionality
                "disinhibition (%)", labels[4])

        safe_plot(axes[1, 2], x_axis, stat_values[5, :],        # Plot Complexity
                "disinhibition (%)", labels[5])

        plt.tight_layout()
        Path("observables").mkdir(parents=True, exist_ok=True)
        plt.savefig("observables/Observables1_of_" + nucleus + ".png")
        plt.close()

        ########## FIGURE-2 ##########
        fig2, (ax1, ax2) = plt.subplots(2, 1)

        safe_plot(ax1, x_axis, stat_values[6, :],
                "disinhibition (%)", labels[6])

        safe_plot(ax2, x_axis, stat_values[7, :],
                "disinhibition (%)", labels[7])

        plt.tight_layout()
        plt.savefig("observables/Synchrony_of_" + nucleus + ".png")
        plt.close()

        plt.close()


# RUN THE SCRIPT
if __name__ == "__main__":
    main()
