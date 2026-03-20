"""
===================================================
Two figures for identifying the oscillatory regimes
===================================================

1. Figure 1
===========

1.1. Structure
--------------

.. code-block:: text

    Figure 1 for each disinhibition experiment

    +---------------------+-----------+-----------+
    |                     |           |           |
    |      subplot 1      | subplot 2 | subplot 3 |
    |                     |           |           |
    +---------------------+-----------+-----------+
    |                     |                       |
    |      subplot 4      |       subplot 5       |
    |                     |                       |
    +---------------------+-----------------------+

Figure 1 contains four subplots such that for each disinhibition experiment it plots:

* subplot 1: raster of all the neurons
* subplot 2: CV distribution of all the neurons
* subplot 3: autocorrelation of all the neurons
* subplot 4: power spectrum of the population rate (mean across all neurons)
* subplot 5: time-series of population rate (mean across all neurons)

1.1. Guide
----------

+-------+---------------------------------------+------------------------------------------------+
|Subplot| Content                               | Interpretation                                 |
+=======+=======================================+================================================+
| 1     | raster of all the neurons             | synchrony by visual detection                  |
+-------+---------------------------------------+------------------------------------------------+
| 2     | CV distribution of all the neurons    | low means regular firing, high means irregular |
+-------+---------------------------------------+------------------------------------------------+
| 3     | autocorrelation of all the neurons    | distinguish SI vs AI                           |
+-------+---------------------------------------+------------------------------------------------+
| 4     | power spectrum of the population rate | detect oscillation even with noise             |
+-------+---------------------------------------+------------------------------------------------+
| 5     | time-series of population rate        | reveals oscillations and variability           |
+-------+---------------------------------------+------------------------------------------------+

phase cancellation issues

2. Figure 2
===========

2.1. Structure
--------------

.. code-block:: text

    Figure 2 shows plots across all disinhibition experiments

    +---------------------+-----------+-----------+
    |                     |           |           |
    |      subplot 1      | subplot 2 | subplot 3 |
    |                     |           |           |
    +---------------------+-----------+-----------+
    |                     |           |           |
    |      subplot 4      | subplot 5 | subplot 6 |
    |                     |           |           |
    +---------------------+-----------+-----------+

Figure 2 contains five subplots such that across all disinhibition experiments it plots:

* subplot 1: times-series of the population rate for all experiments and mean across all experiments
* subplot 2: pooled CV histogram (CV vs Density)
* subplot 3: phase space (CV vs frequency)
* subplot 4: average power spectra across experiments
* subplot 5: peak frequency vs disinhibition
* subplot 6: peak frequency vs disinhibition

2.1. Guide
----------

+-------+---------------------------------------+------------------------------------------------+
|Subplot| Content                               | Interpretation                                 |
+=======+=======================================+================================================+
| 1     | time-series of population rate        | rule out averaging washing out of rates        |
+-------+---------------------------------------+------------------------------------------------+
| 2     | pooled CV histogram (CV vs Density)   | low means regular firing, high means irregular |
+-------+---------------------------------------+------------------------------------------------+
| 3     | phase space (CV vs frequency)         | compare dynamical states                       |
+-------+---------------------------------------+------------------------------------------------+
| 4     | power spectrum of the population rate | rule out phase cancellation issues             |
+-------+---------------------------------------+------------------------------------------------+
| 5     | peak frequency vs disinhibition       | rule out averaging washing out of rates        |
+-------+---------------------------------------+------------------------------------------------+
| 6     | autocorrelation of all the neurons    | distinguish SI vs AI                           |
+-------+---------------------------------------+------------------------------------------------+

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgt.loader import LoadSpikeTimes
from analyseur.cbgt.parameters import SignalAnalysisParams

from analyseur.cbgt.visual.markerplot import plot_raster_in_ax
from analyseur.cbgt.visual.variation import plotCV_in_ax
from analyseur.cbgt.stats.rate import Rate
from analyseur.cbgt.stats.psd import PowerSpectrum
from analyseur.cbgt.stats.compute_shared import autocorr

siganal = SignalAnalysisParams()

decayfolderid = {
    "0": 0, "1": 0.10, "2": 0.15, "3": 0.20, "4": 0.25, "5": 0.30,
    "6": 0.35, "7": 0.40, "8": 0.45, "9": 0.50, "10": 0.55, "11": 0.60,
    "12": 0.65, "13": 0.70, "14": 0.75, "15": 0.80, "16": 0.85, "17": 0.90,
    "18": 0.95, "19": 1.0,
}
rootpath = "/home/lungsi/DockerShare/data/09Feb2026/"
regionfolders = ["CORTEX/", "CORTEX/", "CORTEX/",
                 "BG/",
                 "BG/", "BG/",
                 "BG/", "BG/",
                 "THALAMUS/", "THALAMUS/"
                 ]
filenames = ["/spikes_PTN.csv", "/spikes_CSN.csv", "/spikes_IN.csv",
             "/spikes_MSN.csv",
             "/spikes_STN.csv", "/spikes_FSI.csv",
             "/spikes_GPi.csv", "/spikes_GPe.csv",
             "/spikes_TRN.csv", "/spikes_MD.csv",
             ]

neurons = "all"
window = siganal.window
binsz = siganal.binsz_100perbin

# neurons = 150
# window = (0,1)
# binsz = 0.001

xlim_cv = {"PTN": 1.3, "CSN": 1.5, "IN": 1.5, "MSN": 0.3, "STN": 1.2, "FSI": 1.75,
           "GPi": 0.5, "GPe": 0.4, "TRN": 0.225, "MD": 1.4}
xlim_rate = {"PTN": 500, "CSN": 500, "IN": 1.5e8, "MSN": 250, "STN": 250, "FSI": 400,
             "GPi": 400, "GPe": 300, "TRN": 210, "MD": 210}
# xlim_rate = {"PTN": 50, "CSN": 35, "IN": 1.5e8, "MSN": 8, "STN": 55, "FSI": 80,
#              "GPi": 160, "GPe": 130, "TRN": 75, "MD": 50}
[xlim_rate.update({e: 200}) for e in xlim_rate]

def main():

    for i, region in enumerate(regionfolders):
        use_rootpath = rootpath + region
        use_filename = filenames[i]

        mu_rate_mat = []
        spectra = []
        freqs_ref = None
        all_CV = []
        peak_freqs = []
        cv_trial_mean = []
        autoco = []

        for folder_id in decayfolderid.keys():
            fullfilepath = use_rootpath + folder_id + use_filename

            loadST = LoadSpikeTimes(fullfilepath)
            spiketimes_superset = loadST.get_spiketimes_superset()

            spiketimes_set = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, window=window, neurons=neurons)

            nucleus = loadST.extract_nucleus_name(use_filename)

            #plt.clf()
            fig = plt.figure(figsize=(25,12))
            gs = matplotlib.gridspec.GridSpec(2, 3, width_ratios=[3, 1, 1])

            ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
            ax2 = fig.add_subplot(gs[0, 1])  # row 0, col 1
            ax3 = fig.add_subplot(gs[0, 2])  # row 0, col 2
            ax4 = fig.add_subplot(gs[1, 0])  # row 1, col 0
            ax5 = fig.add_subplot(gs[1, 1:])  # row 1, span cols 1 & 2


            ax1 = plot_raster_in_ax(ax1, spiketimes_set, nucleus=nucleus, neurons=neurons)
            ax2, vec_CV, _ = plotCV_in_ax(ax2, spiketimes_set, mode="portrait")

            for v in vec_CV:  # flat list of finite scalars only
                if np.isscalar(v):
                    if np.isfinite(v):
                        all_CV.append(v)
                else:
                    v_arr = np.array(v).flatten()
                    v_arr = v_arr[np.isfinite(v_arr)]
                    all_CV.extend(v_arr.tolist())

            cv_values = np.array(list(vec_CV))
            cv_values = cv_values[np.isfinite(cv_values)]

            if len(cv_values) > 0:
                cv_trial_mean.append(np.mean(cv_values))
            else:
                cv_trial_mean.append(np.nan)

            # Plot-5
            mu_rate_vec, time_bins = Rate.mean_rate(spiketimes_set=spiketimes_set, window=window,
                                                    binsz=binsz, neurons="all", across="neurons")
            t = np.arange(len(mu_rate_vec)) * binsz

            ax5.plot(t, mu_rate_vec, "b-", linewidth=1)
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Pop. Rate (1/s)")

            # Plot-4
            mu_rate_raw = mu_rate_vec.copy()
            mu_rate_zero = mu_rate_vec - mu_rate_vec.mean()

            freqs, power = PowerSpectrum.compute_for_rate(mu_rate_zero, method="welch")

            max_power = np.max(power)
            if max_power > 0:
                power = power / max_power

            if freqs_ref is None:
                freqs_ref = freqs
            else:
                assert np.allclose(freqs, freqs_ref), "Frequency grids do not match!"

            # ignore DC component (f=0)
            valid_idx = (freqs > 5) & (freqs < 100)

            if np.any(valid_idx):
                f_peak = freqs[valid_idx][np.argmax(power[valid_idx])]
            else:
                f_peak = np.nan

            ax4.plot(freqs, power, color="blue", alpha=0.3, linewidth=1)
            ax4.set_xlabel("Frequency (Hz)")
            ax4.set_yscale("log")

            # Plot-3
            c = autocorr(mu_rate_raw)
            t_lag = np.arange(len(c)) * binsz

            ax3.plot(t_lag, c, color="black", alpha=0.3, linewidth=1)
            ax3.set_xlabel("lag (τ)")
            ax3.set_ylabel("C (τ)")

            mu_rate_mat.append(mu_rate_raw)
            spectra.append(power)
            peak_freqs.append(f_peak)
            autoco.append(c)

            plt.tight_layout()
            #
            Path("dynaregime").mkdir(parents=True, exist_ok=True)
            plt.savefig("dynaregime/Dynamics_(per trial)_of_" + nucleus + "_" + f"{round(decayfolderid[folder_id]*100,2)}%" + ".png")
            #
            plt.close()

        fig2 = plt.figure(figsize=(25,12))
        gs2 = matplotlib.gridspec.GridSpec(2, 3, width_ratios=[3, 1, 1])

        ax1 = fig2.add_subplot(gs2[0, 0])  # row 0, col 0
        ax2 = fig2.add_subplot(gs2[0, 1])  # row 0, col 1
        ax3 = fig2.add_subplot(gs2[0, 2])  # row 0, col 2
        ax4 = fig2.add_subplot(gs2[1, 0])  # row 1, col 0
        ax5 = fig2.add_subplot(gs2[1, 1])  # row 1, col 1
        ax6 = fig2.add_subplot(gs2[1, 2])  # row 1, col 2

        #=============================================================
        # Plot each population rate across neurons for all neurons
        # and plot the mean of these rates (red)
        #=============================================================
        mu_rate_mat = np.array(mu_rate_mat)
        mu_rate = mu_rate_mat.mean(axis=0)

        for r in mu_rate_mat:
            t = np.arange(len(r)) * binsz
            ax1.plot(t, r, color="gray", alpha=0.3, linewidth=1)

        t = np.arange(len(mu_rate)) * binsz
        ax1.plot(t, mu_rate, "r-", linewidth=2)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pop. Rate (1/s)")

        #=============================================================
        # Plot the Pooled CV histogram (CV vs Density)
        #=============================================================
        cv_array = np.array(all_CV)

        # remove invalid values
        cv_array = cv_array[np.isfinite(cv_array)]
        cv_array = cv_array.flatten()  # avoid hidden shape issues

        bins = np.linspace(np.min(cv_array), np.max(cv_array), 51)
        if len(cv_array) > 0:
            ax2.hist(cv_array, bins=bins, density=True)
        else:
            ax2.set_visible(False)
        ax2.set_xlabel("CV")
        ax2.set_ylabel("Density")

        #=============================================================
        # Plot the Phase space (CV vs frequency)
        #=============================================================
        ax3.scatter(cv_trial_mean, peak_freqs)
        ax3.set_xlabel("Mean CV")
        ax3.set_ylabel("Peak Freq. (Hz)")

        #=============================================================
        # Plot the mean power spectra
        #=============================================================
        spectra = np.array(spectra)
        mean_power = spectra.mean(axis=0)

        ax4.plot(freqs_ref, mean_power, color="red", linewidth=3)
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_yscale("log")

        #=============================================================
        # Plot peak frequency vs disinhibition
        #=============================================================
        x_axis = np.array(list(decayfolderid.values())) * 100
        ax5.plot(x_axis, peak_freqs, marker='o')
        ax5.set_xlabel("disinhibition (%)")
        ax5.set_ylabel("Peak frequency (Hz)")

        #=============================================================
        # Plot autocorrelation across trials
        #=============================================================
        autoco = np.array(autoco)
        mean_c = autoco.mean(axis=0)

        ax6.plot(t_lag, mean_c, color="black", linewidth=3)
        ax6.set_xlabel("lag (τ)")
        ax6.set_ylabel("Average C(τ)")

        plt.tight_layout()
        plt.savefig("dynaregime/Dynamics_of_" + nucleus + ".png")
        plt.close()

# RUN THE SCRIPT
if __name__ == "__main__":
    main()
