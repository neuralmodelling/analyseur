"""
=======================================================
Two figures for viewing firing rates and power spectrum
=======================================================

The figures are invisibly generated and saved under the current working directory and
under the sub-directory `~/psd_rBCBG/`

1. Figure 1
===========

.. code-block:: text

    Figure 1 for visualizing the power spectrum

    +---------------------+
    |                     |
    |      subplot 1      |
    |                     |
    +---------------------+
    |                     |
    |      subplot 2      |
    |                     |
    +---------------------+
    |                     |
    |      subplot 3      |
    |                     |
    +---------------------+

Figure 1 contains two subplots such that:

+-------+---------------------------------------+-------------------------------------------------------------------------------+
|Subplot| Content                               | Interpretation                                                                |
+=======+=======================================+===============================================================================+
| 1     | mean firing rate (across all neurons) | :func:`analyseur.rbcbg.visual.rate.plot_mean_rate_in_ax`                      |
+-------+---------------------------------------+-------------------------------------------------------------------------------+
| 2     | time-varying power spectrum           | :meth:`analyseur.rbcbg.visual.powspec.VizPSD.PowerSpectrum.plot_tv_in_axis`   |
+-------+---------------------------------------+-------------------------------------------------------------------------------+
| 3     | averaged power spectrum               | :meth:`analyseur.rbcbg.visual.powspec.VizPSD.PowerSpectrum.plot_global_in_ax` |
+-------+---------------------------------------+-------------------------------------------------------------------------------+

2. Figure 2
===========

.. code-block:: text

    Figure 2 for visualizing the firing rates

    +---------------------+
    |                     |
    |      subplot 1      |
    |                     |
    +---------------------+
    |                     |
    |      subplot 2      |
    |                     |
    +---------------------+

Figure 2 contains two subplots such that:

+-------+---------------------------------------+-----------------------------------------------------------------+
|Subplot| Content                               | Interpretation                                                  |
+=======+=======================================+=================================================================+
| 1     | firing rates for all neurons          | :func:`analyseur.rbcbg.visual.rate.plot_rate_all_neurons_in_ax` |
+-------+---------------------------------------+-----------------------------------------------------------------+
| 2     | mean firing rate (across all neurons) | :func:`analyseur.rbcbg.visual.rate.plot_mean_rate_in_ax`        |
+-------+---------------------------------------+-----------------------------------------------------------------+

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analyseur.rbcbg.loader import LoadRates
from analyseur.rbcbg.visual.rate import plot_rate_all_neurons_in_ax, plot_mean_rate_in_ax
from analyseur.rbcbg.visual.powspec import VizPSD
from analyseur.rbcbg.parameters import SimulationParams, SignalAnalysisParams

siganal = SignalAnalysisParams()

# rootpath = "/home/lungsi/DockerShare/rBCBG-ANNarchy/Dopamine-experiments/results/firing_rates/"

# nuclei_list = ["CSN", "PTN", "MSNd1", "MSNd2", "FSI", "STN", "GPe", "GPiSNr", "CmPf"]
# decaylist = np.concatenate([[0.05], np.arange(0.1,1.0,0.1)])

rootpath = "/home/lungsi/DockerShare/rBCBG-ANNarchy/decay/"

nuclei_list = ["CSN", "PTN", "CTX_E", "CTX_I", "FSI", "STN", "GPe", "GPiSNr", "TRN", "TH"]
decaylist = np.concatenate([[0.0], np.arange(0.1,1.0,0.05), [1.0]])

window = siganal.window
binsz = siganal.binsz_100perbin
resolution = None
method = None
withbands = None

for nucleus in nuclei_list:
    for i, decay in enumerate(decaylist):
        str_decay = str(round(decay * 100))
        filename = nucleus + "_model_9_percent_" + str_decay + ".csv"
        fullfilepath = rootpath + str(i) + "/" + filename

        loadR = LoadRates(fullfilepath)

        t_sec, rates_Hz = loadR.get_rates()

        #========= FIGURE-1 =========
        fig = plt.figure(figsize=(25, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

        ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # row 1, col 0
        ax3 = fig.add_subplot(gs[2, 0])  # row 2, col 0

        # ax1 = plot_rate_all_neurons_in_ax(ax1, t_sec, rates_Hz, nucleus=nucleus, window=window)
        ax1 = plot_mean_rate_in_ax(ax1, t_sec, rates_Hz, nucleus=nucleus, window=window)
        fig, ax2 = VizPSD.plot_tv_in_axis(fig, ax2, rates_Hz, resolution=resolution, nucleus=nucleus)
        ax3 = VizPSD.plot_global_in_ax(ax3, rates_Hz, binsz, nucleus=nucleus, resolution=resolution,
                                       method=method, withbands=withbands)
        ax3.set_ylim(ax2.get_ylim())
        ax1.set_xlim(t_sec.min(), t_sec.max())

        plt.tight_layout()

        # Path(f"psd_rBCBG_{method}").mkdir(parents=True, exist_ok=True)
        # plt.savefig(f"psd_rBCBG_{method}/PowerSpectrum_of_" + nucleus + "_" + str_decay + "%.png")
        Path(f"psd_rBCBG").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"psd_rBCBG/PowerSpectrum_of_" + nucleus + "_" + str_decay + "%.png")

        plt.close()

        #========= FIGURE-2 =========
        fig = plt.figure(figsize=(25, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # row 1, col 0

        ax1 = plot_rate_all_neurons_in_ax(ax1, t_sec, rates_Hz, nucleus=nucleus, window=window)
        ax2 = plot_mean_rate_in_ax(ax2, t_sec, rates_Hz, nucleus=nucleus, window=window)

        plt.tight_layout()

        # plt.savefig(f"psd_rBCBG_{method}/rates_of_" + nucleus + "_" + str_decay + "%.png")
        plt.savefig(f"psd_rBCBG/rates_of_" + nucleus + "_" + str_decay + "%.png")

        plt.close()
