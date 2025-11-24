# ~/analyseur/rbcbg/visual/powspec.py
#
# Documentation by Lungsi 4 Nov 2025
#
# This contains function for SpikingStats
#
import numbers

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import re

from analyseur.rbcbg.stats.psd import PowerSpectrum
from analyseur.rbcbg.parameters import SignalAnalysisParams


class VizPSD(object):
    __siganal = SignalAnalysisParams()
    __xlabelsec = "Time (s)"
    __xlabelHz = "Frequency (Hz)"
    __ylabelPSD = "Power Spectral Density"


    @staticmethod
    def plot_in_ax(ax, mu_rate_arr, nucleus=None, resolution=None, method=None):
        # ============== DEFAULT Parameters ==============
        __siganal = SignalAnalysisParams()

        freq_bands = __siganal.freq_bands
        del freq_bands["Low Gamma"]
        del freq_bands["High Gamma"]

        # Compute power spectrum using Welch's method
        freqs, power = PowerSpectrum.compute_for_rate(mu_rate_arr, method=method, resolution=resolution)

        # Plot power spectrum
        ax.semilogy(freqs, power, "b-", linewidth=1, label="Power Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Power Spectrum of neurons" + nucname)

        ax.grid(True, alpha=0.3)

        # Add vertical lines and annotations for band boundaries
        band_boundaries = []
        band_labels = []
        for k, v in freq_bands.items():
            band_labels.append(k)
            if k=="Delta":
                band_boundaries.append(v[0])
                band_boundaries.append(v[1])
            else:
                band_boundaries.append(v[1])

        for boundary in band_boundaries:
            ax.axvline(x=boundary, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

        # Add band labels at the top
        for i, (label, start, end) in enumerate(zip(band_labels, band_boundaries[:-1], band_boundaries[1:])):
            mid = (start + end) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.9, label, ha="center", va="top", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        ax.set_xlim(0, 100)

        return ax


    @classmethod
    def plot(cls, mu_rate_arr, nucleus=None, resolution=None, method=None, mode=None,):
        if mode == "portrait":
            fig, ax = plt.subplots(figsize=(6, 10))
        else:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax = cls.plot_in_ax(ax, mu_rate_arr, nucleus=nucleus, resolution=resolution, method=method)

        plt.show()

        return fig, ax

    # @classmethod
    # def plot_spiketrain_in_ax(cls, ax, spiketrains, yticks, time_axis):
    #     for i, spike_train in enumerate(spiketrains):
    #         ax.plot(time_axis, spike_train + i*0.5, label=yticks[i])
    #
    #     ax.set_xlabel(cls.__xlabelsec)
    #     ax.set_ylabel("Neuron (offset for clarity)")
    #     ax.set_title("Binned Spike Trains")
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
    #
    #     return ax
    #
    # @classmethod
    # def plot_with_spiketrains(cls, spiketimes_superset, neurons=None, nucleus=None,
    #                           window=None, sampling_rate=None, resolution=None,):
    #     fig, axes = plt.subplots(12)
    #
    #     axes[0], [frequencies, power_spectra], [spiketrains, yticks, time_axis] = \
    #         cls.plot_in_ax(axes[0], spiketimes_superset, neurons=neurons, nucleus=nucleus,
    #                        window=window, sampling_rate=sampling_rate, resolution=resolution)
    #
    #     axes[1] = cls.plot_spiketrain_in_ax(axes[1], spiketrains, yticks, time_axis)
    #
    #     plt.tight_layout()
    #     plt.show()