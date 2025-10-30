# ~/analyseur/cbgt/visual/scalo.py
#
# Documentation by Lungsi 21 Oct 2025
#
# This contains function for SpikingStats
#
import numpy as np
import matplotlib.pyplot as plt

from analyseur.cbgt.stats.wavelet import WaveletTransform

class Scalogram(object):

    def __init__(self, spiketimes_superset):
        self.spiketimes_superset = spiketimes_superset
        # get_binary_spiketrains(spiketimes_superset, window=(0, 10), sampling_rate=None, neurons="all"):

    def plot_single(self, scales=None, wavelet=None, show=True, save=False, nucleus=None,
                    sampling_rate=None, window=None, neurons=None, sigma=None, neuron_indx=None):
        [coefficients, frequencies, neuronid, time_axis] = \
            WaveletTransform.compute_cwt_single(self.spiketimes_superset, sampling_rate=sampling_rate,
                                                window=window, neurons=neurons, sigma=sigma,
                                                scales=scales, wavelet=wavelet, neuron_indx=neuron_indx)

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f" ({neuronid})"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        mesh = ax.pcolormesh(time_axis, frequencies, np.abs(coefficients),
                             shading="gouraud", cmap="YlOrRd")
        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")

        fig.colorbar(mesh, ax=ax, label="CWT Coefficient Magnitude")

        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax

    def plot_avg(self, scales=None, wavelet=None, show=True, save=False, nucleus=None,
                sampling_rate=None, window=None, neurons=None, sigma=None,):
        [avg_coefficients, frequencies, yticks, time_axis] = \
            WaveletTransform.compute_cwt_avg(self.spiketimes_superset, sampling_rate=sampling_rate,
                                            window=window, neurons=neurons, sigma=sigma,
                                            scales=scales, wavelet=wavelet, )

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f"( {len(yticks)} average)"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        mesh = ax.pcolormesh(time_axis, frequencies, avg_coefficients,
                             shading="gouraud", cmap="YlOrRd")
        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")

        fig.colorbar(mesh, ax=ax, label="CWT Coefficient Magnitude")

        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax

    def plot_sum(self, scales=None, wavelet=None, show=True, save=False, nucleus=None,
                sampling_rate=None, window=None, neurons=None, sigma=None,):
        [coefficients, frequencies, yticks, time_axis] = \
            WaveletTransform.compute_cwt_sum(self.spiketimes_superset, sampling_rate=sampling_rate,
                                            window=window, neurons=neurons, sigma=sigma,
                                            scales=scales, wavelet=wavelet, )

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f"( {len(yticks)} sum)"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        mesh = ax.pcolormesh(time_axis, frequencies, coefficients,
                             shading="gouraud", cmap="YlOrRd")
        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")

        fig.colorbar(mesh, ax=ax, label="CWT Coefficient Magnitude")

        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax

    def plot_avg_coi(self, scales=None, wavelet=None, show=True, save=False, nucleus=None,
                     sampling_rate=None, window=None, neurons=None, sigma=None, ):
        [avg_coefficients, frequencies, yticks, time_axis] = \
            WaveletTransform.compute_cwt_avg(self.spiketimes_superset, sampling_rate=sampling_rate,
                                             window=window, neurons=neurons, sigma=sigma,
                                             scales=scales, wavelet=wavelet, )
        magnitude = np.abs(avg_coefficients)

        # Create Simple parabolic COI
        T = time_axis[-1]
        f_max = frequencies[0]
        f_min = frequencies[-1]

        # Create parabolic COI
        coi_freq = np.logspace(np.log10(f_min), np.log10(f_max), 50)
        coi_width = 0.15 * T * (coi_freq / f_min)**(-0.5)  # parabola shape

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f"( {len(yticks)} average)"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        mesh = ax.pcolormesh(time_axis, frequencies, magnitude,
                             shading="gouraud", cmap="YlOrRd")
        # plot COI
        ax.plot(0.5 * T - coi_width, coi_freq, "w--", linewidth=2, alpha=0.8)
        ax.plot(0.5 * T + coi_width, coi_freq, "w--", linewidth=2, alpha=0.8)

        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        fig.colorbar(mesh, ax=ax, label="CWT Coefficient Magnitude")

        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax