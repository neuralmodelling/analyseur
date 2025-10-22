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

    def plot(self, scales=None, wavelet=None, show=True, save=False, nucleus=None,
             sampling_rate=None, window=None, neurons=None, sigma=None,):
        [coefficients, frequencies, _, time_axis] = \
            WaveletTransform.compute_cwt(self.spiketimes_superset, sampling_rate=sampling_rate,
                                         window=window, neurons=neurons, sigma=sigma,
                                         scales=scales, wavelet=wavelet)
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        mesh = ax.pcolormesh(time_axis, frequencies, np.abs(coefficients),
                             shading="gouraud", cmap="viridis")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_yscale("log")

        fig.colorbar(mesh, ax=ax, label="CWT Coefficient Magnitude")

        nucname = "" if nucleus is None else " of " + nucleus
        ax.set_title("Scalogram" + nucname)

        if show:
            plt.show()

        return fig, ax