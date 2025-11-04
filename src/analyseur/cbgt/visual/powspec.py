# ~/analyseur/cbgt/visual/powspec.py
#
# Documentation by Lungsi 4 Nov 2025
#
# This contains function for SpikingStats
#
import numbers

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, alpha

import re

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.stats.psd import PowerSpectrum
from analyseur.cbgt.parameters import SignalAnalysisParams
# from analyseur.cbgt.parameters import SignalAnalysisParams, SimulationParams
#
# __siganal = SignalAnalysisParams()
# __simparams = SimulationParams()


class VizPSD(object):
    __siganal = SignalAnalysisParams()

    @classmethod
    def plot_in_ax(cls, ax, spiketimes_superset, neurons=None, nucleus=None,
                   window=None, sampling_rate=None, resolution=None, mode=None,):
        # ============== DEFAULT Parameters ==============
        if neurons is None:
            neurons = "all"
        elif isinstance(neurons, numbers.Number):
            neurons = range(neurons)

        if window is None:
            window = cls.__siganal.window
        if sampling_rate is None:
            sampling_rate = 1 / cls.__siganal.sampling_period

        n_neurons = len(spiketimes_superset)

        match mode:
            case "portrait":
                orient = "horizontal"
            case _:
                orient = "landscape"

        frequencies, power_spectra, spiketrains, yticks, time_axis = \
            PowerSpectrum.compute(spiketimes_superset, neurons=neurons, window=window,
                                  sampling_rate=sampling_rate, resolution=resolution)

        colors = ["red", "blue", "green"]
        for i, (f, Pxx) in enumerate(zip(frequencies, power_spectra)):
            ax.semilogy(f, Pxx, color=colors[i], label=yticks[i], linewidth=2)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")

        nucname = "" if nucleus is None else " in " + nucleus
        allno = str(n_neurons)
        if neurons == "all":
            ax.set_title("Power Spectrum of Spike Trains of all (" + allno + ") the neurons" + nucname)
        else:
            ax.set_title("Power Spectrum of Spike Trains of " + str(neurons[0]) + " to " + str(neurons[-1]) + " neurons" + nucname)

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)  # Focus on lower frequencies (where most neural activity occurs)

        return ax, [frequencies, power_spectra], [spiketrains, yticks, time_axis]

    @classmethod
    def plot(cls, spiketimes_superset, neurons=None, nucleus=None,
             window=None, sampling_rate=None, resolution=None, mode=None,):
        if mode == "portrait":
            fig, ax = plt.subplots(figsize=(6, 10))
        else:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax = cls.plot_in_ax(ax, spiketimes_superset, neurons=neurons, nucleus=nucleus,
                            window=window, sampling_rate=sampling_rate, resolution=resolution, mode=mode,)

        plt.show()

        return fig, ax

    @classmethod
    def plot_spiketrain_in_ax(cls, ax, spiketrains, yticks, time_axis):
        for i, spike_train in enumerate(spiketrains):
            ax.plot(time_axis, spike_train + i*0.5, label=yticks[i])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron (offset for clarity)")
        ax.set_title("Binned Spike Trains")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    @classmethod
    def plot_with_spiketrains(cls, spiketimes_superset, neurons=None, nucleus=None,
                              window=None, sampling_rate=None, resolution=None,):
        fig, axes = plt.subplots(12)

        axes[0], [frequencies, power_spectra], [spiketrains, yticks, time_axis] = \
            cls.plot_in_ax(axes[0], spiketimes_superset, neurons=neurons, nucleus=nucleus,
                           window=window, sampling_rate=sampling_rate, resolution=resolution)

        axes[1] = cls.plot_spiketrain_in_ax(axes[1], spiketrains, yticks, time_axis)

        plt.tight_layout()
        plt.show()