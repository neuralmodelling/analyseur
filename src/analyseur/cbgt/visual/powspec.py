# ~/analyseur/cbgt/visual/powspec.py
#
# Documentation by Lungsi 4 Nov 2025
#
# This contains function for SpikingStats
#
import numbers

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    __xlabelsec = "Time (s)"
    __xlabelHz = "Frequency (Hz)"
    __ylabelPSD = "Power Spectral Density"

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

        ax.set_xlabel(cls.__xlabelHz)
        ax.set_ylabel(cls.__ylabelPSD)

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

        ax.set_xlabel(cls.__xlabelsec)
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

    @classmethod
    def plot_aggstat_in_ax(cls, ax_mean_std, ax_percentile, ax_confidence_intervals,
                           spiketimes_set, neurons=None, nucleus=None,
                           window=None, sampling_rate=None, resolution=None,):
        """
        Draws the Aggregate Statistic of the Power Spectral Density of the given neuron population on the given
        `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

        :param ax: 3-objects of the type `matplotlib.pyplot.axis``
        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        [OPTIONAL]

        :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

        :param nucleus: string; name of the nucleus
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param sampling_rate: `1000/dt = 10000` Hz [default]; sampling_rate ∊ (0, 10000)
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: three axes with respective plotting

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if neurons is None:
            neurons = "all"
        elif isinstance(neurons, numbers.Number):
            neurons = range(neurons)

        if window is None:
            window = cls.__siganal.window
        if sampling_rate is None:
            sampling_rate = 1 / cls.__siganal.sampling_period

        n_neurons = len(spiketimes_set)

        frequencies, power_spectra, spiketrains, yticks, time_axis = \
            PowerSpectrum.compute(spiketimes_set, neurons=neurons, window=window,
                                  sampling_rate=sampling_rate, resolution=resolution)

        power_matrix = np.array(power_spectra)
        freqs = frequencies[0]  # all have same frequency axis

        mean_power = np.mean(power_matrix, axis=0)
        std_power = np.std(power_matrix, axis=0)

        median_power = np.median(power_matrix, axis=0)
        p25 = np.percentile(power_matrix, 25, axis=0)
        p75 = np.percentile(power_matrix, 75, axis=0)

        confidence = 0.95
        m = power_matrix.shape[0]
        se = std_power / np.sqrt(m)
        h = se * stats.t.ppf((1 + confidence) / 2., m-1)

        nucname = "" if nucleus is None else " in " + nucleus
        allno = str(n_neurons)

        # Plot1: Mean Power Spectrum ± STD
        ax_mean_std.fill_between(freqs,
                                 mean_power - std_power,
                                 mean_power + std_power,
                                 alpha=0.3, color="blue", label="± STD")
        ax_mean_std.semilogy(freqs, mean_power, "b-", linewidth=2, label="Mean")

        ax_mean_std.set_xlabel(cls.__xlabelHz)
        ax_mean_std.set_ylabel(cls.__ylabelPSD)

        if neurons == "all":
            ax_mean_std.set_title("Mean Power Spectrum ± STD of all (" + allno + ") the neurons" + nucname)
        else:
            ax_mean_std.set_title("Mean Power Spectrum ± STD of " + str(neurons[0]) +
                                  " to " + str(neurons[-1]) + " neurons" + nucname)

        ax_mean_std.legend()
        ax_mean_std.grid(True, alpha=0.3)
        ax_mean_std.set_xlim(0, 100)  # Focus on lower frequencies (where most neural activity occurs)

        # Plot2: Median Power Spectrum with Inter-Quartile Range (IQR)
        ax_percentile.fill_between(freqs, p25, p75, alpha=0.3, color="red",
                                   label="25-75% Percentile")
        ax_percentile.semilogy(freqs, median_power, "r-", linewidth=2, label="Median")

        ax_percentile.set_xlabel(cls.__xlabelHz)
        ax_percentile.set_ylabel(cls.__ylabelPSD)

        if neurons == "all":
            ax_percentile.set_title("Median Power Spectrum with IQR of all (" + allno + ") the neurons" + nucname)
        else:
            ax_percentile.set_title("Mean Power Spectrum with IQR of " + str(neurons[0]) +
                                    " to " + str(neurons[-1]) + " neurons" + nucname)

        ax_percentile.legend()
        ax_percentile.grid(True, alpha=0.3)
        ax_percentile.set_xlim(0, 100)  # Focus on lower frequencies (where most neural activity occurs)

        # Plot3: Mean Power Spectrum with Confidence Interval
        ax_confidence_intervals.fill_between(freqs,
                                             mean_power - h,
                                             mean_power + h,
                                             alpha=0.3, color="green", label=f"{confidence:.0%} CI")
        ax_confidence_intervals.semilogy(freqs, mean_power, "g-", linewidth=2, label="Mean")

        ax_confidence_intervals.set_xlabel(cls.__xlabelHz)
        ax_confidence_intervals.set_ylabel(cls.__ylabelPSD)

        if neurons == "all":
            ax_confidence_intervals.set_title(f"Median Power Spectrum with {confidence:.0%} Confidence Interval of all ("
                                              + allno + ") the neurons" + nucname)
        else:
            ax_confidence_intervals.set_title(f"Median Power Spectrum with {confidence:.0%} Confidence Interval of "
                                              + str(neurons[0]) + " to " + str(neurons[-1]) + " neurons" + nucname)

        ax_confidence_intervals.legend()
        ax_confidence_intervals.grid(True, alpha=0.3)
        ax_confidence_intervals.set_xlim(0, 100)  # Focus on lower frequencies (where most neural activity occurs)

        return ax_mean_std, ax_percentile, ax_confidence_intervals


    @classmethod
    def plot_aggstat(cls, spiketimes_set, neurons=None, nucleus=None,
                     window=None, sampling_rate=None, resolution=None,):
        """
        Visualize the Aggregate Statistic of the Power Spectral Density of the given neuron population.

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        [OPTIONAL]

        :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

        :param nucleus: string; name of the nucleus
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param sampling_rate: `1000/dt = 10000` Hz [default]; sampling_rate ∊ (0, 10000)
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: object figure and three index axes

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        ax1, ax2, ax3 = cls.plot_aggstat_in_ax(axes[0], axes[1], axes[2], spiketimes_set,
                                               neurons=neurons, nucleus=nucleus,
                                               window=window, sampling_rate=sampling_rate, resolution=resolution,)

        plt.show()

        return fig, [ax1, ax2, ax3]

    @classmethod
    def plot_heatmap_in_ax(cls, fig, axes, spiketimes_set, neurons=None, nucleus=None,
                           window=None, sampling_rate=None, resolution=None,):
        """
        Draws the Heatmap of the Power Spectral Density of the given neuron population on the given
        `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

        :param fig: object `matplotlib.figure <https://matplotlib.org/stable/api/figure_api.html>`_
        :param axes: 2-objects of the type `matplotlib.pyplot.axis``
        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        [OPTIONAL]

        :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

        :param nucleus: string; name of the nucleus
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param sampling_rate: `1000/dt = 10000` Hz [default]; sampling_rate ∊ (0, 10000)
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: fig object and the two axes with respective plotting

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if neurons is None:
            neurons = "all"
        elif isinstance(neurons, numbers.Number):
            neurons = range(neurons)

        if window is None:
            window = cls.__siganal.window
        if sampling_rate is None:
            sampling_rate = 1 / cls.__siganal.sampling_period

        n_neurons = len(spiketimes_set)

        frequencies, power_spectra, spiketrains, yticks, time_axis = \
            PowerSpectrum.compute(spiketimes_set, neurons=neurons, window=window,
                                  sampling_rate=sampling_rate, resolution=resolution)

        power_matrix = np.array(power_spectra)
        freqs = frequencies[0]  # all have same frequency axis

        # Sort for peak frequency
        peak_frequencies = []
        for i in range(power_matrix.shape[0]):
            peak_idx = np.argmax(power_matrix[i,:])
            peak_frequencies.append(freqs[peak_idx])

        sort_indices = np.argsort(peak_frequencies)
        sorted_power = power_matrix[sort_indices]

        mean_power = np.mean(power_matrix, axis=0)

        p25 = np.percentile(power_matrix, 25, axis=0)
        p75 = np.percentile(power_matrix, 75, axis=0)

        nucname = "" if nucleus is None else " in " + nucleus
        allno = str(n_neurons)

        # Plot1: Log scale heat map
        im = axes[0].imshow(np.log10(sorted_power + 1e-8),  # avoid RuntimeWarning: divide by zero
                            aspect="auto",
                            extent=[freqs[0], freqs[-1], 0, n_neurons],
                            cmap="viridis")
        fig.colorbar(im, ax=axes[0], label="Log10(Power)")

        axes[0].set_xlabel(cls.__xlabelHz)
        axes[0].set_ylabel("Neuron (sorted by peak freq.)")

        if neurons == "all":
            axes[0].set_title("Power Spectrum Heatmap of all (" + allno + ") the neurons" + nucname)
        else:
            axes[0].set_title("Power Spectrum Heatmap of " + str(neurons[0]) +
                              " to " + str(neurons[-1]) + " neurons" + nucname)

        # Plot2: Average Spectrum
        axes[1].semilogy(freqs, mean_power, "k-", linewidth=2)
        axes[1].fill_between(freqs, p25, p75, alpha=0.3, color="gray")

        axes[1].set_xlabel(cls.__xlabelHz)
        axes[1].set_ylabel(cls.__ylabelPSD)

        if neurons == "all":
            axes[1].set_title("Population Average (median w/ iQR) of all (" + allno + ") the neurons" + nucname)
        else:
            axes[1].set_title("Population Average (median w/ iQR) of " + str(neurons[0]) +
                              " to " + str(neurons[-1]) + " neurons" + nucname)

        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 100)  # Focus on lower frequencies (where most neural activity occurs)

        return axes

    @classmethod
    def plot_heatmap(cls, spiketimes_set, neurons=None, nucleus=None,
                     window=None, sampling_rate=None, resolution=None, ):
        """
        Visualize the Heatmap of the Power Spectral Density of the given neuron population.

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        [OPTIONAL]

        :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

        :param nucleus: string; name of the nucleus
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param sampling_rate: `1000/dt = 10000` Hz [default]; sampling_rate ∊ (0, 10000)
        :param sampling_rate: `10000` [default]
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: object figure and two indexed axes

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

        fig, axes = cls.plot_heatmap_in_ax(fig, axes, spiketimes_set,
                                           neurons=neurons, nucleus=nucleus,
                                           window=window, sampling_rate=sampling_rate, resolution=resolution, )

        plt.show()

        return fig, axes

    @classmethod
    def plot_cluster_in_ax(cls, axes, spiketimes_set, neurons=None, nucleus=None,
                           window=None, sampling_rate=None, resolution=None, ):
        """
        Draws the Power Spectral Density by Cluster of the given neuron population on the given
        `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

        :param axes: 2-objects of the type `matplotlib.pyplot.axis``
        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        [OPTIONAL]

        :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

        :param nucleus: string; name of the nucleus
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param sampling_rate: `1000/dt = 10000` Hz [default]; sampling_rate ∊ (0, 10000)
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: two axes with respective plotting

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if neurons is None:
            neurons = "all"
        elif isinstance(neurons, numbers.Number):
            neurons = range(neurons)

        if window is None:
            window = cls.__siganal.window
        if sampling_rate is None:
            sampling_rate = 1 / cls.__siganal.sampling_period

        n_clusters = 4
        colors = ["red", "blue", "green", "orange", "purple", "brown"]

        n_neurons = len(spiketimes_set)

        frequencies, power_spectra, spiketrains, yticks, time_axis = \
            PowerSpectrum.compute(spiketimes_set, neurons=neurons, window=window,
                                  sampling_rate=sampling_rate, resolution=resolution)

        power_matrix = np.array(power_spectra)
        freqs = frequencies[0]  # all have same frequency axis

        # Standardize and cluster
        X = StandardScaler().fit_transform(power_matrix)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)

        cluster_counts = [np.sum(cluster_labels == i) for i in range(n_clusters)]

        nucname = "" if nucleus is None else " in " + nucleus
        allno = str(n_neurons)

        # Plot by cluster
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_power = power_matrix[cluster_mask]
            cluster_mean = np.mean(cluster_power, axis=0)

            axes[0].semilogy(freqs, cluster_mean,
                             color=colors[cluster_id],
                             linewidth=2,
                             label=f"Cluster {cluster_id} (n={np.sum(cluster_mask)})")
        axes[0].set_xlabel(cls.__xlabelHz)
        axes[0].set_ylabel(cls.__ylabelPSD)

        if neurons == "all":
            axes[0].set_title(f"Power Spectra by Cluster (k={n_clusters}) of all (" + allno + ") the neurons" + nucname)
        else:
            axes[0].set_title(f"Power Spectra by Cluster (k={n_clusters}) of " + str(neurons[0]) +
                              " to " + str(neurons[-1]) + " neurons" + nucname)

        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 100)  # Focus on lower frequencies (where most neural activity occurs)

        # Plot cluster distribution
        axes[1].bar(range(n_clusters), cluster_counts, color=colors[:n_clusters])

        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Number of Neurons")

        if neurons == "all":
            axes[1].set_title("Cluster Distribution of all (" + allno + ") the neurons" + nucname)
        else:
            axes[1].set_title("Cluster Distribution of " + str(neurons[0]) +
                              " to " + str(neurons[-1]) + " neurons" + nucname)

        for i, count in enumerate(cluster_counts):
            axes[1].text(i, count + 1, str(count), ha="center")

        return axes

    @classmethod
    def plot_cluster(cls, spiketimes_set, neurons=None, nucleus=None,
                     window=None, sampling_rate=None, resolution=None, ):
        """
        Visualize the Power Spectral Density by Cluster of the given neuron population.

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        [OPTIONAL]

        :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

        :param nucleus: string; name of the nucleus
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param sampling_rate: `1000/dt = 10000` Hz [default]; sampling_rate ∊ (0, 10000)
        :param sampling_rate: `10000` [default]
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: object figure and two indexed axes

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axes = cls.plot_cluster_in_ax(axes, spiketimes_set,
                                      neurons=neurons, nucleus=nucleus,
                                      window=window, sampling_rate=sampling_rate, resolution=resolution, )

        plt.show()

        return fig, axes