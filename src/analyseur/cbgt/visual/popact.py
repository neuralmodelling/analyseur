# ~/analyseur/cbgt/visual/popact.py
#
# Documentation by Lungsi 10 Oct 2025
#
# This contains function for Population Activity Heatmap
#
from typing import Collection

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import skew, kurtosis

# from ..curate import get_desired_spiketimes_superset
from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.stats.pca import PCA
from analyseur.cbgt.stats.popact import PopAct
from analyseur.cbgt.parameters import SignalAnalysisParams

class VizPopAct(object):
    """
    The PopAct Class is instantiated by passing

    :param spiketimes_superset: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
    
    +--------------------------------+--------------------------------------------------------------------+
    | Methods                        | Return                                                             |
    +================================+====================================================================+
    | :py:meth:`.plot`               | - `matplotlib.pyplot.imshow` object                                |
    +--------------------------------+--------------------------------------------------------------------+

    * `PopAct.plot` gives a spatio-temporal pattern across neurons
    
    **Use Case:**

    1. Setup

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spiketimes_superset = loadST.get_spiketimes_superset()

      from analyseur.cbgt.visual.popact import VizPopAct

      my_pact = PopAct(spiketimes_superset)

    2. Population Activity Heatmap for the entire simulation window

    ::

      my_pact.plot()

    3. Population Activity Heatmap for desired window and bin size

    ::

      my_pact.plot(spike_trains, window=(0,5), binsz=1)    # time unit in seconds
      my_pact.plot(spike_trains, window=(0,5), binsz=0.05)

    """
    __siganal = SignalAnalysisParams()

    @staticmethod
    def _compute_activity(desired_spiketimes_superset, binsz=0.05, window=(0, 10)):
        bins = np.arange(window[0], window[1] + binsz, binsz)

        # Activity Matrix
        activity = np.zeros((len(desired_spiketimes_superset), len(bins) - 1))
        for i, spikes in enumerate(desired_spiketimes_superset):
            counts, _ = np.histogram(spikes, bins=bins)
            activity[i] = counts
        activity = activity[::-1, :]  # reverse it so that neuron 0 is at the bottom

        return activity, bins

    @classmethod
    def plot_heatmap_in_ax(cls, fig, ax, spiketimes_set, binsz=None, window=None, nucleus=None,):
        """
        .. code-block:: text

            Population Activity Heatmap

            Neurons
            ^
            |  █▓▒░█▓▒░█▓▒░█▓▒░█▓▒░█▓▒░
            |  ░▒▓█░▒▓█░▒▓█░▒▓█░▒▓█░▒▓█
            |  █▓▒░█▓▒░█▓▒░█▓▒░█▓▒░█▓▒░
            |  ░▒▓█░▒▓█░▒▓█░▒▓█░▒▓█░▒▓█
            |
            +--------------------------------------------------> Time (s)
            0        2        4        6        8        10

            Each cell represents the spike count of a neuron
            within a time bin. Color intensity encodes activity.

        Displays the Population Activity Heatmap of the given spike times on the given `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_ and returns the plot figure (to save if necessary)

        :param fig: object `matplotlib.pyplot.figure`
        :param ax: object `matplotlib.pyplot.axis``
        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: defines the number of equal-width bins in the range [default: 50]
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)

        :return: object `matplotlib.pyplot.imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_
    
        * `window` controls the binning range as well as the spike counting window
        * CBGT simulation was done in seconds so window `(0, 10)` signifies time 0 s to 10 s
        * `popactivity` gives a spatio-temporal pattern across neurons

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        # Get and set desired_spiketimes_superset as instance attribute
        [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons="all")
        # NOTE: desired_spiketimes_superset as nested list and not numpy array because
        # each neuron may have variable length of spike times
        n_neurons = len(desired_spiketimes_subset)

        # Compute activities in activity matrix and set the results as instance attributes
        [activity_matrix, bins] = cls._compute_activity(desired_spiketimes_subset,
                                                        binsz=binsz, window=window)

        t_axis = bins[:-1] + binsz / 2

        # Plot
        im = ax.imshow(activity_matrix, aspect="auto", cmap="hot",
                       # extent=[window[0], window[1], n_neurons, 0] # if neuron 0 is at the top by default
                       extent=[window[0], window[1], 0, n_neurons])
        fig.colorbar(im, ax=ax, label="Spike Count per Bin")

        ax.set_ylabel("neurons")
        ax.set_xlabel("Time (s)")

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Population Activity Heatmap of " + str(n_neurons) + " neurons" + nucname)

        return fig, ax

    @classmethod
    def plot_heatmap(cls, spiketimes_set, binsz=None, window=None, nucleus=None, ):
        """
        Displays the Population Activity Heatmap of the given spike times and returns the plot figure (to save if necessary) using :py:meth:`.plot_heatmap_in_ax`

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: defines the number of equal-width bins in the range [default: 50]
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)
        :param show: boolean [default: True]
        :return: object `matplotlib.pyplot.imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_

        * `window` controls the binning range as well as the spike counting window
        * CBGT simulation was done in seconds so window `(0, 10)` signifies time 0 s to 10 s
        * `popactivity` gives a spatio-temporal pattern across neurons

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        fig, ax = cls.plot_heatmap_in_ax(fig, ax, spiketimes_set, window=window, binsz=binsz, nucleus=nucleus,)

        plt.show()

        return fig, ax

    @classmethod
    def plot_pcatraj_in_ax(cls, fig, axes, spiketimes_set, binsz=None, window=None, n_comp=None, nucleus=None,):
        """
        .. code-block:: text

            PCA Trajectory of Population Activity

            Principal Component Analysis
            ┌──────────────────────────────────────────────────────────────┐
            │                     PC1 Over Time                            │
            │  PC1                                                         │
            │   ^        ~~      ~~      ~~       ~~      ~~               │
            │   |      ~~  ~~  ~~  ~~  ~~  ~~  ~~  ~~  ~~  ~~              │
            │   |____~~____~~____~~____~~____~~____~~____~~_______________ │
            │                                                              │
            │   +-----------------------------------------------> Time (s) │
            └──────────────────────────────────────────────────────────────┘

            ┌───────────────────────────────┐   ┌───────────────────────────────┐
            │ PCA Trajectory (PC1 vs PC2)   │   │ PCA Variance Explained        │
            │                               │   │                               │
            │  PC2 ^                        │   │ Variance ^                    │
            │      |   • • • • • • •        │   │         █                     │
            │      |  • • • • • • • •       │   │         ███                   │
            │      | • • • • • • • • •      │   │         █████                 │
            │      |  • • • • • • • •       │   │         ███████               │
            │      |   • • • • • • •        │   │         █████████             │
            │      +--------------------->  │   │         +-------------------> │
            │            PC1                │   │           Principal Component │
            └───────────────────────────────┘   └───────────────────────────────┘

            Top: temporal evolution of the first principal component.
            Bottom-left: trajectory of activity in PC space (colored by time).
            Bottom-right: variance explained by each principal component.

        Displays the PCA Trajectory of population activity on the given `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_ and returns the plot figure (to save if necessary)

        :param fig: object `matplotlib.pyplot.figure`
        :param ax: object `matplotlib.pyplot.axis``
        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: defines the number of equal-width bins in the range [default: 50]
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param n_comp: integer or float; `0.95` [default]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)
        :return: fig object and axes object comprising of three suplots

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        pca, pca_trajectory, activity_matrix, time_bins = PCA.compute(spiketimes_set, binsz=binsz,
                                                                      window=window, n_comp=n_comp)
        t_points = np.linspace(window[0], window[1], pca_trajectory.shape[0])
        n_neurons = len(spiketimes_set)

        # Plot
        # PC1 vs Time
        axes[0].plot(t_points, pca_trajectory[:, 0], linewidth=2)
        axes[0].grid(True, alpha=0.3)

        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("PC1")
        axes[0].set_title("PC1 Over Time")

        # PC1 vs PC2
        scatter = axes[1].scatter(pca_trajectory[:,0], pca_trajectory[:,1],
                              c=t_points, cmap="viridis", s=50, alpha=0.7)
        fig.colorbar(scatter, ax=axes[1], label="Time (s)")

        axes[1].set_xlabel("PC1 ({:.1f}%".format(pca.explained_variance_ratio_[0]*100))
        axes[1].set_ylabel("PC2 ({:.1f}%".format(pca.explained_variance_ratio_[1] * 100))
        axes[1].set_title("PCA Trajectory: PC1 vs PC2")

        # Variance explained?
        components = range(1, min(10, len(pca.explained_variance_ratio_)) + 1)
        axes[2].bar(components, pca.explained_variance_ratio_[:len(components)])

        axes[2].set_xlabel("Principal Component")
        axes[2].set_ylabel("Variance Explained")
        axes[2].set_title("PCA Variance Explained")

        nucname = "" if nucleus is None else " in " + nucleus
        fig.suptitle(' Principal Component Analysis of ' + str(n_neurons) + " neurons" + nucname, fontsize=14)

        return fig, axes

    @classmethod
    def plot_pcatraj(cls, spiketimes_set, binsz=None, window=None, nucleus=None, ):
        """
        Displays the Population Activity Heatmap of the given spike times and returns the plot figure (to save if necessary) using :py:meth:`.plot_pcatraj_in_ax`

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: defines the number of equal-width bins in the range [default: 50]
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)
        :param show: boolean [default: True]
        :return: fig object and axes object comprising of three suplots

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        plt.close("all")   # <-- important to avoid an additional blank white pop-up window
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        fig = plt.figure(figsize=(10,6))

        gs = fig.add_gridspec(2,2)

        ax1 = fig.add_subplot(gs[0, :])   # top row (span 2 columns)
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        fig, axes = cls.plot_pcatraj_in_ax(fig, [ax1,ax2,ax3], spiketimes_set,
                                           binsz=binsz, window=window, n_comp=0.95, nucleus=nucleus,)
        plt.tight_layout()
        plt.show()

        return fig, axes

    # def analytics(self, binsz=0.05, window=(0, 10)):
    #     required_attributes = ["activity_matrix", "pca"]
    #
    #     all_required_exist = all(hasattr(self, attr) for attr in required_attributes)
    #
    #     if all_required_exist:
    #         return {
    #             "activity": self.activity_matrix,
    #             "time_points": self.t_points,
    #             "scaler": self.scaler,
    #             "pca": self.pca,  # if n_comp=0.9 => dimensionality = self.pca.n_components_
    #             "pca_trajectory": self.pca_traj,
    #             "explained_variance": self.pca.explained_variance_ratio_,
    #         }
    #     else:
    #         [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(self.spiketimes_superset, neurons="all")
    #
    #         [activity_matrix, _] = self._compute_activity(desired_spiketimes_subset, binsz=binsz, window=window)
    #         [scaler, pca, pca_traj] = self._compute_PCA(activity_matrix, n_comp=0.90)
    #         t_points = np.linspace(window[0], window[1], pca_traj.shape[0])
    #
    #         return {
    #             "activity": activity_matrix,
    #             "time_points": t_points,
    #             "scaler": scaler,
    #             "pca": pca,  # if n_comp=0.9 => dimensionality = self.pca.n_components_
    #             "pca_trajectory": pca_traj,
    #             "explained_variance": pca.explained_variance_ratio_,
    #         }


    @classmethod
    def plot_popcount_dist_in_ax(cls, ax, spiketimes_set, binsz=None, nucleus=None, mode=None):
        """
        .. code-block:: text

            Population Spike Count Distribution

            Probability Density
            ^
            |            █
            |          ████
            |        ███████
            |      ██████████
            |    ██████████████
            |  ██████████████████
            |        │
            |        │  Mean
            +-----------------------------------------------> Spike Counts per Bin
            0      50      100      150      200      250

            Histogram shows the distribution of population spike counts
            (complexity) across time bins. The vertical marker indicates
            the mean spike count.

        Draws the Distribution of Population Spike Counts (complexity) across time bin on the given
        `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: integer or float; `0.01` [default]

        OPTIONAL parameters

        - :param nucleus: string; name of the nucleus
        - :param mode: "portrait" or None/landscape [default]
        - :return: object `ax` with Rate Distribution plotting done into it

        **INTERPRETATION**

        .. table:: Interpretation
        ====================== ===============================
          Distribution Shape     Interpretation
        ====================== ===============================
          Poisson-like           Random Independent firing
          Right skew             Bursty/synchronous activity
          Narrow distribution    Regular/independent firing
        ====================== ===============================

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        n_neurons = len(spiketimes_set)

        match mode:
            case "portrait":
                orient = "horizontal"
            case _:
                orient = "landscape"

        get_axis = lambda orient: "x" if orient == "horizontal" else "y"

        spike_counts_per_bin, _ = PopAct.count_allspikes_per_bin(spiketimes_set, binsz)

        complexities, complexity_counts, pdf = PopAct.compute_complexity_pdf(spike_counts_per_bin)

        mean_complexity = np.mean(spike_counts_per_bin)

        # Calculate statistics
        mean = np.mean(spike_counts_per_bin)
        var = np.var(spike_counts_per_bin)
        sk = skew(spike_counts_per_bin)
        kurt = kurtosis(spike_counts_per_bin)

        # Plot
        if orient == "horizontal":
            ax.barh(complexities, pdf, alpha=0.7)
            # Add statistics
            ax.axhline(y=mean_complexity, color="red", linestyle="--",
                       label=f"Mean: {mean_complexity:.2f}")
            ax.legend()

            ax.set_ylabel("Spike Counts per Bin (Complexity)")
            ax.set_xlabel("Probability Density")
            ax.margins(y=0)
        else:
            ax.bar(complexities, pdf, alpha=0.7)
            # Add statistics
            ax.axvline(mean_complexity, color="red", linestyle="--",
                       label=f"Mean: {mean_complexity:.2f}")
            ax.legend()

            ax.set_xlabel("Spike Counts per Bin (Complexity)")
            ax.set_ylabel("Probability Density")

        # Add statistics
        stats_text = f"Variance: {var:.2f}\n"
        stats_text += f"Fano: {var / (mean + 1e-8):.2f}\n"
        stats_text += f"Skew: {sk:.2f}\n"
        stats_text += f"Kurtosis: {kurt:.2f}"

        ax.text(0.65, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat",
                          alpha=0.5))
        ax.legend(loc="upper left")

        ax.grid(True, alpha=0.3, axis=get_axis(orient))

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Population Spike Counts Distribution of " + str(n_neurons) + " neurons" + nucname)

        return ax

    @classmethod
    def plot_popcount_vs_time_in_ax(cls, ax, spiketimes_set, binsz=None, nucleus=None):
        """
        .. code-block:: text

            Population Spike Counts vs Time

            Spike Count
            ^
            |     ███ ████ ████ ███ ████ ███
            |    ███████████████████████████
            |   ████████████████████████████
            |----------------------------------  Mean
            |
            +--------------------------------------------------> Time (s)
            0        2        4        6        8        10

            Bars represent the number of spikes in the population
            within each time bin. The dashed line indicates the
            mean spike count across bins.

        Draws the Population Spike Counts over time on the given
        `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: integer or float; `0.01` [default]

        OPTIONAL parameters

        - :param nucleus: string; name of the nucleus
        - :return: object `ax` with Rate Distribution plotting done into it

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        n_neurons = len(spiketimes_set)

        spike_counts_per_bin, bin_centers = PopAct.count_allspikes_per_bin(spiketimes_set, binsz)

        complexities, complexity_counts, pdf = PopAct.compute_complexity_pdf(spike_counts_per_bin)

        mean_complexity = np.mean(spike_counts_per_bin)

        # Plot
        ax.bar(bin_centers, spike_counts_per_bin, width = binsz*0.9,
               alpha=0.7, color="steelblue", edgecolor="navy")
        # Add statistics
        ax.axhline(y=mean_complexity, color="red", linestyle="--",
                   label=f"Mean: {mean_complexity:.2f}")
        ax.legend()

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spike Count")

        ax.grid(True, alpha=0.3)

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title(f"Population Spike Counts (bin size={binsz}s) of " + str(n_neurons) + " neurons" + nucname)

        return ax
