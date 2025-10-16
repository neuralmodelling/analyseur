# ~/analyseur/cbgt/visual/popact.py
#
# Documentation by Lungsi 10 Oct 2025
#
# This contains function for Population Activity Heatmap
#

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# from ..loader import get_desired_spiketimes_superset
from analyseur.cbgt.loader import get_desired_spiketimes_subset

class PopAct(object):
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

      from analyseur.cbgt.visual.popact import PopAct

      my_pact = PopAct(spiketimes_superset)

    2. Population Activity Heatmap for the entire simulation window

    ::

      my_pact.plot()

    3. Population Activity Heatmap for desired window and bin size

    ::

      my_pact.plot(spike_trains, window=(0,50), binsz=1)
      my_pact.plot(spike_trains, window=(0,50), binsz=0.05)

    """
    def __init__(self, spiketimes_superset):
        self.spiketimes_superset = spiketimes_superset

    def _compute_PCA(self, activity_matrix, n_comp=3):
        scaler = StandardScaler()
        scaled_activity = scaler.fit_transform(activity_matrix)
        pca = PCA(n_components=n_comp)
        pca_trajectory = pca.fit_transform(scaled_activity)

        return scaler, pca, pca_trajectory

    def _compute_activity(self, desired_spiketimes_superset, binsz=50, window=(0, 10000)):
        bins = np.arange(window[0], window[1] + binsz, binsz)

        # Activity Matrix
        activity = np.zeros((len(desired_spiketimes_superset), len(bins) - 1))
        for i, spikes in enumerate(desired_spiketimes_superset):
            counts, _ = np.histogram(spikes, bins=bins)
            activity[i] = counts
        activity = activity[::-1, :]  # reverse it so that neuron 0 is at the bottom

        return activity, bins

    def plot(self, binsz=50, window=(0, 10000), nucleus=None, show=True):
        """
        Displays the Population Activity Heatmap of the given spike times and returns the plot figure (to save if necessary).

        :param spiketimes_superset: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
        :param binsz: defines the number of equal-width bins in the range [default: 50]
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)
        :param show: boolean [default: True]
        :return: object `matplotlib.pyplot.imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_
    
        * `window` controls the binning range as well as the spike counting window
        * CBGT simulation was done in milliseconds so window `(0, 10000)` signifies time 0 ms to 10,000 ms (or 10 s)
        * `popactivity` gives a spatio-temporal pattern across neurons

        """
        # Set binsz and window as the instance attributes
        self.binsz = binsz
        self.window = window

        # Get and set desired_spiketimes_superset as instance attribute
        [self.desired_spiketimes_superset, _] = get_desired_spiketimes_subset(self.spiketimes_superset)
        # NOTE: desired_spiketimes_superset as nested list and not numpy array because
        # each neuron may have variable length of spike times
        self.n_neurons = len(self.desired_spiketimes_superset)

        # Compute activities in activity matrix and set the results as instance attributes
        [self.activity_matrix, self.bins] = \
            self._compute_activity(self.desired_spiketimes_superset, binsz=binsz, window=window)

        t_axis = self.bins[:-1] + binsz / 2

        # Plot
        plt.figure(1)
        plt.imshow(self.activity_matrix, aspect="auto", cmap="hot",
                   # extent=[window[0], window[1], n_neurons, 0] # if neuron 0 is at the top by default
                   extent=[window[0], window[1], 0, self.n_neurons])
        plt.colorbar(label="Spike Count per Bin")

        plt.ylabel("neurons")
        plt.xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        plt.title("Population Activity Heatmap of " + str(self.n_neurons) + " neurons" + nucname)

        if show:
            plt.show()

        return plt

    def plot_pcatraj(self, n_comp=3, nucleus=None, show=True):
        """
        PCA Trajectory of population activity
        """
        [self.scaler, self.pca, self.pca_traj] = self._compute_PCA(self.activity_matrix, n_comp=n_comp)
        self.t_points = np.linspace(self.window[0], self.window[1], self.pca_traj.shape[0])

        # Plot
        fig = plt.figure(2)

        # PC1 vs PC2
        ax1 = fig.add_subplot(2,2,3)

        scatter = ax1.scatter(self.pca_traj[:,0], self.pca_traj[:,1],
                              c=self.t_points, cmap="viridis", s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax1, label="Time (ms)")

        ax1.set_xlabel("PC1 ({:.1f}%".format(self.pca.explained_variance_ratio_[0]*100))
        ax1.set_ylabel("PC2 ({:.1f}%".format(self.pca.explained_variance_ratio_[1] * 100))
        ax1.set_title("PCA Trajectory: PC1 vs PC2")

        # PC1 vs Time
        ax2 = fig.add_subplot(2,1,1)

        ax2.plot(self.t_points, self.pca_traj[:,0], linewidth=2)
        ax2.grid(True, alpha=0.3)

        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("PC1")
        ax2.set_title("PC1 Over Time")

        # Variance explained?
        ax3 = fig.add_subplot(2,2,4)

        components = range(1, min(10, len(self.pca.explained_variance_ratio_)) + 1)
        ax3.bar(components, self.pca.explained_variance_ratio_[:len(components)])

        ax3.set_xlabel("Principal Component")
        ax3.set_ylabel("Variance Explained")
        ax3.set_title("PCA Variance Explained")

        nucname = "" if nucleus is None else " in " + nucleus
        fig.suptitle(' Principal Component Analysis of ' + str(self.n_neurons) + " neurons" + nucname, fontsize=14)

        plt.tight_layout()

        if show:
            plt.show()

    def analytics(self):
        return {
            "activity": self.activity_matrix,
            "time_points": self.t_points,
            "scaler": self.scaler,
            "pca": self.pca, # if n_comp=0.9 => dimensionality = self.pca.n_components_
            "pca_trajectory": self.pca_traj,
            "explained_variance": self.pca.explained_variance_ratio_,
        }
