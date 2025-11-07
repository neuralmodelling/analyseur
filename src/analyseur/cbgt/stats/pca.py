# ~/analyseur/cbgt/stat/pca.py
#
# Documentation by Lungsi 7 Nov 2025
#

import numpy as np

from sklearn.decomposition import PCA as sklPCA
from sklearn.preprocessing import StandardScaler

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.parameters import SignalAnalysisParams

class PCA(object):
    """
    Computes measures of synchrony among the neurons with given spike times

    +----------------------------------+----------------------------------------------------------------------------------------------------------+
    | Methods                          | Argument                                                                                                 |
    +==================================+==========================================================================================================+
    | :py:meth:`.compute_basic`        | - `spiketimes_superset`: see :class:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`      |
    |                                  | - OPTIONAL: `binsz` (0.01 [default]), `window` ((0, 10) [default])                                       |
    +----------------------------------+----------------------------------------------------------------------------------------------------------+
    | :py:meth:`.compute_basic_slide`  | - `spiketimes_superset`: see :class:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`      |
    |                                  | - OPTIONAL: `binsz` (0.01 [default]), `window` ((0, 10) [default]), `windowsz` (0.5 [default])           |
    +----------------------------------+----------------------------------------------------------------------------------------------------------+
    | :py:meth:`.compute_fano_factor`  | - `spiketimes_superset`: see :class:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`      |
    |                                  | - OPTIONAL: `binsz` (0.01 [default]), `window` ((0, 10) [default])                                       |
    +----------------------------------+----------------------------------------------------------------------------------------------------------+

    =========
    Use Cases
    =========

    ------------------
    1. Pre-requisites
    ------------------

    1.1. Import Modules
    ````````````````````
    ::

        from analyseur.cbgt.loader import LoadSpikeTimes
        from analyseur.cbgt.stats.sync import Synchrony

    1.2. Load file and get spike times
    ```````````````````````````````````
    ::

        loadST = LoadSpikeTimes("spikes_GPi.csv")
        spiketimes_superset = loadST.get_spiketimes_superset()

    ---------
    2. Cases
    ---------

    2.1. Compute basic measure of spike times synchrony (for all neurons)
    ``````````````````````````````````````````````````````````````````````
    ::

        B = Synchrony.compute_basic(spiketimes_superset)

    This returns the value for

    .. math::

       Sync = \\sqrt{\\frac{var\\left(\\left[\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{t}}\\right)\\right]_{\\forall{i}}\\right)}{\\mu\\left(\\left[var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}}

    2.2. Compute the basic measure of synchrony on a smoother frequency estimation
    ```````````````````````````````````````````````````````````````````````````````
    ::

        S = Synchrony.compute_basic_slide(spiketimes_superset)

    This returns the value for

    .. math::

       Sync = \\sqrt{\\frac{var\\left(\\left[\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{t}}\\right)\\right]_{\\forall{i}}\\right)}{\\mu\\left(\\left[var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}}

    2.3. Compute Fano factor as a metric for measuring spike times synchrony (for all neurons)
    ```````````````````````````````````````````````````````````````````````````````````````````
    ::

        Fs = Synchrony.compute_fano_factor(spiketimes_superset)

    This returns the value for

    .. math::

        F_{Sync} = \\frac{var\\left(\\left[\\sum_{\\forall{i}}p^{(i)}(t)\\right]_{\\forall{t}}\\right)}{\\mu\\left(\\left[\\sum_{\\forall{i}}p^{(i)}(t)\\right]_{\\forall{t}}\\right)}

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    __siganal = SignalAnalysisParams()

    @staticmethod
    def __get_spike_matrix(spiketimes_set, window, binsz):
        [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons="all")

        time_bins = np.arange(window[0], window[1] + binsz, binsz)
        n_bins = len(time_bins) - 1

        activity = np.zeros((len(desired_spiketimes_subset), n_bins))

        # Activity Matrix
        for i, spikes in enumerate(desired_spiketimes_subset):
            counts, _ = np.histogram(spikes, bins=time_bins)
            activity[i] = counts
        activity = activity[::-1, :]  # reverse it so that neuron 0 is at the bottom

        return activity, time_bins

    @staticmethod
    def __compute_PCA(activity_matrix, n_comp):
        scaler = StandardScaler()
        scaled_activity = scaler.fit_transform(activity_matrix)
        pca = sklPCA(n_components=n_comp)
        pca_trajectory = pca.fit_transform(scaled_activity)

        return scaler, pca, pca_trajectory

    @classmethod
    def compute(cls, spiketimes_set, binsz=None, window=None, n_comp=None):
        #============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        if n_comp is None:
            n_comp = 3

        activity_matrix, time_bins = cls.__get_spike_matrix(spiketimes_set, window, binsz)

        time_bins_center = (time_bins[:-1] + time_bins[1:]) / 2

        [scaler, pca, pca_trajectory] = cls.__compute_PCA(activity_matrix, n_comp)

        return scaler, pca, pca_trajectory, activity_matrix, time_bins



