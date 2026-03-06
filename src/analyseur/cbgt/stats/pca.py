# ~/analyseur/cbgt/stat/pca.py
#
# Documentation by Lungsi 7 Nov 2025
#

import numpy as np

from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA as sklPCA
from sklearn.preprocessing import StandardScaler

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.parameters import SignalAnalysisParams

class PCA(object):
    """
    Computes PCA

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
        from analyseur.cbgt.stats.pca import PCA

    1.2. Load file and get spike times
    ```````````````````````````````````
    ::

        loadST = LoadSpikeTimes("spikes_GPi.csv")
        spiketimes_superset = loadST.get_spiketimes_superset()

    ---------
    2. Cases
    ---------

    2.1. Compute PCA with default values
    ````````````````````````````````````
    ::

        pca, pca_trajectory, activity_matrix, time_bins = PCA.compute(spiketimes_superset)


    2.2. Compute PCA with desired parameters
    ````````````````````````````````````````
    ::

        pca, pca_trajectory, activity_matrix, time_bins = PCA.compute(
                                                   spiketimes_superset,
                                                   binsz=0.01,
                                                   window=(0,10),
                                                   n_comp=0.95,
                                                   sigma_bins=2)

    2.3. Get participation ratio
    ````````````````````````````
    ::

        pr = PCA.participation_ratio(pca)

    2.4. Get eigen spectrum
    ```````````````````````
    ::

        eig_spec, cum_spec = PCA.eigenspectrum(pca)

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    __siganal = SignalAnalysisParams()

    @staticmethod
    def get_spike_activity_matrix(spiketimes_set, window, binsz):
        """
        .. math::

            t_0, t_1, t_2, ..., t_{n_\\text{bins}} \\in \\mathbb{R} \n
            A \\in \\mathbb{R}^{n_\\text{nuc} \\times n_\\text{bins}}

        where :math:`n_\\text{nuc}` is the number of neurons, :math:`n_\\text{bins}` is the number of time bins,  :math:`t_j = t_0 + j \\cdot \\Delta t` for :math:`j = 0,1, ..., n_\\text{bins}`, :math:`\\Delta t` is bin width, and :math:`a(i,t)` is the number of spikes of neuron :math:`i` in bin :math:`t`.

        Returns the activity matrix and time bins.

        Note that given the activity matrix :math:`A`

        - PCA(:math:`A`) measures **neuron correlation** structure
        - PCA(:math:`A^T`) measures **population activity trajectories over time**.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons="all")

        n_bins = int((window[1] - window[0]) / binsz)
        time_bins = np.linspace(window[0], window[1], n_bins + 1)

        activity = np.zeros((len(desired_spiketimes_subset), n_bins))

        # Activity Matrix
        for i, spikes in enumerate(desired_spiketimes_subset):
            # counts, _ = np.histogram(spikes, bins=time_bins)
            # activity[i, :] = counts

            if len(spikes) == 0:
                continue

            bin_idx = ((spikes - window[0]) / binsz).astype(int)
            bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bins)]

            np.add.at(activity[i], bin_idx, 1)

        activity = activity[::-1, :]  # reverse it so that neuron 0 is at the bottom

        return activity, time_bins

    @staticmethod
    def __compute_PCA(activity_matrix, n_comp, sigma_bins):
        """
        PCA pipeline-2

        - temporal smoothing to spike counts before PCA
        - Without smoothing, PCA tends to capture Poisson noise

        Given,
        .. math::

            A \\in \\mathbb{R}^{n_\\text{nuc} \\times n_\\text{bins}}

        where :math:`a(i,t)` is the number of spikes of neuron :math:`i` in bin :math:`t`. :math:`A` is a noisy matrix because :math:`var(a(i,t)) \\approx \\mathbb{E}[a(i,t)]`.

        **Step-1:** Temporal smoothing
        .. math::

            \\widetilde{a}(i,t) = \\sum_\\tau \\left[a(i,\\tau) \\cdot K(t-\\tau)\\right]

        where smoothing kernel K is
        .. math::

            K(\\Delta t) = \\frac{1}{\\sqrt{2\\pi\\sigma}} \\cdot e^\\frac{\\Delta t^2}{2\\sigma^2}

        **Step-2:** Mean subtraction
        .. math::

            [\\widetilde{a}(i,t)] = [\\widetilde{a}(i,t)] - \\frac{1}{n_\\text{nuc}} \\sum_{i=1}^{n_\\text{nuc}} \\widetilde{a}(i,t)

        removes global population activity.

        **Step-3:** Transpose activity matrix
        .. math::

            \\widetilde{A}^T \\in \\mathbb{R}^{n_\\text{bins} \\times n_\\text{nuc}} \n
            X = \\widetilde{A}^T

        for analyzing population activity trajectories over time where each row :math:`x(t,\\forall)` is the population activity vector at time :math:`t`.

        **Step-4:**
        Smallest $P$ chosen such that
        .. math::

            \\sum_{p=1}^{P}\\lambda_p \\ge q \\sum_{i=1}^{n_\\text{nuc}}\\lambda_i

        **Step-5:** PCA model
        .. math::

            C_{n_\\text{nuc} \\times n_\\text{nuc}} = \\frac{1}{(n_\\text{bins} - 1)} X^T X \n
            C \\vec{w_p} = \\lambda_p \\vec{w_p} \n
            W_{n_\\text{nuc} \\times P} = [\\vec{w_1}, ..., \\vec{w_P}]

        where :math:`\\vec{w}_p \\in \\mathbb{R}^{n_\\text{nuc}}` is a principal component direction across neurons and :math:`W` is the basis matrix.

        **Step-6:** PCA trajectory
        .. math::

            Z_{n_\\text{bins} \\times P} = X W \n

        where :math:`z(t,p) = \\sum_{i=1}^{n_\\text{nuc}}w(i,p)x(i,t)` and :math:`\\vec{z}(t)` is the population state in low-dimensional space.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # PCA pipeline-1
        # activity_matrix = activity_matrix.T
        # scaler = StandardScaler()
        # # scaled_activity = scaler.fit_transform(activity_matrix)
        # scaled_activity = activity_matrix - activity_matrix.mean(axis=0)
        # pca = sklPCA(n_components=n_comp)
        # pca_trajectory = pca.fit_transform(scaled_activity)

        # return scaler, pca, pca_trajectory

        # PCA pipeline-2 (temporal smoothing of spike counts)
        activity_matrix = gaussian_filter1d(activity_matrix,
                                            sigma=sigma_bins, # = 2 => 2 x binsz = 2 x 0.01 = 20 ms smoothing
                                            axis=1,)
        activity_matrix = activity_matrix - activity_matrix.mean(axis=1, keepdims=True)
        activity_matrix = activity_matrix.T
        pca = sklPCA(n_components=n_comp) # 0.95 variance
        pca_trajectory = pca.fit_transform(activity_matrix)

        return pca, pca_trajectory

    @classmethod
    def compute(cls, spiketimes_set, binsz=None, window=None, n_comp=None, sigma_bins=None):
        """
        Computes PCA given

        :param spiketimes_set: Dictionary returned using
        :param binsz: integer or float; `0.01` [default]
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param n_comp: integer or float; `0.95` [default]
        :param sigma_bins: integer or float; `2` [default]
        :return:

        * pca : `PCA model <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
        * pca_trajectory : numpy array, Projection matrix
        * activity_matrix : numpy array, Population activity
        * time_bins : numpy array,

        **Given,**

        .. math::

            A \\in \\mathbb{R}^{n_\\text{nuc} \\times n_\\text{bins}}

        where :math:`a(i,t)` is the number of spikes of neuron :math:`i` in bin :math:`t`. :math:`A` is a noisy matrix because :math:`var(a(i,t)) \\approx \\mathbb{E}[a(i,t)]`.

        **Step-1:** Temporal smoothing

        .. math::

            \\widetilde{a}(i,t) = \\sum_\\tau \\left[a(i,\\tau) \\cdot K(t-\\tau)\\right]

        where smoothing kernel K is

        .. math::

            K(\\Delta t) = \\frac{1}{\\sqrt{2\\pi\\sigma}} \\cdot e^\\frac{\\Delta t^2}{2\\sigma^2}

        **Step-2:** Mean subtraction

        .. math::

            [\\widetilde{a}(i,t)] = [\\widetilde{a}(i,t)] - \\frac{1}{n_\\text{nuc}} \\sum_{i=1}^{n_\\text{nuc}} \\widetilde{a}(i,t)

        removes global population activity.

        **Step-3:** Transpose activity matrix

        .. math::

            \\tilde{A}^T \\in \\mathbb{R}^{n_\\text{bins} \\times n_\\text{nuc}} \n
            X = \\tilde{A}^T

        for analyzing population activity trajectories over time where each row :math:`x(t,\\forall)` is the population activity vector at time :math:`t`.

        **Step-4:**

        Smallest $P$ chosen such that

        .. math::

            \\sum_{p=1}^{P}\\lambda_p \\ge q \\sum_{i=1}^{n_\\text{nuc}}\\lambda_i

        **Step-5:** PCA model

        .. math::

            C_{n_\\text{nuc} \\times n_\\text{nuc}} = \\frac{1}{(n_\\text{bins} - 1)} X^T X \n
            C \\vec{w_p} = \\lambda_p \\vec{w_p} \n
            W_{n_\\text{nuc} \\times P} = [\\vec{w_1}, ..., \\vec{w_P}]

        where :math:`\\vec{w}_p \\in \\mathbb{R}^{n_\\text{nuc}}` is a principal component direction across neurons and :math:`W` is the basis matrix.

        **Step-6:** PCA trajectory

        .. math::

            Z_{n_\\text{bins} \\times P} = X W \n

        where :math:`z(t,p) = \\sum_{i=1}^{n_\\text{nuc}}w(i,p)x(i,t)` and :math:`\\vec{z}(t)` is the population state in low-dimensional space.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        #============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        if n_comp is None:
            n_comp = 0.95 # 3

        if sigma_bins is None:
            sigma_bins = 2

        activity_matrix, time_bins = cls.get_spike_activity_matrix(spiketimes_set, window, binsz)

        time_bins_center = (time_bins[:-1] + time_bins[1:]) / 2

        [pca, pca_trajectory] = cls.__compute_PCA(activity_matrix, n_comp)

        return pca, pca_trajectory, activity_matrix, time_bins

    @staticmethod
    def participation_ratio(pca):
        """
        Given the computed `PCA model <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_

        .. math::

            D_\\text{PR} = \\frac{\\left(\\sum_{i=1}^P \\lambda_i\\right)}{\\sum_{i=1}^P \\lambda_i^2}

        measures how spread the variance is across components. Consequently, the complexity is

        .. math::

            \\text{complexity} = \\frac{D_\\text{PR}}{n_\\text{nuc}}

        **NOTE:**

        The `n_comp` in :py:meth:`.compute` can be either a **fixed components**, :math:`dim=` `n_comp` or a bound for **variance threshold**, :math:`\\sum_{i=1}^{P}\\lambda_i \\ge` `n_comp`. This estimate for dimensionality depends on arbitrary threshold.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        eigvals = pca.explained_variance_

        return (eigvals.sum() ** 2) / (np.sum(eigvals ** 2))

    @staticmethod
    def eigenspectrum(pca):
        """
        Given the computed `PCA model <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_

        .. math::

            \\tilde{\\lambda_p} = \\frac{\\lambda_p}{\\sum_{i}\\lambda_i}

        is the fraction of explained variance which is a function of component index :math:`p`. The cumulative spectrum is

        .. math::

            S_p = \\sum_{i=1}^p \\tilde{\\lambda_i}

        **COMMENT:**

        Plot of the eigenspectrum can reveal activity as low-dimensional or high-dimensional.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        eigvals = pca.explained_variance_
        var_ratio = eigvals / eigvals.sum()
        cumvar = np.cumsum(var_ratio)

        return var_ratio, cumvar


