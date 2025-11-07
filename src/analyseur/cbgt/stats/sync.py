# ~/analyseur/cbgt/stat/sync.py
#
# Documentation by Lungsi 16 Oct 2025
#

import numpy as np

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.parameters import SignalAnalysisParams

class Synchrony(object):
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
        n_neurons = len(desired_spiketimes_subset)

        time_bins = np.arange(window[0], window[1] + binsz, binsz)
        n_bins = len(time_bins)

        spike_matrix = np.zeros((n_neurons, n_bins))

        # Fill the spike matrix
        for i, indiv_spiketimes in enumerate(desired_spiketimes_subset):
            for spiketime in indiv_spiketimes:
                if window[0] <= spiketime < window[1]:
                    j = int((spiketime - window[0]) // binsz)
                    spike_matrix[i, j] = 1

        return spike_matrix, time_bins

    @staticmethod
    def __get_count_rate_matrix(spiketimes_set, window, binsz):
        [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons="all")
        n_neurons = len(desired_spiketimes_subset)

        time_bins = np.arange(window[0], window[1] + binsz, binsz)
        n_bins = len(time_bins) - 1

        count_matrix = np.zeros((n_neurons, n_bins))
        rate_matrix = np.zeros((n_neurons, n_bins))

        # Fill the count and rate matrix
        for i, indiv_spiketimes in enumerate(desired_spiketimes_subset):
            counts, _ = np.histogram(indiv_spiketimes, bins=time_bins)
            count_matrix[i, :] = counts
            rate_matrix[i, :] = counts / binsz

        return count_matrix, rate_matrix, time_bins

    @staticmethod
    def __get_spikearray_and_window(spiketimes_superset, window, neurons="all"):
        [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)
        n_neurons = len(desired_spiketimes_subset)

        spike_arrays = [np.array(spike_times) for spike_times in desired_spiketimes_subset]

        if window is None:
            all_spikes = np.concatenate(spike_arrays)
            window = (np.min(all_spikes), np.max(all_spikes))

        return spike_arrays, window

    # @staticmethod
    # def __compute_for_basic(freq_matrix):
    #     # Remove time bins with no activity
    #     valid_bins = np.sum(freq_matrix, axis=0) > 0
    #     freq_matrix = freq_matrix[:, valid_bins]
    #
    #     if freq_matrix.size == 0:
    #         return 0.0
    #
    #     # Compute A: variance over time of mean frequency across neurons
    #     mean_across_neurons = np.mean(freq_matrix, axis=0)  # μ_k(freq_k(t)) for each t
    #     A = np.var(mean_across_neurons)  # var_t of above
    #
    #     # Compute B: mean over time of variance across neurons
    #     variance_across_neurons = np.var(freq_matrix, axis=0)  # var_k(freq_k(t)) for each t
    #     B = np.mean(variance_across_neurons)  # μ_t of above
    #
    #     if B == 0:
    #         if A == 0:
    #             S = 0.0  # perfect synchrony edge case
    #         else:
    #             S = np.inf  # infinite synchrony (theoretical)
    #     else:
    #         S = np.sqrt(A / B)
    #
    #     return S

    @staticmethod
    def __compute_for_basic(freq_matrix):
        if len(freq_matrix) > 0:
            # Mean of rates across neurons for all t
            colmn_wise_means = np.mean(freq_matrix, axis=0)
            # Variance of rates across neurons for all t
            colmn_wise_vars = np.var(freq_matrix, axis=0)

            variance_F = np.var(colmn_wise_means)
            mean_F = np.mean(colmn_wise_vars)

            if mean_F == 0:
                s_sync = 0.0
            else:
                s_sync = variance_F / mean_F
        else:
            s_sync = 0.0  # variance_F = 0.0, mean_F = 0.0

        return s_sync, colmn_wise_means, colmn_wise_vars


    @staticmethod
    def __compute_fano(pop_spike_count_matrix):
        if len(pop_spike_count_matrix) > 0:
            # Sum of spike counts across neurons for all t
            colmn_wise_sums = np.sum(pop_spike_count_matrix, axis=0)

            variance_S = np.var(colmn_wise_sums)
            mean_S = np.mean(colmn_wise_sums)

            if mean_S == 0:
                fanofactor = 0.0
            else:
                fanofactor = variance_S / mean_S
        else:
            fanofactor = 0.0  # variance_S = 0.0, mean_S = 0.0

        return fanofactor, colmn_wise_sums


    @classmethod
    def compute_basic(cls, spiketimes_set, binsz=None, window=None):
        """
        Returns the basic measure of synchrony of spiking from all neurons.

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: integer or float; `0.01` [default]
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :return: a number

        **Formula**

        .. table:: Formula
        ====================================================================================================================== ===============================================================
          Definitions                                                                                                             Interpretation
        ====================================================================================================================== ===============================================================
         total neurons, :math:`n_{Nuc}`                                                                                           total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                                                  i-th neuron in the pool of :math:`n_{Nuc}` neurons
         frequency, :math:`f^{(i)}(t)`                                                                                            frequency of the i-th neuron at time :math:`t`
         frequency matrix, :math:`F = \\left[f(a,b) = f^{(a)}(b)\\right]_{\\forall{a \\in [1, n_{Nuc}], b \\in [t_0, t_T]}}`          frequencies of all (:math:`n_{Nuc}`) neurons for all times
        ====================================================================================================================== ===============================================================

        Let the :math:`var(\\cdot)`, `variance function <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_ and
        the :math:`\\mu(\\cdot)`, `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_
        be implemented as shown

        .. math::

            F = \\overset{\\begin{matrix}t_0 & \\quad\\quad & t_1 & & & &\\ldots & & & t_T\\end{matrix}}
                {\\underset{
                    \\begin{matrix}
                        \\quad\\quad\\uparrow & \\quad\\quad\\quad & \\uparrow & \\quad &\\ldots & & & \\uparrow \n
                        \\quad\\mu_{t_0} & \\quad\\quad\\quad & \\mu_{t_1} & \\quad &\\ldots & & & \\mu_{t_T} & \\rightarrow var_{\\forall{t}}
                    \\end{matrix}}
               {\\begin{bmatrix}
                 f^{(1)}(t_0) & f^{(1)}(t_1) & \\ldots & f^{(1)}(t_T) \n
                 f^{(2)}(t_0) & f^{(2)}(t_1) & \\ldots & f^{(2)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(i)}(t_0) & f^{(i)}(t_1) & \\ldots & f^{(i)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(n_{Nuc})}(t_0) & f^{(n_{Nuc})}(t_1) & \\ldots & f^{(n_{Nuc})}(t_T)
                \\end{bmatrix}
                }}

        Then, we define

        .. math::

            A \\triangleq var\\left(\\begin{bmatrix}
                                       \\mu_{t_0} & \\mu_{t_1} & \\ldots & \\mu_{t_T}
                                     \\end{bmatrix}\\right) = var_{\\forall{t}}

        Implementing the `variance function <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_ and
        the `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_ as shown below

        .. math::

            F = \\overset{\\begin{matrix}t_0 & \\quad\\quad & t_1 & & & &\\ldots & & & t_T\\end{matrix}}
                {\\underset{
                    \\begin{matrix}
                        \\quad\\quad\\uparrow & \\quad\\quad\\quad & \\uparrow & \\quad &\\ldots & & & \\uparrow \n
                        \\quad var_{t_0} & \\quad\\quad\\quad & var_{t_1} & \\quad &\\ldots & & & var_{t_T} & \\rightarrow \\mu_{\\forall{t}}
                    \\end{matrix}}
               {\\begin{bmatrix}
                 f^{(1)}(t_0) & f^{(1)}(t_1) & \\ldots & f^{(1)}(t_T) \n
                 f^{(2)}(t_0) & f^{(2)}(t_1) & \\ldots & f^{(2)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(i)}(t_0) & f^{(i)}(t_1) & \\ldots & f^{(i)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(n_{Nuc})}(t_0) & f^{(n_{Nuc})}(t_1) & \\ldots & f^{(n_{Nuc})}(t_T)
                \\end{bmatrix}
                }}

        we make another definition

        .. math::

            B \\triangleq \\mu\\left(\\begin{bmatrix}
                                         var_{t_0} & var_{t_1} & \\ldots & var_{t_T}
                                     \\end{bmatrix}\\right) = \\mu_{\\forall{t}}

        Then, synchrony is measured as

        .. math::

            Sync = \\sqrt{\\frac{A}{B}} = \\sqrt{\\frac{var\\left(\\left[\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}{\\mu\\left(\\left[var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}}

        NOTE: This method is a simple histogram-based approach that uses fixed bins.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # # ============== DEFAULT Parameters ==============
        # if window is None:
        #     window = cls.__siganal.window
        #
        # if binsz is None:
        #     binsz = cls.__siganal.binsz_100perbin
        #
        # [spike_arrays, window] = cls.__get_spikearray_and_window(spiketimes_set, window, neurons="all")
        # n_neurons = len(spike_arrays)
        #
        # time_bins = np.arange(window[0], window[1] + binsz, binsz)
        # n_bins = len(time_bins) - 1
        #
        # freq_matrix = np.zeros((n_neurons, n_bins))
        #
        # for i, spike_times in enumerate(spike_arrays):
        #     counts, _ = np.histogram(spike_times, bins=time_bins)
        #     freq_matrix[i, :] = counts / binsz
        #
        # return cls.__compute_for_basic(freq_matrix)
        #============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        _, freq_matrix, time_bins = cls.__get_count_rate_matrix(spiketimes_set, window, binsz)

        time_bins_center = (time_bins[:-1] + time_bins[1:]) / 2

        [s_sync, _, _] = cls.__compute_for_basic(freq_matrix)

        return s_sync, freq_matrix, time_bins_center


    @classmethod
    def compute_basic_slide(cls, spiketimes_set, binsz=None, window=None, windowsz=None):
        """
        Returns the basic measure of synchrony of spiking from all neurons.

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: integer or float; `0.01` [default]
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param windowsz: integer or float, sliding window size; `0.5` [default]
        :return: a number

        NOTE: The computation is done on a sliding window resulting in a smoother frequency estimation otherwise
        this is the same as :py:meth:`.compute_basic`. Therefore, unlike the simple histogram-based approach the use
        of overlapping windows (sliding windows) results in a smoother frequency estimation.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        if windowsz is None:
            windowsz = 0.5

        [spike_arrays, window] = cls.__get_spikearray_and_window(spiketimes_set, window, neurons="all")
        n_neurons = len(spike_arrays)

        eval_times = np.arange(window[0] + windowsz/2, window[1] - windowsz, binsz)
        n_times = len(eval_times)

        freq_matrix = np.zeros((n_neurons, n_times))

        for i, spike_times in enumerate(spike_arrays):
            for t, t_center in enumerate(eval_times):
                start = t_center - windowsz/2
                stop = t_center + windowsz/2

                spikes_in_slidingwindow = np.sum((spike_times >= start) & (spike_times <= stop))
                freq_matrix[i, t] = spikes_in_slidingwindow / windowsz

        return cls.__compute_for_basic(freq_matrix)


    @classmethod
    def compute_fano_factor(cls, spiketimes_set, binsz=None, window=None):
        """
        Returns the Fano factor as a measure of synchrony of spiking from all neurons.

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: integer or float; `0.01` [default]
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :return: a number

        **Formula**

        .. table:: Formula
        ====================================================================================================================== =====================================================================
          Definitions                                                                                                             Interpretation
        ====================================================================================================================== =====================================================================
         total neurons, :math:`n_{Nuc}`                                                                                           total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                                                  i-th neuron in the pool of :math:`n_{Nuc}` neurons
         spike count, :math:`p^{(i)}(t)`                                                                                          spike count of the i-th neuron at time :math:`t`
         count matrix, :math:`P = \\left[p(a,b) = p^{(a)}(b)\\right]_{\\forall{a \\in [1, n_{Nuc}], b \\in [t_0, t_T]}}`                spike counts of all (:math:`n_{Nuc}`) neurons for all times
        ====================================================================================================================== =====================================================================

        Let the :math:`var(\\cdot)`, `variance function <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_ and
        the :math:`\\mu(\\cdot)`, `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_
        be implemented as shown

        .. math::

            P = \\overset{\\begin{matrix}t_0 & \\quad\\quad & t_1 & & & &\\ldots & & & t_T\\end{matrix}}
                {\\underset{
                    \\begin{matrix}
                        \\quad\\quad\\uparrow & \\quad\\quad\\quad & \\uparrow & \\quad &\\ldots & & & \\uparrow \n
                        \\quad\\pi_{t_0} & \\quad\\quad\\quad & \\pi_{t_1} & \\quad &\\ldots & & & \\pi_{t_T} & \\rightarrow var_{\\forall{t}} \n
                        \\quad\\pi_{t_0} & \\quad\\quad\\quad & \\pi_{t_1} & \\quad &\\ldots & & & \\pi_{t_T} & \\rightarrow \\mu_{\\forall{t}}
                    \\end{matrix}}
               {\\begin{bmatrix}
                 p^{(1)}(t_0) & p^{(1)}(t_1) & \\ldots & p^{(1)}(t_T) \n
                 p^{(2)}(t_0) & p^{(2)}(t_1) & \\ldots & p^{(2)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 p^{(i)}(t_0) & p^{(i)}(t_1) & \\ldots & p^{(i)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 p^{(n_{Nuc})}(t_0) & p^{(n_{Nuc})}(t_1) & \\ldots & p^{(n_{Nuc})}(t_T)
                \\end{bmatrix}
                }}

        We define

        .. math::

            A &\\triangleq var\\left(\\begin{bmatrix}
                                       \\pi_{t_0} & \\pi_{t_1} & \\ldots & \\pi_{t_T}
                                     \\end{bmatrix}\\right) = var_{\\forall{t}} \n
            B &\\triangleq \\mu\\left(\\begin{bmatrix}
                                       \\pi_{t_0} & \\pi_{t_1} & \\ldots & \\pi_{t_T}
                                     \\end{bmatrix}\\right) = \\mu_{\\forall{t}}

        Then, synchrony is measured as

        .. math::

            F_{Sync} = \\frac{A}{B} = \\frac{var\\left(\\left[\\sum_{\\forall{i}}p^{(i)}(t)\\right]_{\\forall{t}}\\right)}{\\mu\\left(\\left[\\sum_{\\forall{i}}p^{(i)}(t)\\right]_{\\forall{t}}\\right)}

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        #============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        # spike_matrix, time_bins = cls.__get_spike_matrix(spiketimes_set, window, binsz)
        spike_matrix, _, time_bins = cls.__get_count_rate_matrix(spiketimes_set, window, binsz)

        time_bins_center = (time_bins[:-1] + time_bins[1:]) / 2

        [fanofactor, _] = cls.__compute_fano(spike_matrix)

        return fanofactor, spike_matrix, time_bins_center



