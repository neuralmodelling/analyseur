# ~/analyseur/cbgt/stat/sync.py
#
# Documentation by Lungsi 16 Oct 2025
#
# This contains function for loading the files
#

import numpy as np

from analyseur.cbgt.curate import get_desired_spiketimes_subset

class Synchrony(object):

    @staticmethod
    def __get_spikearray_and_window(spiketimes_superset, window, neurons="all"):
        [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)
        n_neurons = len(desired_spiketimes_subset)

        spike_arrays = [np.array(spike_times) for spike_times in desired_spiketimes_subset]

        if window is None:
            all_spikes = np.concatenate(spike_arrays)
            window = (np.min(all_spikes), np.max(all_spikes))

        return spike_arrays, window

    @staticmethod
    def __compute_for_basic(freq_matrix):
        # Remove time bins with no activity
        valid_bins = np.sum(freq_matrix, axis=0) > 0
        freq_matrix = freq_matrix[:, valid_bins]

        if freq_matrix.size == 0:
            return 0.0

        # Compute A: variance over time of mean frequency across neurons
        mean_across_neurons = np.mean(freq_matrix, axis=0)  # μ_k(freq_k(t)) for each t
        A = np.var(mean_across_neurons)  # var_t of above

        # Compute B: mean over time of variance across neurons
        variance_across_neurons = np.var(freq_matrix, axis=0)  # var_k(freq_k(t)) for each t
        B = np.mean(variance_across_neurons)  # μ_t of above

        if B == 0:
            if A == 0:
                S = 0.0  # perfect synchrony edge case
            else:
                S = np.inf  # infinite synchrony (theoretical)
        else:
            S = np.sqrt(A / B)

        return S

    @staticmethod
    def __compute_fano(pop_spike_count_matrix, bins_option="non_zero"):
        if bins_option == "non_zero":
            # Remove bins with no activity
            non_zero_mask = pop_spike_count_matrix > 0
            S_t = pop_spike_count_matrix[non_zero_mask]
        else:  # include all bins
            S_t = pop_spike_count_matrix

        if len(S_t) > 0:
            colmn_wise_sums = np.sum(S_t, axis=0)
            variance_S = np.var(colmn_wise_sums)
            mean_S = np.mean(colmn_wise_sums)

            if mean_S == 0:
                fanofactor = 0.0
            else:
                fanofactor = variance_S / mean_S
        else:
            fanofactor = 0.0  # variance_S = 0.0, mean_S = 0.0

        return fanofactor, S_t


    @classmethod
    def compute_basic(cls, spiketimes_superset, binsz=0.05, window=(0, 10)):
        """
        Returns the basic measure of synchrony of spiking from all neurons.

        :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :param binsz: 0.05 [default]
        :param window: Tuple in the form `(start_time, end_time)`; (0, 10) [default]
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

            Sync = \\sqrt{\\frac{A}{B}} = \\sqrt{\\frac{var\\left(\\left[\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{t}}\\right)\\right]_{\\forall{i}}\\right)}{\\mu\\left(\\left[var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}}

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        [spike_arrays, window] = cls.__get_spikearray_and_window(spiketimes_superset, window, neurons="all")
        n_neurons = len(spike_arrays)

        time_bins = np.arange(window[0], window[1] + binsz, binsz)
        n_bins = len(time_bins) - 1

        freq_matrix = np.zeros((n_neurons, n_bins))

        for i, spike_times in enumerate(spike_arrays):
            counts, _ = np.histogram(spike_times, bins=time_bins)
            freq_matrix[i, :] = counts / binsz

        return cls.__compute_for_basic(freq_matrix)


    @classmethod
    def compute_basic_slide(cls, spiketimes_superset, binsz=0.05, window=(0, 10), windowsz=0.5):
        [spike_arrays, window] = cls.__get_spikearray_and_window(spiketimes_superset, window, neurons="all")
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
    def compute_fano_factor(cls, spiketimes_superset, binsz=0.05, window=(0, 10), bins_option="all_bins"):
        """
        Returns the Fano factor as a measure of synchrony of spiking from all neurons.

        :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :param binsz: 0.05 [default]
        :param window: Tuple in the form `(start_time, end_time)`; (0, 10) [default]
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
        [spike_arrays, window] = cls.__get_spikearray_and_window(spiketimes_superset, window, neurons="all")

        time_bins = np.arange(window[0], window[1] + binsz, binsz)
        n_bins = len(time_bins) - 1

        time_bins_center = (time_bins[:-1] + time_bins[1:]) / 2

        # Population spike count array
        S_t = np.zeros(n_bins)

        for spike_times in spike_arrays:
            counts, _ = np.histogram(spike_times, bins=time_bins)
            S_t += counts

        [fanofactor, S_t] = cls.__compute_fano(S_t, bins_option=bins_option)

        return fanofactor, S_t, time_bins_center



