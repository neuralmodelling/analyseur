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
            variance_S = np.var(S_t)
            mean_S = np.mean(S_t)

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
        Returns the basic measure of synchrony of spiking from all individual neurons.

        :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :param binsz: 0.05 [default]
        :param window: Tuple in the form `(start_time, end_time)`; (0, 10) [default]
        :return: dictionary of individual neurons whose values are their respective coefficient of variation value

        **Formula**

        .. table:: Formula
        ================================= ======================================================
          Definitions                       Interpretation
        ================================= ======================================================
         total neurons, :math:`n_{Nuc}`     total number of neurons in the Nucleus
         neuron index, :math:`i`            i-th neuron in the pool of :math:`n_{Nuc}` neurons
         frequency, :math:`f^{(i)}`         mean spiking frequency of the i-th neuron
        ================================= ======================================================

        .. math::

            F = \\overset{\\begin{matrix}t_0 & \\quad\\quad\\quad & t_1 & & & &\\ldots & & & t_{n_{Nuc}}\\end{matrix}}
            {\\underset{
                \\begin{matrix}
                    \\quad\\quad\\quad\\quad\\quad\\uparrow & \\quad\\quad\\quad\\quad & \\uparrow & &\\ldots & & & \\uparrow \n
                    \\quad\\quad\\quad\\quad\\mu_{t_0} & \\quad\\quad\\quad\\quad & \\mu_{t_1} & &\\ldots & & & \\mu_{t_{n_{Nuc}}} & \\rightarrow var_{\\forall{t}}
                \end{matrix}}
           {\\begin{bmatrix}
             f^{(1)}(t_0) & f^{(1)}(t_1) & \\ldots & f^{(1)}(t_{n_{Nuc}}) \n
             f^{(2)}(t_0) & f^{(2)}(t_1) & \\ldots & f^{(2)}(t_{n_{Nuc}}) \n
             \\vdots & \\vdots & \\ldots & \\vdots \n
             f^{(i)}(t_0) & f^{(i)}(t_1) & \\ldots & f^{(i)}(t_{n_{Nuc}}) \n
             \\vdots & \\vdots & \\ldots & \\vdots \n
             f^{(n_{Nuc})}(t_0) & f^{(n_{Nuc})}(t_1) & \\ldots & f^{(n_{Nuc})}(t_{n_{Nuc}})
            \\end{bmatrix}
            }}


        .. math::

            M = \\begin{bmatrix}
                    1 & 4 & 7 \n
                    2 & 5 & 8 \n
                    3 & 6 & 9
                \\end{bmatrix}

        Then, synchrony is measured as

        .. math::

            Sync = \\sqrt{\\frac{var\\left(\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{t}}\\right)\\right)}{\\mu\\left(var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right)}}

        where, :math:`var(\\cdot)` is the `variance function <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_ over the given dimension and
        :math:`\\mu(\\cdot)` is the `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_ over the given dimension.

        NOTE: The array :math:`\\vec{F}` is obtained by calling :py:meth:`.mean_freqs`

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



