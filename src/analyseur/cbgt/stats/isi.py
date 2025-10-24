# ~/analyseur/cbgt/stat/isi.py
#
# Documentation by Lungsi 2 Oct 2025
#
# This contains function for loading the files
#

import numpy as np

# from .compute_shared import compute_grand_mean as cgm
from analyseur.cbgt.stats.compute_shared import compute_grand_mean as cgm

class InterSpikeInterval(object):
    """
    Computes interspike intervals for the given spike times

    +------------------------------+-------------------------------------------------------+
    | Methods                      | Argument                                              |
    +==============================+=======================================================+
    | :py:meth:`.compute`          | - :param all_neurons_spiketimes: Dictionary returned |
    |                              | using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`  |
    +------------------------------+-------------------------------------------------------+
    | :py:meth:`.mean_freqs`       | - :param all_neurons_isi: Dictionary returned         |
    |                              | using :py:meth:`.compute`                             |
    +------------------------------+-------------------------------------------------------+
    | :py:meth:`.grand_mean_freq`  | - :param all_neurons_isi: Dictionary returned         |
    |                              | using :py:meth:`.compute`                             |
    +------------------------------+-------------------------------------------------------+

    """

    @classmethod
    def compute(cls, all_neurons_spiketimes=None):
        """
        Returns the interspike interval for all individual neurons.

        :param all_neurons_spiketimes: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
        :return: dictionary of individual neurons whose values are their respective interspike interval

        .. table:: Formula

        ================================================== ======================================================
          Definitions                                        Interpretation
        ================================================== ======================================================
          total neurons, :math:`n_{Nuc}`                     total number of neurons in the Nucleus
          neuron index, :math:`i`                            i-th neuron in the pool of :math:`n_{Nuc}` neurons
          time index, :math:`k`                              integer of indices of the spike times
          total spikes, :math:`n_{spk}^{(i)}`                total number of spikes (spike times) by i-th neuron
          interspike interval, :math:`isi_{k}^{(i)}`         absolute interval between successive spike times
          :math:`isi_{k}^{(i)} = t_{k+1}^{(i)} - t_{k}^{(i)} = \\lvert t_{k}^{(i)} - t_{k+1}^{(i)} \\rvert`
          array of interspike interval, :math:`ISI^{(i)}`    interspike interval between all spike times of i-th neuron
        ================================================== ======================================================

        """
        interspike_intervals = {}

        for n_id, spiketimes in all_neurons_spiketimes.items():
            interspike_intervals[n_id] = np.diff(spiketimes)

        return interspike_intervals

    @classmethod
    def mean_freqs(cls, all_neurons_isi=None):
        """
        Returns the mean frequencies for all individual neurons.

        :param all_neurons_isi: Dictionary returned using :py:meth:`.compute`
        :return: dictionary of individual neurons whose values are their respective mean frequencies

        .. table:: Formula

        ================================================== ======================================================
          Definitions                                        Interpretation
        ================================================== ======================================================
          total neurons, :math:`n_{Nuc}`                     total number of neurons in the Nucleus
          neuron index, :math:`i`                            i-th neuron in the pool of :math:`n_{Nuc}` neurons
          total spikes, :math:`n_{spk}^{(i)}`                total number of spikes (spike times) by i-th neuron
          :math:`\\overrightarrow{isi}^{(i)}`                array of interspike intervals of i-th neuron
          array of interspike interval, :math:`\\vec{I}`    interspike interval between all spike times of i-th neuron
          array of interspike interval, :math:`\\vec{I}`    interspike interval between all spike times of i-th neuron
          interspike interval, :math:`isi_{k}^{(i)}`         absolute interval between successive spike times
          mean frequency, :math:`\\overline{f^{(i)}}`        mean spiking frequency of i-th neuron
        ================================================== ======================================================

        .. math::

            \\overline{f^{(i)}} = \\frac{1}{(n_{spk}^{(i)} - 1)} \\sum_{j=1}^{(n_{spk}^{(i)} - 1)}\\frac{1}{isi_{j}^{(i)}}\\
            \\overrightarrow{ISI^i} = [\\overrightarrow{ISI^{(i)}}]
            
            \\overrightarrow{ISI}^{(i)} = [isi_k^{(i)}]_{\\forall{k \\in [1, n_{spk}^{(i)})}}

        .. table::

        =========================================================================================   ============================================
        Definitions                                                                                 Interpretations
        =========================================================================================   ============================================
        :math:`\\overrightarrow{ISI}^{(i)} = [isi_k^{(i)}]_{\\forall{k \\in [1, n_{spk}^{(i)})}}`   array of interspike intervals of i-th neuron
        =========================================================================================   ============================================

        """
        mean_spiking_freq = {}

        for n_id, isi in all_neurons_isi.items():
            # n_spikes = len(isi) + 1
            if len(isi) == 0:
                mean_spiking_freq[n_id] = 0
            else:
                mean_spiking_freq[n_id] = (1 / len(isi)) * np.sum(1 / isi)

        return mean_spiking_freq

    @classmethod
    def grand_mean_freq(cls, all_neurons_isi=None):
        """
        Returns the grand mean frequency which is the mean of mean frequencies of all the neurons
        
        :param all_neurons_isi: Dictionary returned using :py:meth:`.compute`
        :return: a number
        
        .. table:: Formula

        ================================================== ======================================================
          Definitions                                        Interpretation
        ================================================== ======================================================
          total neurons, :math:`n_{Nuc}`                     total number of neurons in the Nucleus
          neuron index, :math:`i`                            i-th neuron in the pool of :math:`n_{Nuc}` neurons
          total spikes, :math:`n_{spk}^{(i)}`                total number of spikes (spike times) by i-th neuron
          mean frequency, :math:`\\overline{f^{(i)}}`        mean spiking frequency of i-th neuron
          grand mean frequency, :math:`\\overline{f}`        grand or global mean spiking frequency of all :math:`n_{Nuc}`
        ================================================== ======================================================
        
        """
        all_neurons_mean_freq = cls.mean_freqs(all_neurons_isi)
        return cgm(all_neurons_mean_freq)