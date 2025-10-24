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
    | :py:meth:`.compute`          | - :param all_neurons_spiketimes: Dictionary returned  |
    |                              | using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`  |
    +------------------------------+-------------------------------------------------------+
    | :py:meth:`.mean_freqs`       | - :param all_neurons_isi: Dictionary returned         |
    |                              | using :py:meth:`.compute`                             |
    +------------------------------+-------------------------------------------------------+
    | :py:meth:`.grand_mean_freq`  | - :param all_neurons_isi: Dictionary returned         |
    |                              | using :py:meth:`.compute`                             |
    +------------------------------+-------------------------------------------------------+

    ----

    """

    @classmethod
    def compute(cls, all_neurons_spiketimes=None):
        """
        Returns the interspike interval for all individual neurons.

        :param all_neurons_spiketimes: Dictionary returned using :class:`~analyseur/cbgt/loader.LoadSpikeTimes`
        :return: dictionary of individual neurons whose values are their respective interspike interval

        **Formula**

        .. table:: Formula
        ========================================================================================= ======================================================
          Definitions                                                                             Interpretation
        ========================================================================================= ======================================================
         total neurons, :math:`n_{nuc}`                                                            total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                   i-th neuron in the pool of :math:`n_{Nuc}` neurons
         total spikes, :math:`n_{spk}^{(i)}`                                                       total number of spikes (spike times) by i-th neuron
         interspike interval, :math:`isi_{k}^{(i)}`                                                k-th absolute interval between successive spike times
         :math:`\\overrightarrow{ISI}^{(i)} = [isi_k^{(i)}]_{\\forall{k \\in [1, n_{spk}^{(i)})}}`       array of all interspike intervals of i-th neuron
         :math:`\\vec{I} = [\\overrightarrow{ISI}^{(i)}]_{\\forall{i \\in [1, n_{nuc}]}}`                array of array interspike intervals of all neurons
        ========================================================================================= ======================================================

        ----

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

        **Formula**

        .. table:: Formula_mean_freqs_1.1
        ========================================================================================= ======================================================
          Definitions                                                                             Interpretation
        ========================================================================================= ======================================================
         total neurons, :math:`n_{nuc}`                                                            total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                   i-th neuron in the pool of :math:`n_{Nuc}` neurons
         total spikes, :math:`n_{spk}^{(i)}`                                                       total number of spikes (spike times) by i-th neuron
         interspike interval, :math:`isi_{k}^{(i)}`                                                k-th absolute interval between successive spike times
         :math:`\\overrightarrow{ISI}^{(i)} = [isi_k^{(i)}]_{\\forall{k \\in [1, n_{spk}^{(i)})}}`       array of all interspike intervals of i-th neuron
         :math:`\\vec{I} = [\\overrightarrow{ISI}^{(i)}]_{\\forall{i \\in [1, n_{nuc}]}}`                array of array interspike intervals of all neurons
        ========================================================================================= ======================================================

        Then, the mean spiking frequency of i-th neuron is

        .. math::

            \\overline{f^{(i)}} = \\frac{1}{(n_{spk}^{(i)} - 1)} \\sum_{j=1}^{(n_{spk}^{(i)} - 1)}\\frac{1}{isi_{j}^{(i)}}

        We therefore get

        .. table:: Formula_mean_freqs_1.2
        ========================================================================== ======================================================
          Definitions                                                                Interpretation
        ========================================================================== ======================================================
         :math:`\\overline{f^{(i)}}`                                                 mean spiking frequency of i-th neuron
         :math:`\\vec{F} = [\\overline{f^{(i)}}]_{\\forall{i \\in [1, n_{nuc}]}}`       array of mean frequencies of all neurons
        ========================================================================== ======================================================

        ----

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

        **Formula**
        
        .. table:: Formula
        ========================================================================== ======================================================
          Definitions                                                                Interpretation
        ========================================================================== ======================================================
         total neurons, :math:`n_{Nuc}`                                              total number of neurons in the Nucleus
         neuron index, :math:`i`                                                     i-th neuron in the pool of :math:`n_{Nuc}` neurons
         mean frequency, :math:`\\overline{f^{(i)}}`                                 mean spiking frequency of i-th neuron
         :math:`\\vec{F} = [\\overline{f^{(i)}}]_{\\forall{i \\in [1, n_{nuc}]}}`       array of mean frequencies of all (:math:`n_{Nuc}`) neurons
         grand mean frequency, :math:`\\overline{f}`                                 grand or global mean spiking frequency
        ========================================================================== ======================================================

        ----

        """
        all_neurons_mean_freq = cls.mean_freqs(all_neurons_isi)
        return cgm(all_neurons_mean_freq)