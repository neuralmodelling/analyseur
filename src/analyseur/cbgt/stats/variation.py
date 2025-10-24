# ~/analyseur/cbgt/stat/variation.py
#
# Documentation by Lungsi 2 Oct 2025
#
# This contains function for loading the files
#

import numpy as np

# from compute_shared import compute_grand_mean as cgm
from analyseur.cbgt.stats.compute_shared import compute_grand_mean as cgm

class Variations(object):
    """
    Computes variation or dispersion in the data

    +------------------------+-----------------------------------------------+
    | Methods                | Argument                                      |
    +========================+===============================================+
    | :py:meth:`.computeCV`  | - :param all_neurons_isi: Dictionary returned |
    |                        | using :py:meth:`.compute`                     |
    +------------------------+-----------------------------------------------+
    | :py:meth:`.computeCV2` | - :param all_neurons_isi: Dictionary returned |
    |                        | using :py:meth:`.compute`                     |
    +------------------------+-----------------------------------------------+
    | :py:meth:`.computeLV`  | - :param all_neurons_isi: Dictionary returned |
    |                        | using :py:meth:`.compute`                     |
    +------------------------+-----------------------------------------------+
    | :py:meth:`grandCV`     | - :param all_neurons_isi: Dictionary returned |
    |                        | using :py:meth:`.compute`                     |
    +------------------------+-----------------------------------------------+
    | :py:meth:`grandCV2`    | - :param all_neurons_isi: Dictionary returned |
    |                        | using :py:meth:`.compute`                     |
    +------------------------+-----------------------------------------------+
    | :py:meth:`grandLV`     | - :param all_neurons_isi: Dictionary returned |
    |                        | using :py:meth:`.compute`                     |
    +------------------------+-----------------------------------------------+
    
    """

    @classmethod
    def computeCV(cls, all_neurons_isi=None):
        """
        Returns

        :param all_neurons_isi: Dictionary returned using :meth:`~analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :return:

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

        Then, the coefficient of variation of the i-th neuron is

        .. math::

            \\overline{cv^{(i)}} = \\frac{\\sigma(\\overrightarrow{ISI}^{(i)})}{\\mu(\\overrightarrow{ISI}^{(i)})}

        We therefore get

        .. table:: Formula_mean_freqs_1.2
        ========================================================================== ======================================================
          Definitions                                                                Interpretation
        ========================================================================== ======================================================
         :math:`\\overline{f^{(i)}}`                                                 mean spiking frequency of i-th neuron
         :math:`\\vec{F} = [\\overline{f^{(i)}}]_{\\forall{i \\in [1, n_{nuc}]}}`       array of mean frequencies of all neurons
        ========================================================================== ======================================================

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        all_CV = {}

        for n_id, isi in all_neurons_isi.items():
            if len(isi) == 0:
                all_CV[n_id] = np.zeros(1)
            else:
                all_CV[n_id] = np.std(isi) / np.mean(isi)

        return all_CV

    @classmethod
    def computeCV2(cls, all_neurons_isi=None):
        """
        Returns

        :param all_neurons_isi:
        :return:
        """
        all_CV2 = {}

        for n_id, isi in all_neurons_isi.items():
            if len(isi) == 0:
                all_CV2[n_id] = np.zeros(1)
            else:
                abs_diff_over_sum = np.abs( np.diff(isi) / (isi[1:] + isi[:-1]) )
                all_CV2[n_id] = np.mean(2 * abs_diff_over_sum)

        return all_CV2

    @classmethod
    def computeLV(cls, all_neurons_isi=None):
        """
        Returns

        :param all_neurons_isi:
        :return:
        """
        all_LV = {}

        for n_id, isi in all_neurons_isi.items():
            if len(isi) == 0:
                all_LV[n_id] = np.zeros(1)
            else:
                sq_diff_over_sum = np.square(np.diff(isi)) / np.square((isi[1:] + isi[:-1]))
                all_LV[n_id] = np.mean(3 * sq_diff_over_sum)

        return all_LV

    @classmethod
    def grandCV(cls, all_neurons_isi=None):
        """

        :param all_neurons_isi:
        :return:
        """
        all_neurons_CV = cls.computeCV(all_neurons_isi)
        return cgm(all_neurons_CV)

    @classmethod
    def grandCV2(cls, all_neurons_isi=None):
        """

        :param all_neurons_isi:
        :return:
        """
        all_neurons_CV2 = cls.computeCV2(all_neurons_isi)
        return cgm(all_neurons_CV2)

    @classmethod
    def grandLV(cls, all_neurons_isi=None):
        """

        :param all_neurons_isi:
        :return:
        """
        all_neurons_LV = cls.computeLV(all_neurons_isi)
        return cgm(all_neurons_LV)
