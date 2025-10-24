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
        Returns the coefficient of variation for all individual neurons.

        :param all_neurons_isi: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :return: dictionary of individual neurons whose values are their respective coefficient of variation value

        **Formula**

        .. table:: Formula_computeCV_1.2
        ========================================================================================= ======================================================
          Definitions                                                                             Interpretation
        ========================================================================================= ======================================================
         total neurons, :math:`n_{nuc}`                                                            total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                   i-th neuron in the pool of :math:`n_{Nuc}` neurons
         total spikes, :math:`n_{spk}^{(i)}`                                                       total number of spikes (spike times) by i-th neuron
         interspike interval, :math:`isi_{k}^{(i)}`                                                k-th absolute interval between successive spike times
         :math:`\\overrightarrow{ISI}^{(i)} = [isi_k^{(i)}]_{\\forall{k \\in [1, n_{spk}^{(i)})}}`       array of all interspike intervals of i-th neuron
         :math:`\\vec{I} = \\left[\\overrightarrow{ISI}^{(i)}\\right]_{\\forall{i \\in [1, n_{nuc}]}}`        array of array interspike intervals of all neurons
        ========================================================================================= ======================================================

        Then, the coefficient of variation of the i-th neuron is

        .. math::

            cv^{(i)} = \\frac{\\sigma\\left(\\overrightarrow{ISI}^{(i)}\\right)}{\\mu\\left(\\overrightarrow{ISI}^{(i)}\\right)}

        where, :math:`\\sigma(\\cdot)` is the `standard deviation function <https://numpy.org/doc/stable/reference/generated/numpy.std.html>`_ over the given dimension and
        :math:`\\mu(\\cdot)` is the `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_ over the given dimension.

        We therefore get

        .. table:: Formula_computeCV_1.2
        ================================================================================== ======================================================
          Definitions                                                                       Interpretation
        ================================================================================== ======================================================
         :math:`cv^{(i)}`                                                                   coefficient of variation of i-th neuron
         :math:`\\overrightarrow{CV} = \\left[cv^{(i)}\\right]_{\\forall{i \\in [1, n_{nuc}]}}`       array of coefficient of variation of all neurons
        ================================================================================== ======================================================

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

        :param all_neurons_isi: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :return: dictionary of individual neurons whose values are their respective local coefficient of variation value

        **Formula**

        .. table:: Formula_computeCV2_1.2
        ========================================================================================= ======================================================
          Definitions                                                                             Interpretation
        ========================================================================================= ======================================================
         total neurons, :math:`n_{nuc}`                                                            total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                   i-th neuron in the pool of :math:`n_{Nuc}` neurons
         total spikes, :math:`n_{spk}^{(i)}`                                                       total number of spikes (spike times) by i-th neuron
         interspike interval, :math:`isi_{k}^{(i)}`                                                k-th absolute interval between successive spike times
         :math:`\\overrightarrow{ISI}^{(i)} = [isi_k^{(i)}]_{\\forall{k \\in [1, n_{spk}^{(i)})}}`       array of all interspike intervals of i-th neuron
         :math:`\\vec{I} = \\left[\\overrightarrow{ISI}^{(i)}\\right]_{\\forall{i \\in [1, n_{nuc}]}}`        array of array interspike intervals of all neurons
        ========================================================================================= ======================================================

        Then, the coefficient of variation of the i-th neuron is

        .. math::

            cv_2^{(i)} = \\mu\\left(2\\frac{\\left[\\left|isi^{(i)}_k - isi^{(i)}_{k-1}\\right|\\right]_{\\forall{k \\in [1, n_{spk}^{(i)})}}}{\\left[isi^{(i)}_k + isi^{(i)}_{k-1}\\right]_{\\forall{k \\in [1, n_{spk}^{(i)})}}}\\right)

        where, :math:`\\mu(\\cdot)` is the `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_ over the given dimension.

        We therefore get

        .. table:: Formula_computeCV2_1.2
        ======================================================================================= ======================================================
          Definitions                                                                            Interpretation
        ======================================================================================= ======================================================
         :math:`cv_2^{(i)}`                                                                      local coefficient of variation of the i-th neuron
         :math:`\\overrightarrow{CV_2} = \\left[cv_2^{(i)}\\right]_{\\forall{i \\in [1, n_{nuc}]}}`         array of local coefficient of variation of all neurons
        ======================================================================================= ======================================================

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

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

        :param all_neurons_isi: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :return: dictionary of individual neurons whose values are their respective local variation value

        **Formula**

        .. table:: Formula_computeLV_1.2
        ========================================================================================= ======================================================
          Definitions                                                                             Interpretation
        ========================================================================================= ======================================================
         total neurons, :math:`n_{nuc}`                                                            total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                   i-th neuron in the pool of :math:`n_{Nuc}` neurons
         total spikes, :math:`n_{spk}^{(i)}`                                                       total number of spikes (spike times) by i-th neuron
         interspike interval, :math:`isi_{k}^{(i)}`                                                k-th absolute interval between successive spike times
         :math:`\\overrightarrow{ISI}^{(i)} = [isi_k^{(i)}]_{\\forall{k \\in [1, n_{spk}^{(i)})}}`       array of all interspike intervals of i-th neuron
         :math:`\\vec{I} = \\left[\\overrightarrow{ISI}^{(i)}\\right]_{\\forall{i \\in [1, n_{nuc}]}}`        array of array interspike intervals of all neurons
        ========================================================================================= ======================================================

        Then, the local variation of the i-th neuron is

        .. math::

            lv^{(i)} = \\mu\\left(3\\frac{\\left[\\left(isi^{(i)}_k - isi^{(i)}_{k-1}\\right)^2\\right]_{\\forall{k \\in [1, n_{spk}^{(i)})}}}{\\left[\\left(isi^{(i)}_k + isi^{(i)}_{k-1}\\right)^2\\right]_{\\forall{k \\in [1, n_{spk}^{(i)})}}}\\right)

        where, :math:`\\mu(\\cdot)` is the `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_ over the given dimension.

        We therefore get

        .. table:: Formula_computeLV_1.2
        ======================================================================================= ======================================================
          Definitions                                                                            Interpretation
        ======================================================================================= ======================================================
         :math:`lv^{(i)}`                                                                        local variation of the i-th neuron
         :math:`\\overrightarrow{LV} = \\left[lv^{(i)}\\right]_{\\forall{i \\in [1, n_{nuc}]}}`         array of local coefficient of variation of all neurons
        ======================================================================================= ======================================================

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

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
