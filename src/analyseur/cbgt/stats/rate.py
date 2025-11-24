# ~/analyseur/cbgt/stats/rate.py
#
# Documentation by Lungsi 17 Nov 2025
#

import numpy as np

from analyseur.cbgt.curate import get_desired_spiketimes_subset
# from .compute_shared import compute_grand_mean as cgm
from analyseur.cbgt.stats.compute_shared import compute_grand_mean as cgm
from analyseur.cbgt.parameters import SignalAnalysisParams

class Rate(object):
    __siganal = SignalAnalysisParams()

    @classmethod
    def get_count_rate_matrix(cls, spiketimes_set=None, window=None, binsz=None, neurons="all"):
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        [desired_spiketimes_subset, _] = get_desired_spiketimes_subset(spiketimes_set, neurons=neurons)
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

    @classmethod
    def mean_rate(cls, spiketimes_set=None, window=None, binsz=None, neurons="all", across=None):
        """
        Returns the basic measure of synchrony of spiking from all neurons.

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: integer or float; `0.01` [default]
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]

        :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

        :param across: "neurons" or "times"
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

        **Formula: Mean Rate across = "neurons"**

        .. math::

            F = \\overset{\\begin{matrix}t_0 & \\quad\\quad & t_1 & & & &\\ldots & & & t_T\\end{matrix}}
                {\\underset{
                    \\begin{matrix}
                        \\quad\\quad\\downarrow & \\quad\\quad\\quad & \\downarrow & \\quad &\\ldots & & & \\downarrow \n
                        \\quad\\mu_{t_0} & \\quad\\quad\\quad & \\mu_{t_1} & \\quad &\\ldots & & & \\mu_{t_T} & \\quad \\quad
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

        **Formula: Mean Rate across = "times"**

        .. math::

            F = \\overset{\\begin{matrix}t_0 & \\quad\\quad & t_1 & & & &\\ldots & & & t_T\\end{matrix}}
               {\\begin{bmatrix}
                 f^{(1)}(t_0) & f^{(1)}(t_1) & \\ldots & f^{(1)}(t_T) \n
                 f^{(2)}(t_0) & f^{(2)}(t_1) & \\ldots & f^{(2)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(i)}(t_0) & f^{(i)}(t_1) & \\ldots & f^{(i)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(n_{Nuc})}(t_0) & f^{(n_{Nuc})}(t_1) & \\ldots & f^{(n_{Nuc})}(t_T)
                \\end{bmatrix}
                }}
                \\begin{matrix}
                 \\rightarrow \\mu^{(1)} \n
                 \\rightarrow \\mu^{(2)} \n
                 \\vdots \n
                 \\rightarrow \\mu^{(i)} \n
                 \\vdots \n
                 \\rightarrow \\mu^{(n_{Nuc})}
                \\end{matrix}

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        _, rate_matrix, time_bins = cls.get_count_rate_matrix(spiketimes_set=spiketimes_set,
                                                              window=window, binsz=binsz,
                                                              neurons=neurons)

        # Calculate mean of rate of ...
        if across is None:
            raise AttributeError("across MUST BE 'times' or 'neurons'.")
        elif across=="times":
            # Calculate mean of rate of all the neurons across time
            mu_rate_vec = rate_matrix.mean(axis=1)
        elif across=="neurons":
            # Calculate mean of rate of all the neurons across time
            mu_rate_vec = rate_matrix.mean(axis=0)

        return mu_rate_vec, time_bins