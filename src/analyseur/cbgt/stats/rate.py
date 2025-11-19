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
    def mean_rate(cls, spiketimes_set=None, window=None, binsz=None, neurons="all"):
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        _, rate_matrix, time_bins = cls.get_count_rate_matrix(spiketimes_set=spiketimes_set,
                                                              window=window, binsz=binsz,
                                                              neurons=neurons)
        # Calculate mean of rate of all the neurons across time
        mu_rate_vec = rate_matrix.mean(axis=0)

        return mu_rate_vec, time_bins