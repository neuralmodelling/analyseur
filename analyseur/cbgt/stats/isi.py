# ~/analyseur/cbgt/stat/isi.py
#
# Documentation by Lungsi 2 Oct 2025
#
# This contains function for loading the files
#

import numpy as np

from utilities import compute_grand_mean as cgm

class InterSpikeInterval(object):
    """This class

    * sdf
    * sdf

    """

    @classmethod
    def compute(cls, all_neurons_spiketrains=None):
        interspike_intervals = {}

        for n_id, spiketimes in all_neurons_spiketrains.items():
            interspike_intervals[n_id] = np.diff(spiketimes)

        return interspike_intervals

    @classmethod
    def mean_freqs(cls, all_neurons_isi=None):
        mean_spiking_freq = {}

        for n_id, isi in all_neurons_isi.items():
            # n_spikes = len(isi) + 1
            mean_spiking_freq[n_id] = (1/len(isi)) * np.sum(1/isi)

        return mean_spiking_freq

    @classmethod
    def grand_mean_freq(cls, all_neurons_isi=None):
        all_neurons_mean_freq = cls.mean_freqs(all_neurons_isi)
        return cgm(all_neurons_mean_freq)