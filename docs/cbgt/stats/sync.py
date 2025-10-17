# ~/analyseur/cbgt/stat/sync.py
#
# Documentation by Lungsi 16 Oct 2025
#
# This contains function for loading the files
#

import numpy as np

from analyseur.cbgt.curate import get_binary_spiketrains

class Synchrony(object):

    def __init__(self, spiketimes_superset, window=(0, 10000), sampling_rate=10, neurons="all"):
        self.spiketimes_superset = spiketimes_superset
        self.window = window
        self.spiketrains = get_binary_spiketrains(spiketimes_superset, window=window,
                                                  sampling_rate=sampling_rate, neurons=neurons)

    @classmethod
    def basic(cls, all_neurons_spiketimes=None):
        interspike_intervals = {}

        for n_id, spiketimes in all_neurons_spiketimes.items():
            interspike_intervals[n_id] = np.diff(spiketimes)


