# ~/analyseur/cbgt/stat/utilities.py
#
# Documentation by Lungsi 3 Oct 2025
#
# This contains function for loading the files
#

import numpy as np

def compute_grand_mean(all_neuron_stat=None):
    stat_array = np.zeros(len(all_neuron_stat))

    i = 0
    for val in all_neuron_stat.values():
        stat_array[i] = val
        i += 1

    return np.mean(stat_array)