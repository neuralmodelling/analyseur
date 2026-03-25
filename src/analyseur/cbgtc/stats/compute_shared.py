# ~/analyseur/cbgtc/stat/compute_shared.py
#
# Documentation by Lungsi 3 Oct 2025
#
# This contains function for loading the files
#

import numpy as np

def compute_grand_mean(all_neuron_stat=None):
    """
    Returns the grand/global mean of a given statistics of all the neurons in a nucleus.

    :param all_neuron_stat:
    :return: a number

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    stat_array = np.zeros(len(all_neuron_stat))

    i = 0
    for val in all_neuron_stat.values():
        stat_array[i] = val
        i += 1

    return np.mean(stat_array)

def autocorr(x):
    """
    Performs autocorrelation

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2:]  # keep non-negative lags
    return corr / corr[0]         # normalize
