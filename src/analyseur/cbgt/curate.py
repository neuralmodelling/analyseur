# ~/analyseur/cbgt/curate.py
#
# Documentation by Lungsi 17 Oct 2025
#
# This contains function for loading the files
#

import re

import numpy as np

from analyseur.cbgt.parameters import SpikeAnalysisParams

spikeanal = SpikeAnalysisParams()

def __extract_neuron_no(neuron_id):
    match = re.search(r'n(\d+)', neuron_id)
    return int(match.group(1))

def get_desired_spiketimes_subset(spiketimes_superset, neurons="all"):
    """
    Returns nested list of spike times (row-i for neuron ni, column-j for j-th spike time)
    and its associated yticks (list of neuron labels corresponding to the spike trains).

    :param spiketimes_superset: Dictionary returned using :py:class:`LoadSpikeTimes`
    :param neurons: [OPTIONAL] None or name of the nucleus (string)
    :return: nested_list, label_list
    """
    desired_spiketimes_subset = []
    yticks = []

    if neurons=="all":
        for nX, data in spiketimes_superset.items():
            desired_spiketimes_subset.append( list(data) )
            # yticks.append( _extract_neuron_no(nX) )
            yticks.append(nX)
    else: # neurons = range(a, b) or neurons = [1, 4, 5, 9]
        for i in neurons:
            neuron_id = "n" + str(i)
            desired_spiketimes_subset.append( list(spiketimes_superset[neuron_id]) )
            # yticks.append( _extract_neuron_no(neuron_id) )
            yticks.append(neuron_id)
    return desired_spiketimes_subset, yticks


def __get_valid_indices(indiv_spiketimes, window, sampling_rate, num_samples):
    """
    This function is essential because spiketimes are only recorded when spikes occur
    as opposed to recording for every time points.
    """
    indices = ((indiv_spiketimes - window[0]) * sampling_rate).astype(int)
    valid_indices = indices[(indices >= 0) & (indices < num_samples)]

    return valid_indices


def get_binary_spiketrains(spiketimes_superset, window=None, sampling_rate=None, neurons="all"):
    """
    Returns nested list of spike trains (row-i for neuron ni, column-j for j-th spike time)
    and its associated yticks (list of neuron labels corresponding to the spike trains).

    :param spiketimes_superset: Dictionary returned using :py:class:`LoadSpikeTimes`
    :param neurons: [OPTIONAL] "all" [default] or list: range(a, b) or [1, 4, 5, 9]
    :param window: Tuple (start, end)
    :param sampling_rate: number
    :return: nested_list, label_list, times_axis
    """
    # if window is None:
    #     window = spikeanal.window
    #
    # if sampling_rate is None:
    #     sampling_rate = 1 / spikeanal.sampling_period

    total_duration = window[1] - window[0]

    num_samples = int(total_duration * sampling_rate)
    time_axis = np.linspace(window[0], window[1], num_samples)
    num_neurons = len(spiketimes_superset)

    yticks = []

    if neurons=="all":
        spiketrains = np.zeros((num_neurons, num_samples))
        row = 0
        for nX, indiv_spiketimes in spiketimes_superset.items():
            index = __get_valid_indices(indiv_spiketimes, window, sampling_rate, num_samples)
            spiketrains[row, index] = 1.0
            yticks.append(nX)
            row += 1
    else: # neurons = range(a, b) or neurons = [1, 4, 5, 9]
        spiketrains = np.zeros((len(neurons), num_samples))
        row = 0
        for i in neurons:
            neuron_id = "n" + str(i)
            indiv_spiketimes = spiketimes_superset[neuron_id]
            index = __get_valid_indices(indiv_spiketimes, window, sampling_rate, num_samples)
            spiketrains[row, index] = 1.0  # row != i because neurons can be = [13, 14, 15 ,16 ...]
            yticks.append(neuron_id)

    return spiketrains, yticks, time_axis
