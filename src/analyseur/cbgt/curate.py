# ~/analyseur/cbgt/curate.py
#
# Documentation by Lungsi 17 Oct 2025
#

import re

import numpy as np

from analyseur.cbgt.parameters import SpikeAnalysisParams

spikeanal = SpikeAnalysisParams()

# ==========================================
# get_desired_spiketimes_subset
# ==========================================

def __extract_neuron_no(neuron_id):
    match = re.search(r'n(\d+)', neuron_id)
    return int(match.group(1))

def get_desired_spiketimes_subset(spiketimes_superset, neurons=None):
    """
    =============================
    get_desired_spiketimes_subset
    =============================

    Returns nested list of spike times (row-i for neuron ni, column-j for j-th spike time)
    and its associated yticks (list of neuron labels corresponding to the spike trains).

    :param spiketimes_superset: Dictionary returned using :class:`~analyseur.cbgt.loader.LoadSpikeTimes`
    :param neurons: [OPTIONAL] `"all"` (default)
    :return: 2-tuple; nested_list and label_list

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    # ============== DEFAULT Parameters ==============
    if neurons is None:
        neurons = "all"

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

# ==========================================
# get_binary_spiketrains
# ==========================================

def get_binary_spiketrains(spiketimes_superset, window=None, sampling_rate=None, neurons=None):
    """
    ======================
    get_binary_spiketrains
    ======================

    Returns nested list of spike trains (row-i for neuron ni, column-j for j-th spike time)
    and its associated yticks (list of neuron labels corresponding to the spike trains).

    :param spiketimes_superset: Dictionary returned using :class:`~analyseur.cbgt.loader.LoadSpikeTimes`
    :param neurons: [OPTIONAL] `"all"` [default] or list: range(a, b) or [1, 4, 5, 9]
    :param window: Tuple (start, end), `(0, 10)` [default]
    :param sampling_rate: `10000` [default]
    :return: 3-tuple; nested_list, label_list and times_axis

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    # ============== DEFAULT Parameters ==============
    if sampling_rate is None:
        sampling_rate = 1 / spikeanal.sampling_period

    if window is None:
        window = spikeanal.window

    if neurons is None:
        neurons = "all"

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
