# ~/analyseur/cbgt/curate.py
#
# Documentation by Lungsi 17 Oct 2025
#

import re

import numpy as np

from analyseur.cbgt.parameters import SignalAnalysisParams

siganal = SignalAnalysisParams()

# ==========================================
# get_desired_spiketimes_subset
# ==========================================

def __extract_neuron_no(neuron_id):
    match = re.search(r'n(\d+)', neuron_id)
    return int(match.group(1))

def get_desired_spiketimes_subset(spiketimes_set, neurons=None):
    """
    Returns nested list of spike times (row-i for neuron ni, column-j for j-th spike time)
    and its associated yticks (list of neuron labels corresponding to the spike trains).

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

        - `"all"` means subset = superset
        - `N` (a scalar) means subset of first N neurons in the superset
        - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :return: 2-tuple; nested_list and label_list

    =========
    Use Cases
    =========

    ------------------
    1. Pre-requisites
    ------------------

    1.1. Import Modules
    ```````````````````
    ::

        from analyseur.cbgt.loader import LoadSpikeTimes
        from  analyseur.cbgt.curate import get_desired_spiketimes_subset

    1.2. Load file and get spike times
    ```````````````````````````````````
    ::

        loadST = LoadSpikeTimes("spikes_GPi.csv")
        spiketimes_superset = loadST.get_spiketimes_superset()

    ---------
    2. Cases
    ---------

    2.1. Convert spike times set to nested list of spike times
    ``````````````````````````````````````````````````````````
    ::

        [spiketimes_superlist, _] = get_desired_spiketimes_subset(spiketimes_superset)

    2.2. Get nested list of spike times for desired neurons; specific range
    ```````````````````````````````````````````````````````````````````````
    ::

        neurons = range(30, 62)  # neuron id from "n30" to "n62"
        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=neurons)

        [spiketimes_nestedlist, neuron_labels] = get_desired_spiketimes_subset(spiketimes_subset,
                                                                               neurons="all")

    Alternatively,
    ::

        neurons = range(30, 62)  # neuron id from "n30" to "n62"
        [spiketimes_nestedlist, neuron_labels] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)

    2.3. Get nested list of spike times for desired neurons; specific list
    ``````````````````````````````````````````````````````````````````````
    ::

        neurons = [1, 2, 3, 6, 9, 10, 11, 21, 31]  # neuron ids "n1", "n2", ..., "n21", "n31"
        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=neurons)

        [spiketimes_nestedlist, neuron_labels] = get_desired_spiketimes_subset(spiketimes_subset,
                                                                               neurons="all")

    Alternatively,
    ::

        neurons = [1, 2, 3, 6, 9, 10, 11, 21, 31]  # neuron ids "n1", "n2", ..., "n21", "n31"
        [spiketimes_nestedlist, neuron_labels] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)

    2.4. Get nested list of spike times for desired neurons; first N neurons
    ````````````````````````````````````````````````````````````````````````
    ::

        N = 50  # first 50 neurons regardless of the neuron id
        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=N)

        [spiketimes_nestedlist, neuron_labels] = get_desired_spiketimes_subset(spiketimes_subset,
                                                                               neurons="all")

    Alternatively,
    ::

        N = 50  # first 50 neurons regardless of the neuron id
        neuron_ids = dict(list(spiketimes_superset.items())[:neurons]).keys()
        neurons = [int(item[1:]) for item in neuron_ids]

        [spiketimes_nestedlist, neuron_labels] = get_desired_spiketimes_subset(spiketimes_superset, neurons=neurons)

    Comments
    ````````
    - In 2.2 to 2.4 the alternative method passes the mother set (superset) of spike times.
    - For 2.2 and 2.3 cases the method choice will depend on the use scenario.
    - But for 2.4 I prefer the first method (not alternative method) because it is more intuitive.

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    if neurons is None:
        neurons = "all"

    desired_spiketimes_subset = []
    yticks = []

    if neurons=="all":
        for nX, data in spiketimes_set.items():
            desired_spiketimes_subset.append( list(data) )
            # yticks.append( _extract_neuron_no(nX) )
            yticks.append(nX)
    else: # neurons = range(a, b) or neurons = [1, 4, 5, 9]
        for i in neurons:
            neuron_id = "n" + str(i)
            desired_spiketimes_subset.append( list(spiketimes_set[neuron_id]) )
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

def get_binary_spiketrains(spiketimes_set, window=None, sampling_rate=None, neurons=None):
    """
    Returns nested list of spike trains (row-i for neuron ni, column-j for j-th spike time)
    and its associated yticks (list of neuron labels corresponding to the spike trains).

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

        - `"all"` means subset = superset
        - `N` (a scalar) means subset of first N neurons in the superset
        - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
    :param sampling_rate: `1000/dt = 10000` Hz [default]; sampling_rate ∊ (0, 10000)
    :return: 3-tuple; nested_list, label_list and times_axis

    =========
    Use Cases
    =========

    ------------------
    1. Pre-requisites
    ------------------

    1.1. Import Modules
    ```````````````````
    ::

        from analyseur.cbgt.loader import LoadSpikeTimes
        from  analyseur.cbgt.curate import get_binary_spiketrains

    1.2. Load file and get spike times
    ```````````````````````````````````
    ::

        loadST = LoadSpikeTimes("spikes_GPi.csv")
        spiketimes_superset = loadST.get_spiketimes_superset()

    ---------
    2. Cases
    ---------

    2.1. Convert spike times set to nested list of binary spike trains
    ``````````````````````````````````````````````````````````````````
    ::

        [spiketrains_superlist, neuron_labels, time_axis] = get_binary_spiketrains(spiketimes_superset)

    2.2. Get nested list of binary spike trains for desired neurons; specific range
    ```````````````````````````````````````````````````````````````````````````````
    ::

        neurons = range(30, 62)  # neuron id from "n30" to "n62"
        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=neurons)

        [spiketrains_nestedlist, neuron_labels, time_axis] = get_binary_spiketrains(spiketimes_subset,
                                                                                    neurons="all")

    Alternatively,
    ::

        neurons = range(30, 62)  # neuron id from "n30" to "n62"
        [spiketrains_nestedlist, neuron_labels, time_axis] = get_binary_spiketrains(spiketimes_superset,
                                                                                    neurons=neurons)

    2.3. Get nested list of binary spike trains for desired neurons; specific list
    ``````````````````````````````````````````````````````````````````````````````
    ::

        neurons = [1, 2, 3, 6, 9, 10, 11, 21, 31]  # neuron ids "n1", "n2", ..., "n21", "n31"
        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=neurons)

        [spiketrains_nestedlist, neuron_labels, time_axis] = get_binary_spiketrains(spiketimes_subset,
                                                                                    neurons="all")

    Alternatively,
    ::

        neurons = [1, 2, 3, 6, 9, 10, 11, 21, 31]  # neuron ids "n1", "n2", ..., "n21", "n31"
        [spiketrains_nestedlist, neuron_labels, time_axis] = get_binary_spiketrains(spiketimes_superset,
                                                                                    neurons=neurons)

    2.4. Get nested list of binary spike trains for desired neurons; first N neurons
    ````````````````````````````````````````````````````````````````````````````````
    ::

        N = 50  # first 50 neurons regardless of the neuron id
        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=N)

        [spiketrains_nestedlist, neuron_labels, time_axis] = get_binary_spiketrains(spiketimes_subset,
                                                                                    neurons="all")

    Alternatively,
    ::

        N = 50  # first 50 neurons regardless of the neuron id
        neuron_ids = dict(list(spiketimes_superset.items())[:neurons]).keys()
        neurons = [int(item[1:]) for item in neuron_ids]

        [spiketrains_nestedlist, neuron_labels, time_axis] = get_binary_spiketrains(spiketimes_superset,
                                                                                    neurons=neurons)

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    if sampling_rate is None:
        sampling_rate = 1 / siganal.sampling_period

    if window is None:
        window = siganal.window

    if neurons is None:
        neurons = "all"

    total_duration = window[1] - window[0]

    num_samples = int(total_duration * sampling_rate)
    time_axis = np.linspace(window[0], window[1], num_samples)
    num_neurons = len(spiketimes_set)

    yticks = []

    if neurons=="all":
        spiketrains = np.zeros((num_neurons, num_samples))
        row = 0
        for nX, indiv_spiketimes in spiketimes_set.items():
            index = __get_valid_indices(indiv_spiketimes, window, sampling_rate, num_samples)
            spiketrains[row, index] = 1.0
            yticks.append(nX)
            row += 1
    else: # neurons = range(a, b) or neurons = [1, 4, 5, 9]
        spiketrains = np.zeros((len(neurons), num_samples))
        row = 0
        for i in neurons:
            neuron_id = "n" + str(i)
            indiv_spiketimes = spiketimes_set[neuron_id]
            index = __get_valid_indices(indiv_spiketimes, window, sampling_rate, num_samples)
            spiketrains[row, index] = 1.0  # row != i because neurons can be = [13, 14, 15 ,16 ...]
            yticks.append(neuron_id)

    return spiketrains, yticks, time_axis
