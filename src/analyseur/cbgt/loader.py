# ~/analyseur/cbgt/loader.py
#
# Documentation by Lungsi 2 Oct 2025
#
# This contains function for loading the files
#

import re
import pandas as pd

class LoadSpikeTimes(object):
    """
    Loads the csv file containing spike times for all the neurons
    in a particular nucleus and returns all their spike trains in milliseconds.

    +------------------------------+------------------------------------+--------------------------------------------+
    | Methods                      | Argument                           | Return                                     |
    +==============================+====================================+============================================+
    | :py:meth:`.get_spiketrains`  | - no arguments                     | - dictionary with keys, `n<X>` where       |
    |                              | - instantiated with full file path | :math:`X \\in [0, N] \\subset \\mathbb{Z}` |
    +------------------------------+------------------------------------+--------------------------------------------+

    **Use Case:**

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketrains()

    """
    _description = ( "LoadSpikeTimes loads the spike times containing csv file "
                   + "and `get_spiketrains` returns a dictionary containing the "
                   + "spike times in milliseconds for all the neurons recorded." )

    pattern_with_nucleus_name = r"\_(.*?)\."
    nuclei_ctx = ["CSN", "PTN", "IN"]
    nuclei_bg = ["FSI", "GPe", "GPi", "MSN", "STN"]
    significant_digits = 5
    ms_to_s = 1000  # multiplicand
    t_start_recording = 2000  # subtrahend


    def __init__(self, full_filepath=" "):
        self.full_filepath = full_filepath
        self.filename = full_filepath.split("/")[-1]


    def _extract_nucleus_name(self, filename):
        """Extracts <nucleus> name from `spikes_<nucleus>.csv`"""
        # flist = filename.split("_")
        # nucleus = flist[1].split(".")[0]
        match = re.search(self.pattern_with_nucleus_name, filename)

        if match:
            nucleus = match.group(1)
        else:
            print("Filename is not in the form 'spikes_<nuclues>.csv'.")
            nucleus = None

        return nucleus


    def _get_region_name(self, filename):
        """Returns region name for respective nucleus name for which the spike times are for in the file."""
        nucleus = self._extract_nucleus_name(filename)

        if nucleus in self.nuclei_ctx:
            region = "cortex"
        elif nucleus in self.nuclei_bg:
            region = "bg"
        else:
            region = "thalamus"

        return region


    def _get_multiplicand_subtrahend(self, region):
        """For respective region, returns factors (multiplicand and subtrahend)
        to set spike times unit to milliseconds."""
        if region == "bg":
            multiplicand = self.ms_to_s
            subtrahend = self.t_start_recording
        else:
            multiplicand = 1
            subtrahend = 0

        return [multiplicand, subtrahend]


    def _extract_smallest_largest_neuron_id(self, dataframe):
        """Returns the smallest & largest neuron id whose spike times are
        recorded in the file (loaded as a panda dataframe)."""
        neuron_ids = dataframe.filter(like="i").values

        smallest_id = min(neuron_ids)[0]
        largest_id = max(neuron_ids)[0]

        return [smallest_id, largest_id]


    def _get_spike_times_for_a_neuron(self, dataframe, neuron_id, multiplicand, subtrahend):
        """Returns the spike times (numpy.array data type) in milliseconds for a given neuron."""
        raw_neuron_id_times = dataframe[ dataframe["i"] == neuron_id ]["t"]
        spike_times = (raw_neuron_id_times.apply(lambda x: round(x, self.significant_digits)).values
                       * multiplicand - subtrahend)

        return spike_times


    def get_spiketrains(self):
        """
        Returns a dictionary containing the spike times (numpy.array data type) in milliseconds
        for all the neurons recorded into the file as value of the key `n<X>` where
        :math:`X \\in [0, N] \\subset \\mathbb{Z}`.
        """
        dataframe = pd.read_csv(self.full_filepath)
        [min_id, max_id] = self._extract_smallest_largest_neuron_id(dataframe)

        region = self._get_region_name(self.filename)
        [multiplicand, subtrahend] = self._get_multiplicand_subtrahend(region)

        spiketrain = {"n" + str(min_id):
                          self._get_spike_times_for_a_neuron(dataframe, min_id, multiplicand, subtrahend)}

        for n_id in range(1, max_id + 1):
            spiketrain["n" + str(n_id)] = self._get_spike_times_for_a_neuron(dataframe, n_id, multiplicand, subtrahend)

        return spiketrain