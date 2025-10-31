# ~/analyseur/cbgt/loader.py
#
# Documentation by Lungsi 2 Oct 2025
#
# This contains function for loading the files
#

import re
import pandas as pd
import numpy as np

from analyseur.cbgt.parameters import SimulationParams

simparams = SimulationParams()

class CommonLoader(object):
    """
    This is the parent class for :class:`.LoadSpikeTimes` and :class:`.LoadChannelVorG`

    - Instantiated with the full file path
        - sets atrributes: `full_filepath`, `filename`
    - Contains static method :meth:`.get_region_name`

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    def __init__(self, full_filepath=" "):
        self.full_filepath = full_filepath
        self.filename = full_filepath.split("/")[-1]

    @staticmethod
    def get_region_name(nucleus):
        """
        Returns region name for respective nucleus name for which the spike times are for in the file.

        +---------------------------------------+--------------+
        | Nuclei                                | Region name  |
        +=======================================+==============+
        | `["CSN", "PTN", "IN"]`                | `"cortex"`   |
        +---------------------------------------+--------------+
        | `["FSI", "GPe", "GPi", "MSN", "STN"]` | `"bg"`       |
        +---------------------------------------+--------------+
        | `['AMPA', 'NMDA', 'GABAA', 'GABAB']`  | `"thalamus"` |
        +---------------------------------------+--------------+

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        if nucleus in simparams.nuclei_ctx:
            region = "cortex"
        elif nucleus in simparams.nuclei_bg:
            region = "bg"
        else:
            region = "thalamus"

        return region


class LoadSpikeTimes(CommonLoader):
    """
    Loads the csv file containing spike times for all the neurons
    in a particular nucleus and returns all their spike times in seconds.

    +-------------------------------------+------------------------------------+-------------------------------------------------------------------+
    | Methods                             | Argument                           | Return                                                            |
    +=====================================+====================================+===================================================================+
    | :py:meth:`.get_spiketimes_superset` | - no arguments                     | - dictionary with keys, `n<X>` where `X ∊ [0, N] ⊂ 𝗭`             |
    |                                     | - instantiated with full file path | - key value is a list of spike times for respective neuron `n<X>` |
    +-------------------------------------+------------------------------------+-------------------------------------------------------------------+

    **Use Case:**

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/spikes_GPi.csv")
      spike_trains = loadST.get_spiketimes_superset()

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    _description = ( "LoadSpikeTimes loads the spike times containing csv file "
                   + "and `get_spiketimes_superset` returns a dictionary containing the "
                   + "spike times in milliseconds for all the neurons recorded." )
    __pattern_with_nucleus_name = r"\_(.*?)\."


    def _extract_nucleus_name(self, filename):
        """Extracts <nucleus> name from `spikes_<nucleus>.csv`"""
        # flist = filename.split("_")
        # nucleus = flist[1].split(".")[0]
        match = re.search(self.__pattern_with_nucleus_name, filename)

        if match:
            nucleus = match.group(1)
        else:
            print("Filename is not in the form 'spikes_<nuclues>.csv'.")
            nucleus = None

        return nucleus


    def __get_multiplicand_subtrahend(self, region):
        """For respective region, returns factors (multiplicand and subtrahend)
        to set spike times unit to seconds."""
        if region in ("bg", "thalamus"):
            multiplicand = 1
        else:
            multiplicand = 1 / self.simparams._1000ms

        subtrahend = self.simparams.t_start_recording / self.simparams._1000ms

        return [multiplicand, subtrahend]


    def __extract_smallest_largest_neuron_id(self, dataframe):
        """Returns the smallest & largest neuron id whose spike times are
        recorded in the file (loaded as a panda dataframe)."""
        neuron_ids = dataframe.filter(like="i").values

        if len(neuron_ids)==0:
            smallest_id = 0  # Temporary solution to get the code to execute/plot
            largest_id = 0   # even with errors from child functions
        else:
            smallest_id = min(neuron_ids)[0]
            largest_id = max(neuron_ids)[0]

        return [smallest_id, largest_id]


    def __get_spike_times_for_a_neuron(self, dataframe, neuron_id, multiplicand, subtrahend):
        """Returns the spike times (numpy.array data type) in seconds for a given neuron."""
        raw_neuron_id_times = dataframe[ dataframe["i"] == neuron_id ]["t"]

        if len(raw_neuron_id_times) == 0:
            spike_times = np.array([])
        else:
            spike_times = (raw_neuron_id_times.apply(lambda x: round(x, self.simparams.decimal_places)).values
                           * multiplicand - subtrahend)

        return spike_times


    def get_spiketimes_superset(self):
        """
        Returns a dictionary containing the spike times (numpy.array data type) in seconds
        for all the neurons recorded into the file as value of the key `n<X>` where
        :math:`X \\in [0, N] \\subset \\mathbb{Z}`.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        dataframe = pd.read_csv(self.full_filepath)
        [min_id, max_id] = self.__extract_smallest_largest_neuron_id(dataframe)

        nucleus = self._extract_nucleus_name(self.filename)
        region = self.get_region_name(nucleus)
        [multiplicand, subtrahend] = self.__get_multiplicand_subtrahend(region)

        spiketimes_superset = {"n" + str(min_id):
                                   self.__get_spike_times_for_a_neuron(dataframe, min_id, multiplicand, subtrahend)}

        for n_id in range(1, max_id + 1):
            spiketimes_superset["n" + str(n_id)] = self.__get_spike_times_for_a_neuron(dataframe, n_id, multiplicand, subtrahend)

        return spiketimes_superset

class LoadChannelVorG(CommonLoader):
    """
    Loads the csv file containing spike times for all the neurons
    in a particular nucleus and returns all their spike trains in milliseconds.

    +------------------------------+------------------------------------+--------------------------------------------+
    | Methods                      | Argument                           | Return                                     |
    +==============================+====================================+============================================+
    | :py:meth:`.get_spiketimes_superset`  | - no arguments                     | - dictionary with keys, `n<X>` where       |
    |                              | - instantiated with full file path | :math:`X \\in [0, N] \\subset \\mathbb{Z}` |
    +------------------------------+------------------------------------+--------------------------------------------+

    **Use Case:**

    ::

      from  analyseur.cbgt.loader import LoadSpikeTimes
      loadST = LoadSpikeTimes("/full/path/to/CSN_V_syn_GABAA_1msgrouped_mean_preprocessed4Matlab_SHIFT.csv")
      spike_trains = loadST.get_spiketimes_superset()

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    _description = ( "LoadSpikeTimes loads the spike times containing csv file "
                   + "and `get_spiketimes_superset` returns a dictionary containing the "
                   + "spike times in milliseconds for all the neurons recorded." )
    __pattern_with_nucleus_name = r"(.*?)\_"
    __pattern_with_attrib_name = r"\_V\_syn\_(.*?)\_1msgrouped"
    __nonChnl_attributes = ["L", "g_NMDA", "g_GABAA", "g_GABAB", "g_AMPA"]

    def __prepreprocessSize(self, neurotrans_name):  # As there is a shift in indices in the LFP formula
        full_size = self.simparams.duration - 6 - 1
        if neurotrans_name == 'AMPA':
            start, end = 6, 0
        elif neurotrans_name == 'GABAA':
            start, end = 0, 6
        else:
            start, end = 6, 6
            full_size = self.simparams.duration - 1
        start, end = start, full_size - end

        return start, end

    def _extract_nucleus_attribute_name(self, filename):
        """Extracts <nucleus> name from `<nucleus>_V_syn_<attribute>_1msgrouped_mean_preprocessed4Matlab_SHIFT.csv`"""
        # flist = filename.split("_")
        # nucleus = flist[1].split(".")[0]
        match1 = re.search(self.__pattern_with_nucleus_name, filename)
        match2 = re.search(self.__pattern_with_attrib_name, filename)

        if match1:
            nucleus = match1.group(1)
            if match2:
                attrib = match2.group(1)
            else:
                print("Filename is not in the form '<nucleus>_V_syn_<attribute>_1msgrouped_mean_preprocessed4Matlab_SHIFT.csv'")
                attrib = None
        else:
            print("Filename is not in the form '<nucleus>_V_syn_<attribute>_1msgrouped_mean_preprocessed4Matlab_SHIFT.csv'")
            nucleus = None
            attrib = None

        return nucleus, attrib

    def get_measurables(self):
        """

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        [nucleus, attrib] = self._extract_nucleus_attribute_name(self.filename)
        # region = self.get_region_name(nucleus)

        if attrib in self.simparams.neurotrans + self.__nonChnl_attributes:
            start, end = self.__prepreprocessSize(attrib)
            dataframe = pd.read_csv(self.full_filepath).iloc[start:end, [0]]
            measurables = dataframe.apply(lambda x: round(x, self.simparams.decimal_places_ephys)).values
        else:
            print("Attributes must be from " + str(self.simparams.neurotrans + self.__nonChnl_attributes))
            measurables = None

        return measurables