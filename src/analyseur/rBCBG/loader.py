# ~/analyseur/rbcbg/loader.py
#
# Documentation by Lungsi 18 Nov 2025
#
# This contains function for loading the files
#

import re
import numbers

import pandas as pd
import numpy as np

from analyseur.rbcbg.parameters import SimulationParams, SignalAnalysisParams


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
        self.simparams = SimulationParams()
        self.siganal = SignalAnalysisParams()


    def get_region_name(self, nucleus):
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
        if nucleus in self.simparams.nuclei_ctx:
            region = "cortex"
        elif nucleus in self.simparams.nuclei_bg:
            region = "bg"
        else:
            region = "thalamus"

        return region


class LoadRates(CommonLoader):
    """
    Loads the csv file containing spike times for all the neurons
    in a particular nucleus and **returns all their spike times in seconds** by calling :py:meth:`.get_spiketimes_superset`.

    +-------------------------------------+---------------------------------------------------------+-------------------------------------------------------------------+
    | Methods                             | Argument                                                | Return                                                            |
    +=====================================+=========================================================+===================================================================+
    | :py:meth:`.get_spiketimes_superset` | - no arguments                                          | - dictionary with keys, `n<X>` where `X ∊ [0, N] ⊂ 𝗭`             |
    |                                     | - instantiated with full file path                      | - key value is a array of spike times for respective neuron `n<X>`|
    +-------------------------------------+---------------------------------------------------------+-------------------------------------------------------------------+
    | :py:meth:`.get_spiketimes_subset`   | - superset (return of :meth:`.get_spiketimes_superset`) | - dictionary with keys, `n<X>` where `X ∊ neurons`                |
    |                                     | - `"neurons"` ("all", range or list)                    | - key value is a array of spike times for respective neuron `n<X>`|
    +-------------------------------------+---------------------------------------------------------+-------------------------------------------------------------------+

    =========
    Use Cases
    =========

    ------------------
    1. Pre-requisites
    ------------------

    1.1. Import Modules and Instantiate
    ```````````````````````````````````
    ::

        from analyseur.cbgt.loader import LoadSpikeTimes

        loadST = LoadSpikeTimes("spikes_GPi.csv")

    ---------
    2. Cases
    ---------

    2.1. Load file and get the whole spike times
    `````````````````````````````````````````````
    ::

        spiketimes_superset = loadST.get_spiketimes_superset()

    2.2. From the whole spike times get a subset; specific range
    ````````````````````````````````````````````````````````````
    ::

        neurons = range(30, 62)  # neuron id from "n30" to "n62"
        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=neurons)

    2.3. From the whole spike times get a subset; specific list
    ```````````````````````````````````````````````````````````
    ::

        neurons = [1, 2, 3, 6, 9, 10, 11, 21, 31]  # neuron ids "n1", "n2", ..., "n21", "n31"
        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=neurons)

    2.4. From the whole spike times get a subset; first N neurons
    `````````````````````````````````````````````````````````````
    ::

        N = 50  # first 50 neurons regardless of the neuron id
        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons=N)

    2.5 Superset and subset are the same
    ````````````````````````````````````
    ::

        spiketimes_subset = LoadSpikeTimes.get_spiketimes_subset(spiketimes_superset, neurons="all")

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    _description = ( "LoadSpikeTimes loads the spike times containing csv file "
                   + "and `get_spiketimes_superset` returns a dictionary containing the "
                   + "spike times in milliseconds for all the neurons recorded." )
    __pattern_with_nucleus_name = r"^(.*?)_"
    __pattern_with_modelID = r"^.+model_(\d+)"
    __pattern_with_percentage = r"^.+percent_(\d+)\."


    def extract_nucleus_name(self, filename):
        """
        Extracts <nucleus> name from `spikes_<nucleus>.csv`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # flist = filename.split("_")
        # nucleus = flist[1].split(".")[0]
        match = re.match(self.__pattern_with_nucleus_name, filename)

        if match:
            nucleus = match.group(1)
        else:
            print("Filename is not in the form '<nucleus>_model_\d+_percent_\d+.csv'.")
            nucleus = None

        return nucleus

    def extract_modelID(self, filename):
        """
        Extracts <nucleus> name from `spikes_<nucleus>.csv`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        match = re.match(self.__pattern_with_modelID, filename)

        if match:
            modelID = int(match.group(1))
        else:
            print("Filename is not in the form '<nucleus>_model_<ID>_percent_\d+.csv'.")
            modelID = None

        return modelID


    def extract_percentage(self, filename):
        """
        Extracts <nucleus> name from `spikes_<nucleus>.csv`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        match = re.match(self.__pattern_with_percentage, filename)

        if match:
            percentage = int(match.group(1))
        else:
            print("Filename is not in the form '<nucleus>_model_\d+_percent_<value>.csv'.")
            percentage = None

        return percentage


    def get_rates_superset(self):
        """
        Returns a dictionary containing the spike times (numpy.array data type) in seconds
        for all the neurons recorded into the file as value of the key `n<X>` where
        :math:`X \\in [0, N] \\subset \\mathbb{Z}`.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        dataframe = pd.read_csv(self.full_filepath)
        n_time, n_channels = dataframe.shape

        rates_superset = {}
        # for c_id in range(n_channels):
        #     rates_superset["c" + str(c_id)] = dataframe.values[:,c_id] / self.siganal._1000ms
        for c_id, cols in dataframe.items():
            rates_superset["c" + str(c_id)] = cols / self.siganal._1000ms

        return rates_superset

    def get_mean_rates(self):
        """
        Returns an array of mean rates across the ten channels.

        Returns a dictionary containing the spike times (numpy.array data type) in seconds
        for all the neurons recorded into the file as value of the key `n<X>` where
        :math:`X \\in [0, N] \\subset \\mathbb{Z}`.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        dataframe = pd.read_csv(self.full_filepath)

        return np.mean(dataframe.values, axis=1)