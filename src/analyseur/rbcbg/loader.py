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
    This is the parent class for :class:`.LoadRates`

    - Instantiated with the full file path
        - sets atrributes: `full_filepath`, `filename`
    - Contains instance method :meth:`.get_region_name`

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
        Returns region name for respective nucleus name for which the firing rates are in the file.

        +---------------------------------------+--------------+
        | Nuclei                                | Region name  |
        +=======================================+==============+
        | `["CSN", "PTN", "CTX_E", "CTX_I"]`    | `"cortex"`   |
        +---------------------------------------+--------------+
        | `["FSI", "STN", "GPe", "GPiSNr"]`     | `"bg"`       |
        +---------------------------------------+--------------+
        | `["TRN", "TH"]`                       | `"thalamus"` |
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
    Loads the csv file containing firing rates for all the neurons in a particular nucleus and
    **returns the firing rates (average across all channels) and associated time in seconds**
    by calling :py:meth:`.get_mean_rates`.

    +-------------------------------------+---------------------------------------------------------+
    | Methods                             | Argument                                                |
    +=====================================+=========================================================+
    | :py:meth:`.extract_nucleus_name`    | - no arguments (access instantiated attribute)          |
    +-------------------------------------+---------------------------------------------------------+
    | :py:meth:`.extract_modelID`         | - no arguments (access instantiated attribute)          |
    +-------------------------------------+---------------------------------------------------------+
    | :py:meth:`.extract_percentage`      | - no arguments (access instantiated attribute)          |
    +-------------------------------------+---------------------------------------------------------+
    | :py:meth:`.get_mean_rates`          | - no arguments (access instantiated attribute)          |
    +-------------------------------------+---------------------------------------------------------+

    =========
    Use Cases
    =========

    ------------------
    1. Pre-requisites
    ------------------

    1.1. Import Modules and Instantiate
    ```````````````````````````````````
    ::

        from analyseur.rbcbg.loader import LoadRates

        loadFR = LoadRates("GPiSNr_model_9_percent_0.csv")

    ---------
    2. Cases
    ---------

    2.1. Load file and get the firing rates
    ```````````````````````````````````````
    ::

        t_sec, rates_Hz = loadFR.get_mean_rates()

    2.2. Extract the nucleus name
    `````````````````````````````
    ::

        nuc = loadFR.extract_nucleus_name()

    2.3. Extract the model ID
    `````````````````````````
    ::

        modelID = loadFR.extract_modelID()

    2.4. Extract the disinhibition percentage
    `````````````````````````````````````````
    ::

        pc = loadFR.extract_percentage()

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    _description = ( "LoadSpikeTimes loads the spike times containing csv file "
                   + "and `get_spiketimes_superset` returns a dictionary containing the "
                   + "spike times in milliseconds for all the neurons recorded." )
    __pattern_with_nucleus_name = r"^(.*?)_"
    __pattern_with_modelID = r"^.+model_(\d+)"
    __pattern_with_percentage = r"^.+percent_(\d+)\."


    def extract_nucleus_name(self):
        """
        Extracts <nucleus> name from `<nucleus>_model_<ID>_percent_<value>.csv`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # flist = filename.split("_")
        # nucleus = flist[1].split(".")[0]
        match = re.match(self.__pattern_with_nucleus_name, self.filename)

        if match:
            nucleus = match.group(1)
        else:
            print("Filename is not in the form '<nucleus>_model_<ID>_percent_<value>.csv'.")
            nucleus = None

        return nucleus

    def extract_modelID(self):
        """
        Extracts <ID> name from `<nucleus>_model_<ID>_percent_<value>.csv`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        match = re.match(self.__pattern_with_modelID, self.filename)

        if match:
            modelID = int(match.group(1))
        else:
            print("Filename is not in the form '<nucleus>_model_<ID>_percent_<value>.csv'.")
            modelID = None

        return modelID


    def extract_percentage(self):
        """
        Extracts <value> name from `<nucleus>_model_<ID>_percent_<value>.csv`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        match = re.match(self.__pattern_with_percentage, self.filename)

        if match:
            percentage = int(match.group(1))
        else:
            print("Filename is not in the form '<nucleus>_model_<ID>_percent_<value>.csv'.")
            percentage = None

        return percentage


    def __get_rates_superset(self):
        """
        Returns a dictionary containing the firing rates (numpy.array data type) in seconds
        for all the neurons recorded with a sampling period of 1 ms.

        NOTE: old version of rBCBG simulations spits results (firing rate) for each channel (total=4)
        current version spits results simply as the average firing rate of all the neurons across all channels

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

    def __get_mean_rates(self):
        """
        Returns a the average (across all channels) firing rates (numpy.array data type) in seconds
        for all the neurons recorded with a sampling period of 1 ms.

        NOTE: old version of rBCBG simulations spits results (firing rate) for each channel (total=4)
        current version spits results simply as the average firing rate of all the neurons across all channels

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        dataframe = pd.read_csv(self.full_filepath)

        return np.mean(dataframe.values, axis=1)

    def get_mean_rates(self):
        """
        Returns the average (across all channels) firing rate and its corresponding time stamps (in seconds)
        recorded with a sampling period of 1 ms.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        dataframe = pd.read_csv(self.full_filepath)

        time_sec = np.linspace(0,
                               (len(dataframe)-1)*self.siganal.sampling_period,
                               len(dataframe))

        rates_Hz = dataframe.squeeze().values    # squeeze removes the column dimension

        return times_sec, rates_Hz
