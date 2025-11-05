# ~/analyseur/cbgt/stat/isi.py
#
# Documentation by Lungsi 2 Oct 2025
#
#

import numpy as np

# from .compute_shared import compute_grand_mean as cgm
from analyseur.cbgt.stats.compute_shared import compute_grand_mean as cgm
from analyseur.cbgt.parameters import SignalAnalysisParams

class InterSpikeInterval(object):
    """
    Computes interspike intervals for the given spike times

    +------------------------------+--------------------------------------------------------------------------------------------------------------------+
    | Methods                      | Argument                                                                                                           |
    +==============================+====================================================================================================================+
    | :py:meth:`.compute`          | - `spiketimes_set`: Dictionary returned; see :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset` |
    |                              | - also see :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`                                     |
    +------------------------------+--------------------------------------------------------------------------------------------------------------------+
    | :py:meth:`.inst_rates`       | - `isi_set`: Dictionary returned; see :py:meth:`.compute`                                                          |
    +------------------------------+--------------------------------------------------------------------------------------------------------------------+
    | :py:meth:`.avg_inst_rates`   | - `inst_rates_set`: Dictionary returned; see :py:meth:`.inst_rates`                                                |
    |                              | - `tbins_set`: 2nd tuple (Dictionary) returned; see :py:meth:`.compute`                                            |
    |                              | - `binsz`: [OPTIONAL] 0.01 (default)                                                                               |
    +------------------------------+--------------------------------------------------------------------------------------------------------------------+
    | :py:meth:`.mean_freqs`       | - `isi_set`: Dictionary returned; see :py:meth:`.compute`                                                          |
    +------------------------------+--------------------------------------------------------------------------------------------------------------------+
    | :py:meth:`.grand_mean_freq`  | - `isi_set`: Dictionary returned; see :py:meth:`.compute`                                                          |
    +------------------------------+--------------------------------------------------------------------------------------------------------------------+

    =========
    Use Cases
    =========

    ------------------
    1. Pre-requisites
    ------------------

    1.1. Import Modules
    ````````````````````
    ::

        from analyseur.cbgt.loader import LoadSpikeTimes
        from analyseur.cbgt.stats.isi import InterSpikeInterval

    1.2. Load file and get spike times
    ```````````````````````````````````
    ::

        loadST = LoadSpikeTimes("spikes_GPi.csv")
        spiketimes_superset = loadST.get_spiketimes_superset()

    ---------
    2. Cases
    ---------

    2.1. Compute Inter-Spike Intervals (for all neurons)
    `````````````````````````````````````````````````````
    ::

        [I, all_t] = InterSpikeInterval.compute(spiketimes_superset)

    This returns the value for
    :math:`I = \\left\\{\\overrightarrow{ISI}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`;
    see formula :py:meth:`.compute`.

    2.2. Compute Instantaneuous Rates (for all neurons)
    ```````````````````````````````````````````````````
    ::

        J = InterSpikeInterval.inst_rates(I)

    This returns the value for
    :math:`R = \\left\\{\\vec{R}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`;
    see formula :py:meth:`.inst_rates`

    2.3. Compute Average Instantaneuous Rates (for all neurons)
    ```````````````````````````````````````````````````````````
    ::

        E = InterSpikeInterval.avg_inst_rates(J, all_t)

    This returns the value for :math:`\\vec{\\Xi} = [\\xi_t]_{\\forall{t}}`;
    see formula :py:meth:`.avg_inst_rates`

    2.4. Compute Mean Frequencies (for all neurons)
    ````````````````````````````````````````````````
    ::

        F = InterSpikeInterval.mean_freqs(I)

    This returns the value for :math:`\\vec{F} = \\left[\\overline{f^{(i)}}\\right]_{\\forall{i \\in [1, n_{nuc}]}}`;
    see formula :py:meth:`.mean_freqs`

    2.5. Compute Global Mean Frequency
    ```````````````````````````````````
    ::

        grand_f = InterSpikeInterval.grand_mean_freq(I)

    This returns the value for :math:`\\overline{f}`; see formula :py:meth:`.grand_mean_freq`

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    __siganal = SignalAnalysisParams()

    @classmethod
    def compute(cls, spiketimes_set=None):
        """
        Returns the interspike interval for all individual neurons.

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :return: 2-tuple

        - dictionary of individual neurons whose values are their respective interspike interval
        - dictionary of individual neurons whose values are their respective times for corresponding interspike interval

        **Formula**

        .. table:: Formula
        ================================================================================================== ======================================================
          Definitions                                                                                       Interpretation
        ================================================================================================== ======================================================
         total neurons, :math:`n_{nuc}`                                                                     total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                            i-th neuron in the pool of :math:`n_{Nuc}` neurons
         total spikes, :math:`n_{spk}^{(i)}`                                                                total number of spikes (spike times) by i-th neuron
         interspike interval, :math:`isi_{k}^{(i)}`                                                         k-th absolute interval between successive spike times
         :math:`\\overrightarrow{ISI}^{(i)} = \\left[isi_k^{(i)}\\right]_{\\forall{k \\in [1, n_{spk}^{(i)})}}`       array of all interspike intervals of i-th neuron
         :math:`I = \\left\\{\\overrightarrow{ISI}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`              set of array interspike intervals of all neurons
        ================================================================================================== ======================================================

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        interspike_intervals = {}
        tbins_set = {}

        for n_id, spiketimes in spiketimes_set.items():
            interspike_intervals[n_id] = np.diff(spiketimes)
            tbins_set[n_id] = spiketimes[1:]

        return interspike_intervals, tbins_set

    @classmethod
    def inst_rates(cls, isi_set=None):
        """
        Returns the instantaneous rates for all individual neurons.

        :param isi_set: Dictionary returned using :py:meth:`.compute`
        :return: dictionary of individual neurons whose values are their respective instantaneous rates

        **Formula**

        .. table::
        ================================================================================================== ======================================================
          Definitions                                                                                       Interpretation
        ================================================================================================== ======================================================
         total neurons, :math:`n_{nuc}`                                                                     total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                            i-th neuron in the pool of :math:`n_{Nuc}` neurons
         total spikes, :math:`n_{spk}^{(i)}`                                                                total number of spikes (spike times) by i-th neuron
         interspike interval, :math:`isi_{k}^{(i)}`                                                         k-th absolute interval between successive spike times
         :math:`\\overrightarrow{ISI}^{(i)} = \\left[isi_k^{(i)}\\right]_{\\forall{k \\in [1, n_{spk}^{(i)})}}`       array of all interspike intervals of i-th neuron
         :math:`I = \\left\\{\\overrightarrow{ISI}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`              set of array interspike intervals of all neurons
        ================================================================================================== ======================================================

        Then, the instantaneuous rate of i-th neuron is

        .. math::

            \\vec{R}^{(i)} &= \\frac{1}{\\overrightarrow{ISI}^{(i)}} \n
                           &= \\left[\\frac{1}{isi_k^{(i)}}\\right]_{\\forall{k \\in [1, n_{spk}^{(i)})}}

        We therefore get

        .. table::
        =================================================================================== ======================================================
          Definitions                                                                             Interpretation
        =================================================================================== ======================================================
         :math:`\\vec{R}^{(i)}`                                                               array of instantaneous rates of i-th neuron
         :math:`R = \\left\\{\\vec{R}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`           set of array of instaneous rates of all (:math:`n_{Nuc}`) neurons
        =================================================================================== ======================================================

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        inst_rates = {}

        for n_id, isi in isi_set.items():
            # n_spikes = len(isi) + 1
            if len(isi) == 0:
                inst_rates[n_id] = 0
            else:
                inst_rates[n_id] = 1 / isi

        return inst_rates

    @classmethod
    def avg_inst_rates(cls, inst_rates_set=None, tbins_set=None, binsz=None):
        """
        Returns the average instantaneous rates for all individual neurons.

        :param tbins_set: Dictionary returned using :py:meth:`.compute`
        :param inst_rates_set: Dictionary returned using :py:meth:`.inst_rates`
        :param binsz: integer or float; `0.01` [default]
        :return: 3-tuple

        - list of average instantaneous rates
        - array of centers for all the time bins (use this as time axis for plotting)
        - list of number of data point per bin (can be useful for colorbar)

        **Formula**

        .. table::
        =================================================================================== ======================================================
          Definitions                                                                         Interpretation
        =================================================================================== ======================================================
         total neurons, :math:`n_{nuc}`                                                       total number of neurons in the Nucleus
         neuron index, :math:`i`                                                              i-th neuron in the pool of :math:`n_{Nuc}` neurons
         :math:`\\vec{R}^{(i)}`                                                               array of instantaneous rates of i-th neuron
         :math:`R = \\left\\{\\vec{R}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`           set of array of instaneous rates of all (:math:`n_{Nuc}`) neurons
         :math:`\\vec{J}^{(i)}`                                                               array of time points where instantaneous rates of i-th neuron occur
         :math:`J = \\left\\{\\vec{J}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`           set of array of time points of all (:math:`n_{Nuc}`) neurons
        =================================================================================== ======================================================

        For using bin-based conditional average let

        .. table::
        ====================================================================================== ======================================================
          Definitions                                                                            Interpretation
        ====================================================================================== ======================================================
         :math:`\\vec{P} = vec(J)`                                                               array containing all the time points from all the neurons
         :math:`\\vec{\\Xi} = vec(R)`                                                            array containing all the instantaneous rates from all the neurons
         total bins, :math:`n_{bins} = \\mid vec(J) \\mid`                                       total number of time points from all neurons
         bin size, :math:`w`                                                                     fixed bin width for each time bin
         bin center, :math:`c_{\\forall{t} \\in [0, n_{bins} - 1]}`                              center of t-th time bin
         bin interval, :math:`b_t = \\left[c_t - \\frac{w}{2}, c_t + \\frac{w}{2}\\right)`            interval of t-th time bin
        ====================================================================================== ======================================================

        Then, the average instantaneuous rate for t-th bin is

        .. math::

            \\xi_t &= \\mathbb{E}\\left[\\Xi_p \\mid p \\in b_t\\right] \n
                &= \\frac{\\sum_{p \\in P}(\\Xi_p \\cdot 1_{\\{p \\in b_t\\}})}{\\sum_{p \\in P} 1_{\\{p \\in b_t\\}}}

        where

        - :math:`\\mathbb{E}` is the expectation function,
        - :math:`1_{\\{p \\in b_t\\}}` is the indicator function; 1 if condition is true otherwise 0,
        - :math:`\\sum_{p \\in P} 1_{\\{p \\in b_t\\}}` is the number of time points that fall in the t-th bin
        - numerator is the sum of instantaneous rates that fall in the t-th bin

        We therefore get

        .. table::
        ================================================== ======================================================
          Definitions                                        Interpretation
        ================================================== ======================================================
         :math:`\\xi_t`                                      average instantaneous rate for t-th bin
         :math:`\\vec{\\Xi} = [\\xi_t]_{\\forall{t}}`           array of average instantaneous rates for all bins
        ================================================== ======================================================

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        # Put all times and instantaneuous rates of respective neurons into one list
        vec_all_times = []
        vec_all_inst = []
        [vec_all_times.extend(x) for x in tbins_set.values()]
        [vec_all_inst.extend(x) for x in inst_rates_set.values()]

        # Convert the above two lists to arrays
        arr_all_times = np.array(vec_all_times)
        arr_all_inst = np.array(vec_all_inst)

        # Create time bins
        [t_min, t_max] = [np.min(arr_all_times), np.max(arr_all_times)]
        bins = np.arange(t_min, t_max + binsz, binsz)

        # Digitize time bins
        bin_indices = np.digitize(arr_all_times, bins) - 1

        # Calculate average instantaneuous rate per bin
        bin_centers = (bins[:-1] + bins[1:]) / 2  # time or x-axis for plotting the average rates
        avg_rates = []
        bin_counts = []  # Tracks number of data points per bin
        for i in range(len(bins) - 1):
            rates_in_bin = arr_all_inst[bin_indices == i]
            # avg_rates.append(np.mean(rates_in_bin) if rates_in_bin.size > 0 else 0)
            if rates_in_bin.size > 0:
                avg_rates.append(np.mean(rates_in_bin))
                bin_counts.append(rates_in_bin.size)
            else:
                avg_rates.append(0)
                bin_counts.append(0)

        return avg_rates, bin_centers, bin_counts

    @classmethod
    def mean_freqs(cls, isi_set=None):
        """
        Returns the mean frequencies for all individual neurons.

        :param isi_set: Dictionary returned using :py:meth:`.compute`
        :return: dictionary of individual neurons whose values are their respective mean frequencies

        **Formula**

        .. table:: Formula_mean_freqs_1.1
        ================================================================================================== ======================================================
          Definitions                                                                                       Interpretation
        ================================================================================================== ======================================================
         total neurons, :math:`n_{nuc}`                                                                     total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                            i-th neuron in the pool of :math:`n_{Nuc}` neurons
         total spikes, :math:`n_{spk}^{(i)}`                                                                total number of spikes (spike times) by i-th neuron
         interspike interval, :math:`isi_{k}^{(i)}`                                                         k-th absolute interval between successive spike times
         :math:`\\overrightarrow{ISI}^{(i)} = \\left[isi_k^{(i)}\\right]_{\\forall{k \\in [1, n_{spk}^{(i)})}}`       array of all interspike intervals of i-th neuron
         :math:`I = \\left\\{\\overrightarrow{ISI}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`              set of array interspike intervals of all neurons
        ================================================================================================== ======================================================

        Then, the mean spiking frequency of i-th neuron is

        .. math::

            \\overline{f^{(i)}} = \\frac{1}{(n_{spk}^{(i)} - 1)} \\sum_{j=1}^{(n_{spk}^{(i)} - 1)}\\frac{1}{isi_{j}^{(i)}}

        We therefore get

        .. table:: Formula_mean_freqs_1.2
        =================================================================================== ======================================================
          Definitions                                                                         Interpretation
        =================================================================================== ======================================================
         :math:`\\overline{f^{(i)}}`                                                          mean spiking frequency of i-th neuron
         :math:`\\vec{F} = \\left[\\overline{f^{(i)}}\\right]_{\\forall{i \\in [1, n_{nuc}]}}`             array of mean frequencies of all (:math:`n_{Nuc}`) neurons
        =================================================================================== ======================================================

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        mean_spiking_freq = {}

        for n_id, isi in isi_set.items():
            # n_spikes = len(isi) + 1
            if len(isi) == 0:
                mean_spiking_freq[n_id] = 0
            else:
                mean_spiking_freq[n_id] = (1 / len(isi)) * np.sum(1 / isi)

        return mean_spiking_freq

    @classmethod
    def grand_mean_freq(cls, isi_set=None):
        """
        Returns the grand mean frequency which is the mean of mean frequencies of all the neurons
        
        :param isi_set: Dictionary returned using :py:meth:`.compute`
        :return: a number

        **Formula**
        
        .. table:: Formula
        =================================================================================== ======================================================
          Definitions                                                                         Interpretation
        =================================================================================== ======================================================
         total neurons, :math:`n_{Nuc}`                                                       total number of neurons in the Nucleus
         neuron index, :math:`i`                                                              i-th neuron in the pool of :math:`n_{Nuc}` neurons
         mean frequency, :math:`\\overline{f^{(i)}}`                                          mean spiking frequency of i-th neuron
         :math:`\\vec{F} = \\left[\\overline{f^{(i)}}\\right]_{\\forall{i \\in [1, n_{nuc}]}}`             array of mean frequencies of all (:math:`n_{Nuc}`) neurons
         grand mean frequency, :math:`\\overline{f} = \\mu\\left(\\vec{F}\\right)`                    grand or global mean spiking frequency
        =================================================================================== ======================================================

        where, :math:`\\mu(\\cdot)` is the `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_ over the given dimension.

        NOTE: The array :math:`\\vec{F}` is obtained by calling :py:meth:`.mean_freqs`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        all_neurons_mean_freq = cls.mean_freqs(isi_set)
        return cgm(all_neurons_mean_freq)