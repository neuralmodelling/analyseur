# ~/analyseur/cbgt/stat/psd.py
#
# Documentation by Lungsi 27 Oct 2025
#
# This contains function for loading the files
#

import numpy as np
from scipy import signal

from analyseur.cbgt.parameters import SpikeAnalysisParams
from analyseur.cbgt.curate import get_binary_spiketrains

spikeanal = SpikeAnalysisParams()

class PowerSpectrum(object):
    """
    Computes the power spectra
    (`Welch’s method <https://doi.org/10.1109/TAU.1967.1161901>`_ using
    `scipy.signal.welch <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>`_)
    of all the neurons with given spike times

    +----------------------------+----------------------------------------------------------------------------------------------------------+
    | Methods                    | Argument                                                                                                 |
    +============================+==========================================================================================================+
    | :py:meth:`.compute`        | - `spiketimes_superset`: see :class:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`      |
    |                            | - `sampling_rate` [OPTIONAL]: `1000/dt = 10000 Hz` [default]                                             |
    |                            | - `window` [OPTIONAL]: Tuple `(0, 10) seconds` [default]                                                 |
    |                            | - `neurons` [OPTIONAL]: "all" [default]                                                                  |
    |                            | - `resolution` [OPTIONAL]: `~ 9.76 Hz = sampling_rate/1024` [default]                                    |
    +----------------------------+----------------------------------------------------------------------------------------------------------+

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
        from analyseur.cbgt.stats.psd import PowerSpectrum

    1.2. Load file and get spike times
    ```````````````````````````````````
    ::

        loadST = LoadSpikeTimes("spikes_GPi.csv")
        spiketimes_superset = loadST.get_spiketimes_superset()

    ---------
    2. Cases
    ---------

    2.1. Compute power spectral density (for all neurons)
    ``````````````````````````````````````````````````````
    ::

        B = PowerSpectrum.compute(spiketimes_superset)

    This returns

    - array of sample frequencies
    - power spectral density (or power spectrum)
    - list of spike trains
    - list of neuron id's
    - array of time

    .. math::

       Sync = \\sqrt{\\frac{var\\left(\\left[\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{t}}\\right)\\right]_{\\forall{i}}\\right)}{\\mu\\left(\\left[var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}}

    2.2. Compute the basic measure of synchrony on a smoother frequency estimation
    ```````````````````````````````````````````````````````````````````````````````
    ::

        S = Synchrony.compute_basic_slide(spiketimes_superset)

    This returns the value for

    .. math::

       Sync = \\sqrt{\\frac{var\\left(\\left[\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{t}}\\right)\\right]_{\\forall{i}}\\right)}{\\mu\\left(\\left[var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}}

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    @classmethod
    def compute(cls, spiketimes_superset, sampling_rate=None,
                window=None, neurons=None, resolution=None):
        """
        Returns the power spectral density (or power spectrum) of spiking times from all neurons.

        :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`
        :param sampling_rate: 1000/dt = 10000 Hz [default]
        :param window: Tuple in the form `(start_time, end_time)`; (0, 10) [default]
        :param neurons: "all" [default]                                                                  |
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: a tuple
        - array of sample frequencies
        - power spectral density (or power spectrum)
        - list of spike trains
        - list of neuron id's
        - array of time

        **Formula**

        .. table:: Formula
        ====================================================================================================================== ===============================================================
          Definitions                                                                                                             Interpretation
        ====================================================================================================================== ===============================================================
         total neurons, :math:`n_{Nuc}`                                                                                           total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                                                  i-th neuron in the pool of :math:`n_{Nuc}` neurons
         frequency, :math:`f^{(i)}(t)`                                                                                            frequency of the i-th neuron at time :math:`t`
         frequency matrix, :math:`F = \\left[f(a,b) = f^{(a)}(b)\\right]_{\\forall{a \\in [1, n_{Nuc}], b \\in [t_0, t_T]}}`          frequencies of all (:math:`n_{Nuc}`) neurons for all times
        ====================================================================================================================== ===============================================================

        Let the :math:`var(\\cdot)`, `variance function <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_ and
        the :math:`\\mu(\\cdot)`, `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_
        be implemented as shown

        .. math::

            F = \\overset{\\begin{matrix}t_0 & \\quad\\quad & t_1 & & & &\\ldots & & & t_T\\end{matrix}}
                {\\underset{
                    \\begin{matrix}
                        \\quad\\quad\\uparrow & \\quad\\quad\\quad & \\uparrow & \\quad &\\ldots & & & \\uparrow \n
                        \\quad\\mu_{t_0} & \\quad\\quad\\quad & \\mu_{t_1} & \\quad &\\ldots & & & \\mu_{t_T} & \\rightarrow var_{\\forall{t}}
                    \\end{matrix}}
               {\\begin{bmatrix}
                 f^{(1)}(t_0) & f^{(1)}(t_1) & \\ldots & f^{(1)}(t_T) \n
                 f^{(2)}(t_0) & f^{(2)}(t_1) & \\ldots & f^{(2)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(i)}(t_0) & f^{(i)}(t_1) & \\ldots & f^{(i)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(n_{Nuc})}(t_0) & f^{(n_{Nuc})}(t_1) & \\ldots & f^{(n_{Nuc})}(t_T)
                \\end{bmatrix}
                }}

        Then, we define

        .. math::

            A \\triangleq var\\left(\\begin{bmatrix}
                                       \\mu_{t_0} & \\mu_{t_1} & \\ldots & \\mu_{t_T}
                                     \\end{bmatrix}\\right) = var_{\\forall{t}}

        Implementing the `variance function <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_ and
        the `arithmetic mean function <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_ as shown below

        .. math::

            F = \\overset{\\begin{matrix}t_0 & \\quad\\quad & t_1 & & & &\\ldots & & & t_T\\end{matrix}}
                {\\underset{
                    \\begin{matrix}
                        \\quad\\quad\\uparrow & \\quad\\quad\\quad & \\uparrow & \\quad &\\ldots & & & \\uparrow \n
                        \\quad var_{t_0} & \\quad\\quad\\quad & var_{t_1} & \\quad &\\ldots & & & var_{t_T} & \\rightarrow \\mu_{\\forall{t}}
                    \\end{matrix}}
               {\\begin{bmatrix}
                 f^{(1)}(t_0) & f^{(1)}(t_1) & \\ldots & f^{(1)}(t_T) \n
                 f^{(2)}(t_0) & f^{(2)}(t_1) & \\ldots & f^{(2)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(i)}(t_0) & f^{(i)}(t_1) & \\ldots & f^{(i)}(t_T) \n
                 \\vdots & \\vdots & \\ldots & \\vdots \n
                 f^{(n_{Nuc})}(t_0) & f^{(n_{Nuc})}(t_1) & \\ldots & f^{(n_{Nuc})}(t_T)
                \\end{bmatrix}
                }}

        we make another definition

        .. math::

            B \\triangleq \\mu\\left(\\begin{bmatrix}
                                         var_{t_0} & var_{t_1} & \\ldots & var_{t_T}
                                     \\end{bmatrix}\\right) = \\mu_{\\forall{t}}

        Then, synchrony is measured as

        .. math::

            Sync = \\sqrt{\\frac{A}{B}} = \\sqrt{\\frac{var\\left(\\left[\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{t}}\\right)\\right]_{\\forall{i}}\\right)}{\\mu\\left(\\left[var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}}

        NOTE: This method is a simple histogram-based approach that uses fixed bins.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        #============== DEFAULT Parameters ==============
        if sampling_rate is None:
            sampling_rate = 1 / spikeanal.sampling_period

        if window is None:
            window = spikeanal.window

        if neurons is None:
            neurons = "all"

        if resolution is None:
            points_per_segment = 1024
        else:
            points_per_segment = sampling_rate / resolution

        # Spike times > Spike Train
        [spiketrains, yticks, time_axis] = get_binary_spiketrains(spiketimes_superset, sampling_rate=sampling_rate,
                                                                  window=window, neurons=neurons)

        # Calculate Power Spectra (a.k.a power spectral density, PSD)
        frequencies = []
        power_spectra = []

        for i, spike_train in enumerate(spiketrains):
            f, Pxx = signal.welch(spike_train, fs=sampling_rate, nperseg=points_per_segment)
            frequencies.append(f)
            power_spectra.append(Pxx)

        return frequencies, power_spectra, spiketrains, yticks, time_axis