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

    2.1. Compute basic measure of spike times synchrony (for all neurons)
    ``````````````````````````````````````````````````````````````````````
    ::

        B = Synchrony.compute_basic(spiketimes_superset)

    This returns the value for

    .. math::

       Sync = \\sqrt{\\frac{var\\left(\\left[\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{t}}\\right)\\right]_{\\forall{i}}\\right)}{\\mu\\left(\\left[var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}}

    2.2. Compute the basic measure of synchrony on a smoother frequency estimation
    ```````````````````````````````````````````````````````````````````````````````
    ::

        S = Synchrony.compute_basic_slide(spiketimes_superset)

    This returns the value for

    .. math::

       Sync = \\sqrt{\\frac{var\\left(\\left[\\mu\\left(\\left[f^{{i}}(t)\\right]_{\\forall{t}}\\right)\\right]_{\\forall{i}}\\right)}{\\mu\\left(\\left[var\\left(\\left[f^{(i)}(t)\\right]_{\\forall{i}}\\right)\\right]_{\\forall{t}}\\right)}}

    2.3. Compute Fano factor as a metric for measuring spike times synchrony (for all neurons)
    ```````````````````````````````````````````````````````````````````````````````````````````
    ::

        Fs = Synchrony.compute_fano_factor(spiketimes_superset)

    This returns the value for

    .. math::

        F_{Sync} = \\frac{var\\left(\\left[\\sum_{\\forall{i}}p^{(i)}(t)\\right]_{\\forall{t}}\\right)}{\\mu\\left(\\left[\\sum_{\\forall{i}}p^{(i)}(t)\\right]_{\\forall{t}}\\right)}

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    @classmethod
    def compute(cls, spiketimes_superset, sampling_rate=None,
                window=None, neurons=None, resolution=None):
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