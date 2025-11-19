# ~/analyseur/rbcbg/stats/psd.py
#
# Documentation by Lungsi 18 Nov 2025
#
# This contains function for loading the files
#
import numbers

import numpy as np
from scipy import signal

from analyseur.rbcbg.parameters import SignalAnalysisParams

class PowerSpectrum(object):
    """
    Computes the power spectra
    (`Welch’s method <https://doi.org/10.1109/TAU.1967.1161901>`_ using
    `scipy.signal.welch <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>`_)
    of all the neurons with given spike times

    +-------------------------------+-------------------------------------------------------------------------------------------------+
    | Methods                       | Argument                                                                                        |
    +===============================+=================================================================================================+
    | :py:meth:`.compute_for_spike` | - `spiketimes_set`: see :class:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`  |
    |                               |      -  also :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`                |
    |                               | - `sampling_rate` [OPTIONAL]: `1000/dt = 10000 Hz` [default]                                    |
    |                               | - `window` [OPTIONAL]: Tuple `(0, 10) seconds` [default]                                        |
    |                               | - `neurons` [OPTIONAL]: "all" [default] or a scalar or list: range(a, b) or [1, 4, 5, 9]        |
    |                               | - `resolution` [OPTIONAL]: `~ 9.76 Hz = sampling_rate/1024` [default]                           |
    +-------------------------------+-------------------------------------------------------------------------------------------------+

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
        spiketimes_set = loadST.get_spiketimes_superset()

    ---------
    2. Cases
    ---------

    2.1. Compute power spectral density (for all neurons)
    ``````````````````````````````````````````````````````
    ::

        B = PowerSpectrum.compute(spiketimes_set)

    2.2. Compute power spectral density for chosen neurons with desired frequency resolution
    ````````````````````````````````````````````````````````````````````````````````````````
    ::

        B = PowerSpectrum.compute(spiketimes_set, neurons=range(30, 120), resolution=5)

    Power spectral density for neurons 30 to 120 with the desired frequency resolution of 5 Hz.

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    __siganal = SignalAnalysisParams()


    @classmethod
    def compute_for_rate(cls, mu_rate_array, resolution=None, method=None):
        # ============== DEFAULT Parameters ==============
        n = len(mu_rate_array)
        T = cls.__siganal.sampling_period  # seconds
        sampling_fs = 1 / T  # Hz

        if resolution is None:
            points_per_segment = 1024
        else:
            points_per_segment = sampling_fs / resolution

        if method is None or method=="welch":
            # Compute power spectrum using Welch's method
            freqs, power = signal.welch(mu_rate_array, sampling_fs, nperseg=points_per_segment)
        elif method=="fft":
            # Compute FFT
            fft_result = np.fft.fft(mu_rate_array)
            # Compute Frequencies (positive only)
            freqs = np.fft.fftfreq(n, T)[:n//2]
            # Compute Power Spectral Density
            power = (1.0 / (sampling_fs * n)) * np.abs(fft_result[:n//2]) ** 2
        elif method=="fft-mag":
            # Compute FFT
            fft_result = np.fft.fft(mu_rate_array)
            # Compute Frequencies (positive only)
            freqs = np.fft.fftfreq(n, T)[:n // 2]
            # [Alternative] Compute Magnitude Spectrum
            power = (2.0 / n) * np.abs(fft_result[:n // 2])

        return freqs, power