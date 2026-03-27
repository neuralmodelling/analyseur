# ~/analyseur/cbgtc/stats/psd.py
#
# Documentation by Lungsi 27 Oct 2025
#
# This contains function for loading the files
#
import numbers

import numpy as np
from scipy import signal

from analyseur.cbgtc.parameters import SignalAnalysisParams
from analyseur.cbgtc.curate import get_binary_spiketrains

class PowerSpectrum(object):
    """
    Computes the power spectra
    (`Welch’s method <https://doi.org/10.1109/TAU.1967.1161901>`_ using
    `scipy.signal.welch <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>`_)
    of all the neurons with given spike times

    +-------------------------------+-------------------------------------------------------------------------------------------------+
    | Methods                       | Argument                                                                                        |
    +===============================+=================================================================================================+
    | :py:meth:`.compute_for_spike` | - `spiketimes_set`: see :class:`~analyseur.cbgtc.loader.LoadSpikeTimes.get_spiketimes_superset` |
    |                               |      -  also :meth:`~analyseur.cbgtc.loader.LoadSpikeTimes.get_spiketimes_subset`               |
    |                               | - `sampling_rate` [OPTIONAL]: `1000/dt = 10000 Hz` [default]                                    |
    |                               | - `window` [OPTIONAL]: Tuple `(0, 10) seconds` [default]                                        |
    |                               | - `neurons` [OPTIONAL]: "all" [default] or a scalar or list: range(a, b) or [1, 4, 5, 9]        |
    |                               | - `resolution` [OPTIONAL]: `~ 9.76 Hz = sampling_rate/1024` [default]                           |
    +-------------------------------+-------------------------------------------------------------------------------------------------+
    | :py:meth:`.compute_for_rate`  | - `mu_rate_array`: see :meth:`~analyseur.cbgtc.stats.rate.Rate.mean_rate`                       |
    |                               | - `binsz`: integer or float                                                                     |
    |                               | - `method`: "welch" or "fft" or "fft-mag"                                                       |
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

        from analyseur.cbgtc.loader import LoadSpikeTimes
        from analyseur.cbgtc.stats.psd import PowerSpectrum

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
    def compute_for_spike(cls, spiketimes_set, sampling_rate=None,
                          window=None, neurons=None, resolution=None):
        """
        Returns the power spectral density (or power spectrum) of spiking times from all given neurons.

        .. math::

            P_i(f) = \\frac{1}{K} \\sum_{k=1}^K \\left[\\frac{1}{L}\\left|\\sum_{n=0}^{L-1}s_i(n)\\cdot w(n) \\cdot e^{-i2\\pi f n}\\right|^2\\right]

        is the Welch's estimate of the power spectral density :math:`P_i(f)` with :math:`w(n)` window, :math:`K` number of segments, :math:`L` *nperseg*, and the binary spike train :math:`s_i(t_k)` (see :meth:`~analyseur.cbgtc.curate.get_binary_spiketrains`).

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgtc.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgtc.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param sampling_rate: `1000/dt = 10000` Hz [default]; sampling_rate ∊ (0, 10000)
        :param window: Tuple in the form `(start_time, end_time)`; `(0, 10)` [default]
        :param neurons: `"all"` [default] or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: a tuple in the following order
        - array of sample frequencies
        - power spectral density (or power spectrum)
        - list of spike trains
        - list of neuron id's
        - array of time

        .. list-table:: **Notes on resolution**
            :widths: auto
            :header-rows: 0

            * - ``resolution`` or desired frequency resolution is the smallest difference between two frequencies that can be distinguished in the power spectrum.
            * - The sampling rate is proportional to the desired frequency resolution.

        .. math::

            \\text{sampling\\_rate} &\\propto \\text{resolution} \n
            \\text{sampling\\_rate} &= \\text{nperseg} \\times \\text{resolution}

        where the constant of proportionality is the number of points per segment `nperseg`.

        .. list-table::
            :widths: auto
            :header-rows: 1

            * - Analysis Pitfalls
            * - :math:`P_i(f)` returns one PSD per neuron
            * - plotting PSD for just one neuron will mostly be Poisson noise
            * - the **meaningful** quantity is :math:`P_\\text{pop}(f) = \\frac{1}{N}\\sum_{i=1}^{N}P_i(f)`
            * - :math:`P_\\text{pop}(f)` is used in :meth:`~analyseur.cbgtc.visual.powspec.VizPSD.plot_aggstat`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        #============== DEFAULT Parameters ==============
        sampling_frequency = 1 / cls.__siganal.sampling_period
        if sampling_rate is None:
            sampling_rate = sampling_frequency
        elif sampling_rate > sampling_frequency:
            print("sampling_rate > " + f"{sampling_frequency} ∴ sampling_rate = " + f"{sampling_frequency}")
            sampling_rate = sampling_frequency

        if window is None:
            window = cls.__siganal.window

        if neurons is None:
            neurons = "all"
        elif isinstance(neurons, numbers.Number):
            neurons = range(neurons)

        if resolution is None:
            points_per_segment = 1024
        else:
            points_per_segment = int(sampling_rate / resolution)

        # Spike times > Spike Train
        [spiketrains, yticks, time_axis] = get_binary_spiketrains(spiketimes_set, sampling_rate=sampling_rate,
                                                                  window=window, neurons=neurons)

        # Calculate Power Spectra (a.k.a power spectral density, PSD)
        frequencies = []
        power_spectra = []

        for i, spike_train in enumerate(spiketrains):
            # since s(t)∈{0,1}, at f = 0 the DC will have a large component
            spike_train = spike_train - spike_train.mean()

            f, Pxx = signal.welch(spike_train, fs=sampling_rate, nperseg=points_per_segment)
            frequencies.append(f)
            power_spectra.append(Pxx)

        return frequencies, power_spectra, spiketrains, yticks, time_axis

    @classmethod
    def compute_for_rate(cls, mu_rate_array, binsz, resolution=None, method=None):
        """
        Returns the power spectral density (or power spectrum) of firing rate from all given neurons.

        .. math::

            P_r(f) = \\frac{1}{T} \\left|\\sum_{t=0}^{T-1}r(t)\\cdot e^{-i2\\pi f t}\\right|^2

        is the squared magnitude of the Fourier transform of the population firing rate signal :math:`r(t)`  with duration :math:`T`.

        :param mu_rate_array: array of average firing rates for all/multiple neurons using :meth:`~analyseur.cbgtc.stats.rate.Rate.mean_rate`

        :param binsz: integer or float
        :param method: `"welch"` or `"fft"` or `"fft-mag"`
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: a tuple in the following order
        - array of sample frequencies
        - power spectral density (or power spectrum)


        where the constant of proportionality is the number of points per segment `nperseg`.

        Note that the population rate signal is

        .. math::

            r(t) = \\frac{1}{N}\\sum_{i=1}^N s_i(t)

        where :math:`i`-th neuron has spike train :math:`s_i(t)`. Therefore, using the Fourier transform operator :math:`\\mathcal{F}` the power spectrum can be re-written as

        .. math::

            P_r(f) = \\left|\\mathcal{F}\\{r(t)\\}\\right|^2

        .. list-table::
            :widths: auto
            :header-rows: 1

            * - Population synchrony at oscillatory frequency :math:`f_0`
            * - oscillation :math:`r(t) = A \\cdot sin(2\\pi f_0 t)`
            * - PSD peak, :math:`P_r(f_0)`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        sampling_fs = 1.0 / binsz
        T = 1 / sampling_fs
        n = len(mu_rate_array)
        # ============== DEFAULT Parameters ==============
        # T = cls.__siganal.sampling_period  # seconds
        # sampling_fs = 1 / T  # Hz

        if resolution is None:
            points_per_segment = 1024
        else:
            points_per_segment = int(sampling_fs / resolution)

        # at f = 0 the PSD will have a large DC component
        mu_rate_array = mu_rate_array - np.mean(mu_rate_array)

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
