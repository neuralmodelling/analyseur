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
    of the firing rates for all the neurons.

    +----------------------------------+-------------------------------------------------------------------------------------------------+
    | Methods                          | Argument                                                                                        |
    +==================================+=================================================================================================+
    | :py:meth:`.compute_spectrogram`  | - `mu_rate_array`: see :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`                      |
    |                                  | - `resolution` [OPTIONAL]: `~ 9.76 Hz = sampling_rate/1024` [default]                           |
    +----------------------------------+-------------------------------------------------------------------------------------------------+
    | :py:meth:`.compute_for_rate`     | - `mu_rate_array`: see :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`                      |
    |                                  | - `binsz`: integer or float                                                                     |
    |                                  | - `method`: "welch" or "fft" or "fft-mag"                                                       |
    |                                  | - `resolution` [OPTIONAL]: `~ 9.76 Hz = sampling_rate/1024` [default]                           |
    +----------------------------------+-------------------------------------------------------------------------------------------------+

    - :py:meth:`.compute_spectrogram` computes a time-varying power spectrum, that is, power for each time window and hence the power spectrum evolving over time.
    - :py:meth:`.compute_for_rate` computes a single (global) power spectral density, that is, power averaged over the whole signal (no time dimension) and hence one summary spectrum for the entire signal.

    Therefore, :py:meth:`.compute_spectrogram` retains the time structure while :py:meth:`.compute_for_rate` collapses time by averaging.

    =========
    Use Cases
    =========

    -----------------
    1. Pre-requisites
    -----------------

    1.1. Import Modules
    ````````````````````
    ::

        from analyseur.rbcbg.loader import LoadRates
        from analyseur.rbcbg.stats.psd import PowerSpectrum

    1.2. Load file and get the firing rates
    ```````````````````````````````````````
    ::

        loadFR = LoadRates("GPiSNr_model_9_percent_0.csv")
        t_sec, rates_Hz = loadFR.get_rates()

    ---------
    2. Cases
    ---------

    2.1. Time-varying power spectrum for a desired resolution
    `````````````````````````````````````````````````````````
    ::

        freq_arr, time_arr, power_arr = PowerSpectrum.compute_spectrogram(rates_Hz, resolution=10)

    done for resolution 10 Hz.

    2.2. Global power spectral density
    ``````````````````````````````````
    ::

        pow_spec = PowerSpectrum.compute_for_rate(mu_rate_array, binsz)

    doe for default resolution and Welch method (default).

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    __siganal = SignalAnalysisParams()


    @classmethod
    def compute_for_rate(cls, mu_rate_array, binsz, resolution=None, method=None):
        """
        Returns the power spectral density (or power spectrum) of firing rate from all.

        .. math::

            \\hat{P}_{xx}(f) = \\frac{1}{K} \\sum_{m=1}^K S(m,f)

        is the average of the spectrogram :math:`S(m,f)` (see :py:meth:`.compute_spectrogram`) over time windows (Welch PSD). Thus, :math:`P(f)` is averaged over :math:`m` while :math:`S(m,f)` depends on time.

        :param mu_rate_array: array of average firing rates for all/multiple neurons using :meth:`~analyseur.cbgtc.stats.rate.Rate.mean_rate`

        :param binsz: integer or float
        :param method: `"welch"` or `"fft"` or `"fft-mag"`
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :return: a tuple in the following order
        - array of sample frequencies
        - power spectral density (or power spectrum)

        From :py:meth:`.compute_spectrogram` substituting for :math:`S(m,f)` the Welch PSD is expanded as

        .. math::

            \\hat{P}_{xx}(f) = \\frac{1}{K} \\sum_{m=1}^K \\frac{1}{f_s||w||^2}} \\cdot \\left|\\sum_{n=0}^{N-1}x[n+m]\\cdot w[n] \\cdot e^{-j2\\pi fn/f_s} \\right|^2

        where :math:`N` is the segment length (`nperseg`), :math:`f_s` is the sampling frequency (`sample_rate`) and :math:`||w||^2 = \\sum_n w[n]^2` is the normalized window.

        Removing the windowing and segmentation we get the FFT PSD

        .. math::

            P_{xx}(f) = \\frac{1}{f_s N} \\cdot \\left|\\sum_{n=0}^{N-1}x[n] \\cdot e^{-j2\\pi fn/f_s} \\right|^2

        This is the power spectrum from the single whole global window.

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

        **NOTE:*** This average power spectrum of the entire signal tells us the *overall frequency content*.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        n = len(mu_rate_array)
        sampling_fs = 1.0 / binsz
        T = 1 / sampling_fs
        # ============== DEFAULT Parameters ==============
        # T = cls.__siganal.sampling_period  # seconds
        # sampling_fs = 1 / T  # Hz

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


    @staticmethod
    def compute_stft(rate_array, sample_rate, nperseg=256, noverlap=128):
        """
        Computes short-time Fourier transform (STFT)

        :param rate_array: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
        :param sample_rate: float
        :param nperseg: int, Length of each segment
        :param noverlap: int, Overlaps between segments
        :return: a tuple in the following order
        - f: frequency vector
        - t: time vector for spectrogram
        - Sxx: spectrogram power (dB)

        **Mathematically**, the array of firing rates :math:`x[n]` is segmented into overlapping windows
        and according to the window position :math:`m` (time index) and angular frequency :math:`\\omega`
        it is Fourier transformed as

        .. math::

            X(m,\\omega) = \\sum_{n=-\\infty}^\\infty x[n]\\cdot w[n-m] \\cdot e^{-j\\omega n}

        where the window function :math:`w[n]` is given by the `Hann window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html>`_

        .. math::

            w[n] = 0.5\\left(1 - cos\\left(\\frac{2\\pi n}{N-1}\\right)\\right)

        where :math:`N` is the segment length (`nperseg`).

        Then the spectrogram is the squared magnitude of the STFT

        .. math::

            S(m,\\omega) &= |X(m,\\omega)|^2 \n
            S(m, f) &= \\frac{1}{f_s||w||^2} \\cdot \\left|\\sum_{n=0}^{N-1}x[n+m]\\cdot w[n] \\cdot e^{-j2\\pi fn/f_s} \\right|^2

        where :math:`f_s` is the sampling frequency (`sample_rate`) and :math:`||w||^2 = \\sum_n w[n]^2` is the normalized window.

        Finally, for :math:`k=0,1,\\ldots,N/2` the frequency bins are

        .. math::

            f_k = \\frac{k}{N} \\cdot f_s

        the time bins are

        .. math::

            t_m = \\frac{m(N-\\text{noverlap})}{f_s}

        and the power is converted to decibels

        .. math::

            S_\\text{dB}(m,f) = 10 log_{10}(S(m,f) + \\epsilon)

        where :math:`\\epsilon = 10^{-10}` is a small constant to avoid :math:`log(0)`

        Note that the no overlap parameter `noverlap` when subtracted from the segment length :math:`N` yields the *hop size* (step size) :math:`= N - \\text{noverlap}`. It acts like a sliding window such that

        .. code-block:: text

            noverlap = 0

            [--------][--------][--------]

        but with the presence of overlap

        .. code-block:: text

            noverlap = 128

            [--------]
                [--------]
                    [--------]

        resulting in more redundancy but better continuity and hence smoother time resolution.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        f, t_spec, Sxx = scipy.signal.spectrogram(rate_array, fs=sample_rate,
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                window="hann",
                                                scaling="density")
        # Convert to dB and avoid log(0)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        return f, t_spec, Sxx_db

    @classmethod
    def compute_spectrogram(cls, rates_Hz, resolution=None):
        """
        Compute spectrogram using STFT (see :meth:`.compute_stft`)

        :param rates_Hz: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]

        **NOTE:*** This time-varying power spectrum tells us *how frequencies evolve*.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        sampling_rate = 1 / cls.__siganal.sampling_period

        if resolution is None:
            points_per_segment = 1024
        else:
            points_per_segment = sampling_rate / resolution

        noverlap_points_between_segments = points_per_segment // 2

        f, t_spec, Sxx_db = cls.compute_stft(rates_Hz, sampling_rate,
                                             nperseg=points_per_segment,
                                             noverlap=noverlap_points_between_segments)

        return f, t_spec, Sxx_db  # freq, time, power
