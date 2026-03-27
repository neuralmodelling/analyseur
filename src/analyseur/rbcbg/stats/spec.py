"""
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
    from analyseur.rbcbg.stats.spec import compute_spectrogram

1.2. Load file and get the firing rates
```````````````````````````````````````
::

    loadFR = LoadRates("GPiSNr_model_9_percent_0.csv")
    t_sec, rates_Hz = loadFR.get_rates()

---------
2. Cases
---------

2.1. Spectogram for a desired resolution
````````````````````````````````````````
::

    freq_arr, time_arr, power_arr = compute_spectrogram(rates_Hz, resolution=10)

done for resolution 10 Hz.

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">
"""

import scipy
import numpy as np

from analyseur.rbcbg.parameters import SignalAnalysisParams

siganal = SignalAnalysisParams()

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

        w[n] = 0.5\\left(1 - cos\\left\\frac{2\\pi n}{N-1}\\right)\\right)

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

def compute_spectrogram(rates_Hz, resolution=None):
    """
    Compute spectrogram using STFT (see :func:`compute_stft`)

    :param rates_Hz: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
    :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    sampling_rate = 1 / siganal.sampling_period

    if resolution is None:
        points_per_segment = 1024
    else:
        points_per_segment = sampling_rate / resolution

    noverlap_points_between_segments = points_per_segment // 2

    f, t_spec, Sxx_db = compute_stft(rates_Hz, sampling_rate,
                                     nperseg=points_per_segment,
                                     noverlap=noverlap_points_between_segments)

    return f, t_spec, Sxx_db  # freq, time, power

# def compute_band_powers():
#     pass

