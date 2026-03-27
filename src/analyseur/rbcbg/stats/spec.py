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

def compute_band_powers():
    pass

