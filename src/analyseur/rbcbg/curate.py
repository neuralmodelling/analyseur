# ~/analyseur/rbcbg/curate.py
#
# Documentation by Lungsi 18 Nov 2025
#

import re

import numpy as np
import scipy

from analyseur.rbcbg.parameters import SignalAnalysisParams

siganal = SignalAnalysisParams()

# ==========================================
# filter_rates
# ==========================================

def __extract_channel_no(channel_id):
    """
    Given a channel_id

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    match = re.search(r'c(\d+)', channel_id)
    return int(match.group(1))

def filter_rates(times_sec=None, rates_Hz=None, window=None):
    """
    Returns the times (s) and rates (Hz) filtered within the desired window.

    :param times_sec: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
    :param rates_Hz: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
    :param window: 2-tuple; (0, 10) [default]
    :return: 2-tuple; filtered_times, filtered_rates

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = siganal.window

    i_start = int(window[0] * siganal._1000ms)
    i_end = int(window[1] * siganal._1000ms) + 1

    filtered_t = times_sec[i_start:i_end]
    filtered_rates = rates_Hz[i_start:i_end]

    return filtered_t, filter_rates

def __filter_rates_set(rates_set=None, window=None):
    """
    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = siganal.window

    filtered_set = {}
    for c_id, rates in rates_set.items():
        filtered_set[c_id] = filter_rates(rates_array=rates, window=window)

    return filtered_set

def detrend_rates(rates_Hz):
    """
    Detrend the rates by removing the linear trend (but the time points stay exactly the same).

    :param rates_Hz: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
    :return: array

    For example:

    ========== =========== ==============
     time (s)   rate (Hz)   detrend rate
    ========== =========== ==============
     0.000       5           -1
     0.001       6            0
     0.002       7            1
    ========== =========== ==============

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    rates_deterended = scipy.signal.detrend(rates_Hz, type="linear")

    return rates_deterended

def apply_highpass_filter(detrended_rates, fs, hp_freq):
    """
    High-pass filter to remove slow drifts, i.e. remove noise (low frequencies).

    :param detrended_rates: array returned using :py:meth:`.detrend_rates`
    :param fs: scalar value for frequency
    :param hp_freq: scalar value for high-pass cutoff
    :return: array

    This applies a zero-phase high-pass Butterworth filter to remove low-frequency components.

    This function attenuates slow drifts and low-frequency noise from the input signal
    using a 4th-order Butterworth high-pass filter. The filtering is performed with
    forward-backward filtering (`filtfilt`), ensuring zero phase distortion.

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    if hp_freq > 0:
        nyquist = fs / 2

        b, a = scipy.signal.butter(4, hp_freq/nyquist, btype="high")

        return scipy.signal.filtfilt(b, a, detrended_rates)
    else:
        return detrended_rates


def zscore_normalize(filtered_rates):
    """
    Normalize firing rates using z-score standardization, z = (x - mean) / std.

    :param filtered_rates: array returned using :py:meth:`.apply_highpass_filter`
    :return: array

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    rates_zscored = (filtered_rates - np.mean(filtered_rates)) / np.std(filtered_rates)

    return rates_zscored

def preprocess(rates_Hz=None, highpass_freq=None):
    """
    Preprocesses the given rates.

    :param rates_Hz: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
    :param highpass_freq: float; 0.5 [default]
    :return: array of preprocessed rates

    The preprocessing is done in the following order:

    - detrend the rates (removes linear trend); see :py:meth:`.deterend_rates`
    - filter to remove noise; see :py:meth:`.apply_highpass_filter`
    - normalize using z-score; see :py:meth:`.zscore_normalize`

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    sampling_rate = 1 / siganal.sampling_period # Hz

    if highpass_freq is None:
        highpass_freq = 0.5 # Hz

    rates_deterended = deterend_rates(rates_Hz)
    rates_filtered = apply_highpass_filter(rates_deterended, sampling_rate, highpass_freq)
    rates_zscored = zscore_normalize(rates_filtered)

    return  rates_zscored

