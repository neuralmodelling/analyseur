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

def filter_rates(rates_array=None, window=None):
    """
    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = siganal.window

    i_start = int(window[0] * siganal._1000ms)
    i_end = int(window[1] * siganal._1000ms) + 1

    return rates_array[i_start:i_end]

def filter_rates_set(rates_set=None, window=None):
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

def deterend_rates(rates_set):
    """
    Detrend the rates by removing the linear trend.

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    rates_deterended = {}
    for c_id, rates in rates_set.items():
        rates_deterended[c_id] = scipy.signal.detrend(rates, type="linear")

    return rates_deterended

def apply_highpass_filter(detrended_rates_set, fs, hp_freq):
    """
    High-pass filter to remove slow drifts, i.e. remove noise (low frequencies).

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    if hp_freq > 0:
        nyquist = fs / 2
        rates_filtered = {}
        for c_id, rates in detrended_rates_set.items():
            b, a = scipy.signal.butter(4, hp_freq/nyquist, btype="high")
            rates_filtered[c_id] = scipy.signal.filtfilt(b, a, rates)
    else:
        rates_filtered = detrended_rates_set

    return rates_filtered

def zscore_normalize(filtered_rates_set):
    """
    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    rates_zscored = {}
    for c_id, rates in filtered_rates_set.items():
        rates_zscored[c_id] = (rates - np.mean(rates)) / np.std(rates)

    return rates_zscored

def preprocess(rates_set=None, highpass_freq=None):
    """
    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # ============== DEFAULT Parameters ==============
    sampling_rate = 1 / siganal.sampling_period # Hz

    if highpass_freq is None:
        highpass_freq = 0.5 # Hz

    rates_deterended = deterend_rates(rates_set)
    rates_filtered = apply_highpass_filter(rates_deterended, sampling_rate, highpass_freq)
    rates_zscored = zscore_normalize(rates_filtered)

    return  rates_zscored

