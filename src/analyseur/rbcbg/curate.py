# ~/analyseur/rbcbg/curate.py
#
# Documentation by Lungsi 18 Nov 2025
#

import re

import numpy as np

from analyseur.rbcbg.parameters import SignalAnalysisParams

siganal = SignalAnalysisParams()

# ==========================================
# get_desired_spiketimes_subset
# ==========================================

def extract_channel_no(channel_id):
    match = re.search(r'c(\d+)', channel_id)
    return int(match.group(1))

def filter_rates(rates_array=None, window=None):
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = siganal.window

    i_start = int(window[0] * siganal._1000ms)
    i_end = int(window[1] * siganal._1000ms) + 1

    return rates_array[i_start:i_end]

def filter_rates_set(rates_set=None, window=None):
    # ============== DEFAULT Parameters ==============
    if window is None:
        window = siganal.window

    filtered_set = {}
    for c_id, rates in rates_set.items():
        filtered_set[c_id] = filter_rates(rates_array=rates, window=window)

    return filtered_set