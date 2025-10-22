# ~/analyseur/cbgt/stat/wavelet.py
#
# Documentation by Lungsi 22 Oct 2025
#
# This contains function for loading the files
#
import re

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pywt

from analyseur.cbgt.parameters import SpikeAnalysisParams
from analyseur.cbgt.curate import get_binary_spiketrains

spikeanal = SpikeAnalysisParams()

class WaveletTransform(object):

    @staticmethod
    def _is_cmor_format(w): # "cmorB-C" floating points B, C
        pattern = r"^cmor(-?\d+\.?\d*)-(-?\d+\.?\d*)$"
        return bool(re.match(pattern, w))

    @staticmethod
    def _is_shan_format(w):  # "shanB-C" floating points B, C
        pattern = r"^shan(-?\d+\.?\d*)-(-?\d+\.?\d*)$"
        return bool(re.match(pattern, w))

    @staticmethod
    def _is_fbsp_format(w):  # "fbspM-B-C" floating points B, C and integer M
        pattern = r"^fbsp(-?\d+)(-?\d+\.?\d*)-(-?\d+\.?\d*)$"
        return bool(re.match(pattern, w))

    @staticmethod
    def _is_gaus_format(w):  # "gausP" integer P
        pattern = r"^gaus(-?\d+)$"
        return bool(re.match(pattern, w))

    @staticmethod
    def _is_cgau_format(w):  # "cgauP" integer P
        pattern = r"^cgau(-?\d+)$"
        return bool(re.match(pattern, w))

    @classmethod
    def _check_pywt_wavelet_format(cls, wavelet):
        pywt_wavelets1 = ["mexh", "morl", ]
        pywt_wavelets2 = ["cmor", "gaus", "cgau", "shan", "fbsp"]

        if wavelet is not pywt_wavelets1:
            is_any_pywt_wavelet = [getattr(cls, "_is_" + name + "_format")(wavelet) for name in pywt_wavelets2]
            if not any(is_any_pywt_wavelet):
                raise ValueError("wavelet must be one of the strings in the list: "
                                 + str(pywt_wavelets1 + pywt_wavelets2))


    @staticmethod
    def __get_smoothed_signal(spiketimes_superset, sampling_rate=None,
                              window=None, neurons=None, sigma=None):
        if sampling_rate is None:
            sampling_rate = 1 / spikeanal.sampling_period

        sampling_period = 1.0 / sampling_rate

        if window is None:
            window = spikeanal.window

        if neurons is None:
            neurons = "all"

        if sigma is None:
            sigma = spikeanal.std_Gaussian_kernel

        [spiketrains, yticks, time_axis] = get_binary_spiketrains(spiketimes_superset, sampling_rate=sampling_rate,
                                                                  window=window, neurons=neurons)
        return gaussian_filter1d(spiketrains, sigma=sigma), yticks, time_axis, sampling_period


    @classmethod
    def compute_cwt_single(cls, spiketimes_superset, sampling_rate=None,
                           window=None, neurons=None, sigma=None,
                           scales=None, wavelet=None, neuron_indx=None,):
        [smoothed_signal, yticks, time_axis, sampling_period] = \
            cls.__get_smoothed_signal(spiketimes_superset, sampling_rate=sampling_rate,
                                      window=window, neurons=neurons, sigma=sigma)
        if neuron_indx is None:
            neuron_indx = np.random.randint(0, high=len(yticks))

        # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
        if scales is None:
            scales = np.arange(1, 128)
        else:
            scales = np.arange(scales[0], scales[1])

        if wavelet is None:
            wavelet = "cmor1.5-1.0"

        cls._check_pywt_wavelet_format(wavelet)

        single_neuron_train = smoothed_signal[neuron_indx,:].flatten()

        coefficients, frequencies = pywt.cwt(single_neuron_train, scales, wavelet, sampling_period=sampling_period)

        return coefficients, frequencies, yticks[neuron_indx], time_axis

    @classmethod
    def compute_cwt_avg(cls, spiketimes_superset, sampling_rate=None,
                        window=None, neurons=None, sigma=None,
                        scales=None, wavelet=None, ):
        [smoothed_signal, yticks, time_axis, sampling_period] = \
            cls.__get_smoothed_signal(spiketimes_superset, sampling_rate=sampling_rate,
                                      window=window, neurons=neurons, sigma=sigma)

        # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
        if scales is None:
            scales = np.arange(1, 128)
        else:
            scales = np.arange(scales[0], scales[1])

        if wavelet is None:
            wavelet = "cmor1.5-1.0"

        cls._check_pywt_wavelet_format(wavelet)

        all_coefficients = []

        for i in range(smoothed_signal.shape[0]):
            single_neuron_train = smoothed_signal[i, :].flatten()
            coefficients, frequencies = pywt.cwt(single_neuron_train, scales, wavelet, sampling_period=sampling_period)
            all_coefficients.append(np.abs(coefficients))

        return np.mean(all_coefficients, axis=0), frequencies, yticks, time_axis

    @classmethod
    def compute_cwt_sum(cls, spiketimes_superset, sampling_rate=None,
                        window=None, neurons=None, sigma=None,
                        scales=None, wavelet=None, ):
        [smoothed_signal, yticks, time_axis, sampling_period] = \
            cls.__get_smoothed_signal(spiketimes_superset, sampling_rate=sampling_rate,
                                      window=window, neurons=neurons, sigma=sigma)

        # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
        if scales is None:
            scales = np.arange(1, 128)
        else:
            scales = np.arange(scales[0], scales[1])

        if wavelet is None:
            wavelet = "cmor1.5-1.0"

        cls._check_pywt_wavelet_format(wavelet)

        population_train = np.sum(smoothed_signal, axis=0).flatten()

        coefficients, frequencies = pywt.cwt(population_train, scales, wavelet, sampling_period=sampling_period)

        return coefficients, frequencies, yticks, time_axis