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

class ContinuousWaveletTransform(object):
    """
    Continuous Wavelet Tranform

    ===================================================
    Comments on Activity and Choices in Performing CWT
    ===================================================

    +--------------
    | Activity    | Description  | Purpose
    +=============+
    | Signal Smoothing | convert binary spike trains | - some smoothing can help visualize rhythmic spiking |
    |                  | to continuous signals       | - over smoothing can obscure precise timing          |

    """
    #===============================================================
    # Static methods that check for available individual wavelet options
    # in https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
    #===============================================================
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
        """
        This method checks for wavelet options available in
        `pywt.cwt <https://pywavelets.readthedocs.io/en/latest/ref/cwt.html>`_.
        """
        pywt_wavelets1 = ["mexh", "morl", ]
        pywt_wavelets2 = ["cmor", "gaus", "cgau", "shan", "fbsp"]

        if wavelet is not pywt_wavelets1:
            is_any_pywt_wavelet = [getattr(cls, "_is_" + name + "_format")(wavelet) for name in pywt_wavelets2]
            if not any(is_any_pywt_wavelet):
                raise ValueError("wavelet must be one of the strings in the list: "
                                 + str(pywt_wavelets1 + pywt_wavelets2))



    @staticmethod
    def smooth_signal(spiketimes_superset, sampling_rate=None,
                      window=None, neurons=None, sigma=None):
        """
        This method takes the spike times and converts it into respective binary spike trains
        which in turn is smoothened. The returned smoothened signal can be used to create a
        firing rate signal.

        Smoothening is done by Gaussian filtering

        - each binary spike (= 1) is placed into a Gaussian-shaped bump
        - the Gaussian filter replaces each point (spike = 1) with a Gaussian distribution centered at that position
            - the overall result is many Gaussian distributions (each per point when spike = 1)
            - where spikes are close the Gaussians overlap
        - sum together the overlaping Gaussians (i.e convolution)
            - this represents the "dense estimate" of the spikes, i.e, smoothened curve

        **Formula**

        .. table:: Formula
        =========================================================================================================================== ======================================================
          Definitions                                                                                                                         Interpretation
        =========================================================================================================================== ======================================================
         total neurons, :math:`n_{nuc}`                                                                                                       total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                                                              i-th neuron in the pool of :math:`n_{Nuc}` neurons
         total spikes, :math:`n_{spk}^{(i)}`                                                                                                  total number of spikes (spike times) by i-th neuron
         :math:`\\vec{S}^{(i)}`                                                                                                               array of spike times of i-th neuron
         :math:`S = \\left\\{\\vec{S}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`                                                   set of spike times of all neurons
         :math:`\\vec{B}^{(i)} = \\left[\\sum_{k=1}^{n_{spk}^{(i)}} \\delta[t - t_k]\\right]_{\\forall{t} \\in [t_1, t_{n_{spk}^{(i)}}]}`     binary spike train of i-th neuron for spike times :math:`\\vec{S}^{(i)}` at :math:`t_1, t_2, ..., t_{n_{spk}^{(i)}}`
         :math:`B = \\left\\{\\vec{B}^{(i)} \\mid \\forall{i \\in [1, n_{nuc}]} \\right\\}`                                                   set of spike trains of all neurons
         :math:`\\vec{G} = \\left[\\frac{1}{\\sigma \\sqrt{2\\pi}}e^{-\\frac{k^2}{2\\sigma^2}}\\right]_{\\forall{k}}`                         Gaussian kernel
        =========================================================================================================================== ======================================================

        Then, the smoothened signal for i-th neuron is

        .. math::

            \\vec{M}^{(i)} &= \\vec{B}^{(i)} \\ast \\vec{G} \n
                           &\\triangleq \\left[ \\sum_{k=1}^{n_{spk}^{(i)}} \\left(B^{(i)}[t_k] \cdot G[t-t_k]\\right) \\right]_{\\forall{t} \\in [t_1, t_{n_{spk}^{(i)}}]} \n
                           &= \\left[ \\sum_{k=1}^{n_{spk}^{(i)}} G[t-t_k] \\right]_{\\forall{t} \\in [t_1, t_{n_{spk}^{(i)}}]}

        """
        # ============== DEFAULT Parameters ==============
        if sampling_rate is None:
            sampling_rate = 1 / spikeanal.sampling_period

        sampling_period = 1.0 / sampling_rate

        if window is None:
            window = spikeanal.window

        if neurons is None:
            neurons = "all"

        if sigma is None:
            sigma = spikeanal.std_Gaussian_kernel

        # Convert spike times to spike train
        [spiketrains, yticks, time_axis] = get_binary_spiketrains(spiketimes_superset, sampling_rate=sampling_rate,
                                                                  window=window, neurons=neurons)
        # Return the smoothened spike train
        return gaussian_filter1d(spiketrains, sigma=sigma), yticks, time_axis, sampling_period


    @classmethod
    def _compute_cwt_single(cls, spiketimes_superset, sampling_rate=None,
                           window=None, sigma=None,
                           scales=None, wavelet=None, neuron_indx=None,):
        """
        Compute the Continuous Wavelet Transform for a single neuron
        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = spikeanal.window

        if sigma is None:
            sigma = spikeanal.std_Gaussian_kernel

        if scales is None:
            scales = np.arange(1, 128)
        else:
            scales = np.arange(scales[0], scales[1])

        if wavelet is None:
            wavelet = "cmor1.5-1.0"

        # Check wavelet chosen is one of available option in https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
        cls._check_pywt_wavelet_format(wavelet)

        # Convert spike times to spike trains
        [smoothed_signal, yticks, time_axis, sampling_period] = \
            cls.smooth_signal(spiketimes_superset, sampling_rate=sampling_rate,
                              window=window, neurons="all", sigma=sigma)

        # ============== DEFAULT Parameters ==============
        if neuron_indx is None:
            neuron_indx = np.random.randint(0, high=len(yticks))

        # Check wavelet chosen is one of available option in https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
        cls._check_pywt_wavelet_format(wavelet)

        single_neuron_train = smoothed_signal[neuron_indx,:].flatten()

        coefficients, frequencies = pywt.cwt(single_neuron_train, scales, wavelet, sampling_period=sampling_period)

        # Return the results for a single neuron
        return coefficients, frequencies, yticks[neuron_indx], time_axis

    @classmethod
    def compute_cwt_avg(cls, spiketimes_superset, sampling_rate=None,
                        window=None, neurons=None, sigma=None,
                        scales=None, wavelet=None, ):
        """
        Compute the Continuous Wavelet Transform of a single neuron
        """
        # ============== DEFAULT Parameters ==============
        if neurons is None:
            neurons = "all"

        if window is None:
            window = spikeanal.window

        if sigma is None:
            sigma = spikeanal.std_Gaussian_kernel

        if scales is None:
            scales = np.arange(1, 128)
        else:
            scales = np.arange(scales[0], scales[1])

        if wavelet is None:
            wavelet = "cmor1.5-1.0"

        # Check wavelet chosen is one of available option in https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
        cls._check_pywt_wavelet_format(wavelet)

        # Convert spike times to spike trains
        [smoothed_signal, yticks, time_axis, sampling_period] = \
            cls.smooth_signal(spiketimes_superset, sampling_rate=sampling_rate,
                              window=window, neurons=neurons, sigma=sigma)

        # Compute the Continuous Wavelet Transform for every neuron within the chosen neurons option "all" or selective
        all_coefficients = []

        for i in range(smoothed_signal.shape[0]):
            single_neuron_train = smoothed_signal[i, :].flatten()
            coefficients, frequencies = pywt.cwt(single_neuron_train, scales, wavelet, sampling_period=sampling_period)
            all_coefficients.append(np.abs(coefficients))

        # Return the results as average coefficients across chosen neurons
        return np.mean(all_coefficients, axis=0), frequencies, yticks, time_axis

    @classmethod
    def compute_cwt_sum(cls, spiketimes_superset, sampling_rate=None,
                        window=None, neurons=None, sigma=None,
                        scales=None, wavelet=None, ):
        # ============== DEFAULT Parameters ==============
        if neurons is None:
            neurons = "all"

        if window is None:
            window = spikeanal.window

        if sigma is None:
            sigma = spikeanal.std_Gaussian_kernel

        if scales is None:
            scales = np.arange(1, 128)
        else:
            scales = np.arange(scales[0], scales[1])

        if wavelet is None:
            wavelet = "cmor1.5-1.0"

        # Check wavelet chosen is one of available option in https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
        cls._check_pywt_wavelet_format(wavelet)

        # Convert spike times to spike trains
        [smoothed_signal, yticks, time_axis, sampling_period] = \
            cls.smooth_signal(spiketimes_superset, sampling_rate=sampling_rate,
                              window=window, neurons=neurons, sigma=sigma)

        # Compute Population Firing Rate from the sum of all chosen neurons
        population_train = np.sum(smoothed_signal, axis=0).flatten()

        coefficients, frequencies = pywt.cwt(population_train, scales, wavelet, sampling_period=sampling_period)

        # Return Population CWT across chosen neurons
        return coefficients, frequencies, yticks, time_axis