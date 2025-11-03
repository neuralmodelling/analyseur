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

from analyseur.cbgt.parameters import SignalAnalysisParams
from analyseur.cbgt.curate import get_binary_spiketrains

siganal = SignalAnalysisParams()

class ContinuousWaveletTransform(object):
    """
    ============================
    Continuous Wavelet Transform
    ============================

    +------------------------------+-------------------------------------------------------------------------------------------------------+
    | Methods                      | Argument                                                                                              |
    +==============================+=======================================================================================================+
    | :py:meth:`.compute`          | - `all_neurons_spiketimes`: Dictionary returned; see :class:`~analyseur.cbgt.loader.LoadSpikeTimes`   |
    +------------------------------+-------------------------------------------------------------------------------------------------------+
    | :py:meth:`.inst_rates`       | - `all_neurons_isi`: Dictionary returned; see :py:meth:`.compute`                                     |
    +------------------------------+-------------------------------------------------------------------------------------------------------+
    | :py:meth:`.avg_inst_rates`   | - `all_inst_rates`: Dictionary returned; see :py:meth:`.inst_rates`                                   |
    |                              | - `all_times`: 2nd tuple (Dictionary) returned; see :py:meth:`.compute`                               |
    |                              | - `binsz`: [OPTIONAL] 0.01 (default)                                                                  |
    +------------------------------+-------------------------------------------------------------------------------------------------------+
    | :py:meth:`.mean_freqs`       | - `all_neurons_isi`: Dictionary returned; see :py:meth:`.compute`                                     |
    +------------------------------+-------------------------------------------------------------------------------------------------------+
    | :py:meth:`.grand_mean_freq`  | - `all_neurons_isi`: Dictionary returned; see :py:meth:`.compute`                                     |
    +------------------------------+-------------------------------------------------------------------------------------------------------+

    Comments on Activity and Choices in Performing CWT
    --------------------------------------------------

    +------------------+-------------------------------+-------------------------------------------------------------------------------+
    | Activity         | Description                   | Purpose                                                                       |
    +==================+===============================+===============================================================================+
    | Signal Smoothing | converts binary spike trains  | - some smoothing can help visualize rhythmic spiking                          |
    |                  | to continuous signals         | - over smoothing can obscure precise timing                                   |
    +------------------+-------------------------------+-------------------------------------------------------------------------------+
    | scales           | defines the frequencies       | - smaller scales for high frequencies and larger for lower frequencies        |
    |                  | analyzed                      | - voices per ocatve ≜ number of scales between 2 frequencies (≜ octave)       |
    |                  |                               | - higher voices per octave give smoother scalogram but increased computation  |
    +------------------+-------------------------------+-------------------------------------------------------------------------------+
    | wavelet choice   | determines trade-off between  | - Morlet ("cmorB-C") for oscillation                                          |
    |                  | time and frequency resolution | - Mexican hat ("mexh") for transient spike detection                          |
    +------------------+-------------------------------+-------------------------------------------------------------------------------+

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

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
        Smoothen spike times signal
        ===========================
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

        Note that the last line is due to the fact that only non-zero :math:`B^{(i)}[t_k]` occurs at spike positions.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if sampling_rate is None:
            sampling_rate = 1 / siganal.sampling_period

        sampling_period = 1.0 / sampling_rate

        if window is None:
            window = siganal.window

        if neurons is None:
            neurons = "all"

        if sigma is None:
            sigma = siganal.std_Gaussian_kernel

        # Convert spike times to spike train
        [spiketrains, yticks, time_axis] = get_binary_spiketrains(spiketimes_superset, sampling_rate=sampling_rate,
                                                                  window=window, neurons=neurons)
        # Return the smoothened spike train
        return gaussian_filter1d(spiketrains, sigma=sigma), yticks, time_axis, sampling_period


    @staticmethod
    def scale_to_freq(scale=None, wavelet=None, sampling_rate=None):
        """
        Converts scale to frequency
        ===========================
        Converts a scale value to its corresponding frequency value.

        :param scale: a scalar
        :param wavelet: name of wavelet type available in `pywt.cwt <https://pywavelets.readthedocs.io/en/latest/ref/cwt.html>`_
        :param sampling_rate: [OPTIONAL] `10000` [default]


        +---------------------------------------------------------------------------------------------------------------------+
        | Scale is the dilation/compression factor applied to the wavelet.                                                    |
        +=====================================================================================================================+
        | - **Scale vs Frequency**                                                                                            |
        |    - Scales defines the frequencies in the wavelet transform analysis.                                              |
        |    - :math:`s_a < s_b \\overset{\\frown}{=} f_a < f_b` where scale :math:`s_x` corresponds to frequency :math:`f_x` |
        | - Scale vs Voices/Octave                                                                                            |
        |    - Voices per octave controls the number of scales between consecutive frequencies (≜ octave)                     |
        |    - In other words, the number of scales between octaves is the voices per octave                                  |
        +---------------------------------------------------------------------------------------------------------------------|

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if sampling_rate is None:
            # sampling_rate = 1 / siganal.sampling_period
            sampling_period = siganal.sampling_period
        else:
            sampling_period = 1.0 / sampling_rate

        center_freq = pywt.central_frequency(wavelet)  # Property of a specific wavelet
        frequency = center_freq / (scale * sampling_period)

        return frequency


    @staticmethod
    def freq_window_to_scales(freq_window=None, voices_per_octave=None, wavelet=None, sampling_rate=None):
        """
        Create array of scales
        ======================
        Returns the array of scales between octaves given

        :param freq_window: Tuple `(min_freq, max_freq)`
        :param voices_per_octave: scalar
        :param wavelet: name of wavelet type available in `pywt.cwt <https://pywavelets.readthedocs.io/en/latest/ref/cwt.html>`_
        :param sampling_rate: [OPTIONAL] `10000` [default]

        +-------------------+---------------------------------------+------------------------------------+
        | Parameter         | Description                           | Comment                            |
        +===================+=======================================+====================================+
        | frequency range   | - match signal's expected content     | - 1-100 Hz (neuron spikes)         |
        +-------------------+---------------------------------------+------------------------------------+
        | Voices Per Octave | - VPO ∝ frequency resolution          | - 10-16 (general)                  |
        | (VPO)             | - VPO ∝ 1/computational cost          | - 32 (high precision analysis)     |
        +-------------------+---------------------------------------+------------------------------------+
        | wavelet choice    | - time/frequency resolution trade-off | `"cmorB-C"` (good for neural data) |
        +-------------------+---------------------------------------+------------------------------------+

        +---------------------------------------------------------------------------------------------------------------------+
        | Scale is the dilation/compression factor applied to the wavelet.                                                    |
        +=====================================================================================================================+
        | - Scale vs Frequency                                                                                                |
        |    - Scales defines the frequencies in the wavelet transform analysis.                                              |
        |    - :math:`s_a < s_b \\overset{\\frown}{=} f_a < f_b` where scale :math:`s_x` corresponds to frequency :math:`f_x`    |
        | - **Scale vs Voices/Octave**                                                                                        |
        |    - Voices per octave controls the number of scales between consecutive frequencies (≜ octave)                     |
        |    - In other words, the number of scales between octaves is the voices per octave                                  |
        +---------------------------------------------------------------------------------------------------------------------|

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if sampling_rate is None:
            sampling_rate = 1 / siganal.sampling_period

        min_freq = freq_window[0]
        max_freq = freq_window[1]

        num_of_octaves = np.log2(max_freq / min_freq)
        num_of_scales = int(num_of_octaves * voices_per_octave)

        # Convert frequencies to approximate scales
        min_scale = pywt.frequency2scale(wavelet, min_freq, sampling_period=1.0/sampling_rate)
        max_scale = pywt.frequency2scale(wavelet, max_freq, sampling_period=1.0 / sampling_rate)

        # Generate scales
        scales = np.geomspace(min_scale, max_scale, num=num_of_scales)

        return scales


    @classmethod
    def _compute_cwt_single(cls, spiketimes_superset, sampling_rate=None,
                           window=None, sigma=None,
                           scales=None, wavelet=None, neuron_indx=None,):
        """
        Compute the Continuous Wavelet Transform for a single neuron

        :param spiketimes_superset: Dictionary returned using :class:`~analyseur.cbgt.loader.LoadSpikeTimes`

        OPTIONAL parameters

        :param sampling_rate: `10000` [default]
        :param window: Tuple; `(0, 10)` [default]
        :param sigma: standard deviation value, `2` [default]
        :param wavelet: `"cmor1.5-1.0"` [default], for possible options see `pywt.cwt <https://pywavelets.readthedocs.io/en/latest/ref/cwt.html>`_
        :param scales: array `[1, 2, 3, ..., 128]` [default]
        :param neuron_indx: randomly picks one [default]

        :return: 4-tuple

        - Continuous wavelet transform of the input signal
        - corresponding frequencies
        - time axis of the input signal
        - neuron id

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = siganal.window

        if sigma is None:
            sigma = siganal.std_Gaussian_kernel

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
        return coefficients, frequencies, time_axis, yticks[neuron_indx]

    @classmethod
    def compute_cwt_avg(cls, spiketimes_superset, sampling_rate=None,
                        window=None, neurons=None, sigma=None,
                        scales=None, wavelet=None, ):
        """
        Compute the Continuous Wavelet Transform of a single neuron

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if neurons is None:
            neurons = "all"

        if window is None:
            window = siganal.window

        if sigma is None:
            sigma = siganal.std_Gaussian_kernel

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
        """

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # ============== DEFAULT Parameters ==============
        if neurons is None:
            neurons = "all"

        if window is None:
            window = siganal.window

        if sigma is None:
            sigma = siganal.std_Gaussian_kernel

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