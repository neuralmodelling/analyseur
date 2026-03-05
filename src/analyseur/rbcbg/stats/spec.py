import scipy
import numpy as np

from analyseur.rbcbg.parameters import SignalAnalysisParams

siganal = SignalAnalysisParams()

def compute_stft(rate_array, sample_rate, nperseg=256, noverlap=128):
    """

    :param rate_array: array, Preprocessed rate array
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

def compute_spectrogram(rates_set, resolution=None):
    """
    Compute spectrogram using STFT

    :param rates_set: dictionary of arrays (Preprocessed rates)
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

    freq_set = {}
    time_set = {}
    power_set = {}
    for c_id, rates in rates_set.items():
        f, t_spec, Sxx_db = compute_stft(rates, sampling_rate,
                                         nperseg=points_per_segment,
                                         noverlap=noverlap_points_between_segments)
        freq_set[c_id] = f
        time_set[c_id] = time_set
        power_set[c_id] = Sxx_db

    return freq_set, time_set, power_set

def compute_band_powers():
    pass

