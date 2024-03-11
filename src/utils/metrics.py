""" Thompson 2022

    Contains metrics used in
"""
import numpy as np
import scipy
from numpy import ndarray
from scipy.signal import periodogram


def nice_fft_score(dataset, synth):
    """ Return FFT score.

        :param dataset: Reference dataset for synthetic data to be compared to. Expects list of vectors.
        :type dataset: np.ndarray
        :param synth: Synthetic data to be compared. Expects list of vectors.
        :type synth: np.ndarray
        :returns: Average minimum RMSE values between reference and synthetic datasets
        :rtype: float
    """
    # Get ffts
    synth_fft, data_fft = scipy.fft.fft(np.asarray(synth)), scipy.fft.fft(np.asarray(dataset))
    length = data_fft.shape[1]
    min_diffs = np.asarray([fft_helper(data_fft, synth) for synth in synth_fft])
    min_diffs = np.sqrt(np.real(min_diffs / length))
    return min_diffs


def fft_helper(comp_arr, item):
    new_diff = np.min(np.sum(np.square(np.real(comp_arr - item)), axis=1))
    return new_diff


def get_fft_score(dataset, synth):
    """ Return FFT score.

        :param dataset: Reference dataset for synthetic data to be compared to
        :type dataset: np.ndarray
        :param synth: Synthetic data to be compared
        :type synth: np.ndarray
        :returns: Average minimum RMSE values between reference and synthetic datasets
        :rtype: float
    """
    # Get ffts
    synth_fft, data_fft = scipy.fft.fft(np.asarray(synth)), scipy.fft.fft(np.asarray(dataset))
    min_diffs = list()
    for synth_obj in synth_fft:
        min_diff = 1e99
        for data_obj in data_fft:
            diff = np.real(data_obj - synth_obj)
            diff = np.square(diff)
            diff = np.average(diff)
            min_diff = min(min_diff, diff)
        min_diffs.append(min_diff)
    min_diffs = np.sqrt(np.real(min_diffs))
    return np.average(min_diffs)


def sampling_frequency(time_vec: ndarray):
    return (time_vec[-1] - time_vec[0]) / time_vec.size

def abs_mean(vector: ndarray) -> ndarray:
    """ Return absolute mean of vector."""
    return np.mean(np.abs(vector), axis=-1)


def rms(vector: ndarray) -> float:
    """ Return root mean square of vector."""
    return np.sqrt(np.mean(vector ** 2, axis=-1))


def skewness(vector: ndarray) -> float:
    """ Return skewness of vector."""
    mean = np.mean(vector, axis=-1)  # precalculate mean for speed
    if vector.ndim == 2:
        mean = np.expand_dims(mean, 1)
    # number of points minus 1 * the standard deviation squared
    denom = (np.size(vector, axis=-1) - 1) * np.std(vector, axis=-1) ** 3
    return np.sum((vector - mean) ** 3, axis=-1) / denom


def kurtosis(vector: ndarray) -> float:
    """ Return kurtosis of vector."""
    mean = np.mean(vector, axis=-1)  # precalculate mean for speed
    if vector.ndim == 2:
        mean = np.expand_dims(mean, 1)
    # number of points minus 1 * the standard deviation squared
    denom = (np.size(vector, axis=-1) - 1) * np.std(vector, axis=-1) ** 4
    return np.sum((vector - mean) ** 4, axis=-1) / denom


def crest_factor(vector: ndarray) -> float:
    """ Return crest factor of vector."""
    return np.max(vector, axis=-1) / rms(vector)


def shape_factor(vector: ndarray) -> float:
    """ Return shape factor of vector."""
    return rms(vector) / abs_mean(vector)


def impulse_factor(vector: ndarray) -> float:
    """ Return impulse factor of vector."""
    return np.max(vector, axis=-1) / abs_mean(vector)


def frequency_center(vector: ndarray, time_step: float) -> float:
    """ Return frequency center of vector."""
    frequencies, power = periodogram(vector, 1/time_step, scaling='spectrum')
    numerator = np.sum(power * frequencies, axis=-1)
    denominator = np.sum(power, axis=-1)
    return numerator / denominator


def root_mean_square_frequency(vector: ndarray, time_step: float) -> float:
    """ Return root mean square frequency."""
    frequencies, power_spectrum = periodogram(vector, 1/time_step, scaling='spectrum')
    return np.sqrt(np.sum(frequencies**2 * power_spectrum, axis=-1)
                   / np.sum(power_spectrum, axis=-1))


def root_variance_frequency(vector: ndarray, time_step: float) -> float:
    """ Return root variance frequency."""
    frequencies, power_spectrum = periodogram(vector, 1 / time_step, scaling='spectrum')
    if vector.ndim == 2:
        frequencies = np.expand_dims(frequencies, 0)
    freq_center = calc_frequency_center(power_spectrum, frequencies)
    if vector.ndim == 2:
        freq_center = np.expand_dims(freq_center, 1)
    return np.sqrt(np.sum(((frequencies - freq_center) ** 2) * power_spectrum, axis=-1)
                   / np.sum(power_spectrum, axis=-1))


def calc_frequency_center(
        power_spectrum: ndarray, frequencies: ndarray) -> float:
    """ Return frequency center of vector."""
    return np.sum(power_spectrum * frequencies, axis=-1) / np.sum(power_spectrum, axis=-1)
    # return np.mean(fft)


def calc_root_mean_square_frequency(
        power_spectrum: ndarray, frequencies: ndarray) -> float:
    """ Return root mean square frequency."""
    return np.sqrt(np.sum(power_spectrum * frequencies ** 2, axis=-1)
                   / np.sum(power_spectrum, axis=-1))
    # return np.mean(np.abs(np.asarray(fft)) ** 2)


def calc_root_variance_frequency(
        power_spectrum: ndarray, frequencies: ndarray) -> float:
    """ Return root variance frequency."""
    freq_center = calc_frequency_center(power_spectrum, frequencies)
    return np.sqrt(np.sum(((frequencies - freq_center) ** 2) * power_spectrum, axis=-1)
                   / np.sum(power_spectrum, axis=-1))


def trac(ref_signal, predicted_signal):
    """ Implementation of TRAC function."""
    # Get absolute minimum of both signals to shift signals to positive
    y_axis_offset = min(np.min(ref_signal), np.min(predicted_signal))
    signal_1, signal_2 = ref_signal + y_axis_offset, predicted_signal + y_axis_offset
    numerator = np.dot(signal_1, signal_2)**2
    denominator = np.dot(signal_1, signal_1)*np.dot(signal_2, signal_2)
    return numerator/denominator


def trac_2d(ref_signals, predicted_signals):
    """ TRAC function for 2d arrays."""
    return np.asarray(
        [trac(ref, pred) for ref, pred in zip(ref_signals, predicted_signals)])

# Copied from https://github.com/ARTS-Laboratory/PART-MLC
# Zhymir Thompson 2023


def rms(signal: ndarray) -> float:
    """ Return root mean square of signal."""
    return np.sqrt(np.mean(np.square(signal)))


def rmse(actual: ndarray, predicted) -> float:
    """ Return root mean square error of vector."""
    return np.sqrt(np.mean(np.square(actual - predicted)))


def snr(ref_signal: ndarray, predicted_signal: ndarray) -> float:
    """ Implementation of signal-to_noise ratio.
    :param ref_signal: Signal to be compared to.
    :type ref_signal: np.ndarray or List
    :param predicted_signal: Signal to compare.
    :type predicted_signal: np.ndarray or List
    :rtype: float
    """
    return 20 * np.log10(rms(ref_signal)/rmse(ref_signal, predicted_signal))


def snr_2d(ref_signals, predicted_signals):
    return np.asarray(
        [snr(ref, pred) for ref, pred in zip(ref_signals, predicted_signals)])

