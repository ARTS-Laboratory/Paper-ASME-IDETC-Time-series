import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import spectrogram


def get_sampling_freq(time):
    return len(time) / (time[-1] - time[0])


def get_spectrogram(time, data, **kwargs):
    """ """
    print(f'Time shape: {time.shape}, Data shape: {data.shape}')
    fs = get_sampling_freq(time)
    print(f'The sampling rate is {fs} samples/sec')
    window = ('tukey', 0.25)
    window = 'hamming'
    # window = ('gaussian', 0.0005)
    f, t, sxx = spectrogram(data, fs, window=window, nperseg=50_000, noverlap=49_500)
    # sxx_db = 10 * np.log10(sxx)
    # db_min, db_max = -32, None
    return sxx, t, f
    # # The plot begins
    # fig = plt.figure()
    # if to_ms:
    #     plt.pcolormesh(t * 1_000, f / 1_000, sxx_db, shading='nearest')
    #     x_label = 'time (ms)'
    #     y_label = 'frequency (kHz)'
    # else:
    #     plt.pcolormesh(t, f, sxx_db, shading='nearest')
    #     x_label = 'time (s)'
    #     y_label = 'frequency (Hz)'
    # # plt.plot(time, data)
    # plt.ylabel(y_label)
    # plt.xlabel(x_label)
    # plt.colorbar()
    # plt.clim(db_min, db_max)
    # # plt.show()
    # return fig


def plot_spectrogram(sxx, times, freqs, to_ms=False, to_db=False):
    """ """
    # print(f'Time shape: {times.shape}, Frequency shape: {freqs.shape}, spectrum shape: {sxx.shape}')
    fig = plt.figure()
    temp_sxx = sxx if not to_db else 10 * np.log10(sxx)
    freq_lim = 2_500
    freq_indices = freqs <= freq_lim
    if to_ms:
        plt.pcolormesh(times * 1_000, freqs[freq_indices] / 1_000, temp_sxx[freq_indices, :], shading='nearest')
        x_label = 'time (ms)'
        y_label = 'frequency (kHz)'
    else:
        plt.pcolormesh(times, freqs[freq_indices], temp_sxx[freq_indices, :], shading='nearest')
        x_label = 'time (s)'
        y_label = 'frequency (Hz)'
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.colorbar(label='power (m/s\u00b2)\u00b2 (dB)')
    if to_db:
        db_min, db_max = -48, None
        plt.clim(db_min, db_max)
    return fig
