from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import matplotlib.pyplot as plt

from utils import metrics


def plot_signal(time, data, ms=False, fig_size=None):
    fig = plt.figure(layout='constrained') if fig_size is None else plt.figure(figsize=fig_size, layout='constrained')
    ylabel = 'acceleration (m/s\u00b2)'
    if ms:
        xlabel = 'time (\u03bcs)'
        plt.plot(time * 100, data)
        plt.xlim([100 * time[0], 100 * time[-1]])
    else:
        xlabel = 'time (s)'
        plt.plot(time, data)
        plt.xlim([time[0], time[-1]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.tight_layout()
    return fig


def plot_signal_fft(time, data):
    dft = fft(data)
    freq = fftfreq(len(data), d=1/metrics.sampling_frequency(time))
    fig = plt.figure()
    plt.semilogy(freq, dft)
    return fig


def plot_signal_power_spectrum(time, data):
    f, den = periodogram(data, metrics.sampling_frequency(time))
    fig = plt.figure()
    plt.semilogy(f, den)
    return fig
