from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import matplotlib.pyplot as plt

from utils import metrics


def plot_signal(time, data, ms=False):
    fig = plt.figure()
    ylabel = 'acceleration (m/s\u00b2)'
    if ms:
        xlabel = 'time (\u03bcs)'
        plt.plot(time * 100, data)
    else:
        xlabel = 'time (s)'
        plt.plot(time, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
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
