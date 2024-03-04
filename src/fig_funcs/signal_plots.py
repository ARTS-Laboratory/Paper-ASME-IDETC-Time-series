from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

from utils import metrics


def plot_signal(time, data, ms=False):
    fig = plt.figure()
    ylabel = 'acceleration (m/s\u00b2)'
    if ms:
        time *= 100
        xlabel = 'time (\u03bcs)'
    else:
        xlabel = 'time (s)'
    plt.plot(time, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    return fig


def plot_signal_fft(time, data):
    dft = fft(data)
    freq = fftfreq(len(data), d=metrics.sampling_frequency(time))
    fig = plt.figure()
    plt.plot(freq, dft)
    return fig
