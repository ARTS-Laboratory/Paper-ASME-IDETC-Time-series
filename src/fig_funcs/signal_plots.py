import scipy as sp
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

import metrics


def plot_signal(time, data):
    fig = plt.figure()
    plt.plot(time, data)
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (m/s^2)')
    return fig


def plot_signal_fft(time, data):
    dft = fft(data)
    freq = fftfreq(len(data), d=metrics.sampling_frequency(time))
    fig = plt.figure()
    plt.plot(freq, dft)
    return fig
