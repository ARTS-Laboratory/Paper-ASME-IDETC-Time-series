from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import matplotlib.pyplot as plt

from utils import metrics


def plot_signal(time, data, ms=False, fig_size=None):
    fig = plt.figure(layout='constrained') if fig_size is None else plt.figure(figsize=fig_size, layout='constrained')
    ylabel = 'acceleration (m/s\u00b2)'
    if ms:
        xlabel = 'time (ms)'
        plt.plot(time * 1000, data)
        plt.xlim([1000 * time[0], 1000 * time[-1]])
    else:
        xlabel = 'time (s)'
        plt.plot(time, data)
        plt.xlim([time[0], time[-1]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig


def plot_signal_fft(time, data):
    dft = fft(data)
    freq = fftfreq(len(data), d=1/metrics.sampling_frequency(time))
    fig = plt.figure()
    plt.semilogy(freq, dft)
    return fig


def plot_signal_power_spectrum(time, data):
    f, den = periodogram(data, 1 / metrics.sampling_frequency(time))
    fig = plt.figure(layout='tight')
    # plt.plot(f, den)
    plt.semilogy(f, den)
    return fig


def plot_overlay(time, data, ms=False, fig_size=None, labels=False, ylim=None):
    fig = plt.figure(layout='compressed') if fig_size is None else plt.figure(figsize=fig_size, layout='compressed')

    if ms:
        if labels:
            xlabel = 'time (\u03bcs)'
        plt.plot(time * 100, data)
        plt.xlim([100 * time[0], 100 * time[-1]])
    else:
        if labels:
            xlabel = 'time (s)'
        plt.plot(time, data)
        plt.xlim([time[0], time[-1]])
    if labels:
        ylabel = 'acceleration (m/s\u00b2)'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    return fig


def signal_with_overlays(time, data):
    """ """
    fig_size = (6.5, 2)
    fig_size_2 = (2.4375, 0.5)

    # Plots for whole signal
    fig = plot_signal(time, data, ms=True, fig_size=fig_size)
    plt.savefig('figures/full_signal.pdf')
    # y_lim = (-15, 15)
    y_lim = (min(min(data[:100_000]), min(data[400_000:])), max(max(data[:100_000]), max(data[400_000:])))
    normal_fig = plot_overlay(time[:100_000], data[:100_000], ms=True, fig_size=fig_size_2, ylim=y_lim)
    plt.savefig('figures/safe_signal_overlay.pdf')
    # plt.close()
    normal_per_fig = plot_overlay(time[200_000:400_000], data[200_000:400_000], ms=True, fig_size=fig_size_2)
    # plt.savefig('figures/shock_overlay.pdf')
    # plt.close()
    post_shock_fig = plot_overlay(time[400_000:], data[400_000:], ms=True, fig_size=fig_size_2, ylim=y_lim)
    plt.savefig('figures/post_shock_overlay.pdf')
    # plt.close()


def signal_with_inset_axes(time, data, ms=True):
    """ """
    fig_size = (6.5, 2)
    fig_size_2 = (2.4375, 0.5)
    fig, ax = plt.subplots(figsize=fig_size, layout='constrained')
    # Plots for whole signal
    ylabel = 'acceleration (m/s\u00b2)'
    if ms:
        xlabel = 'time (ms)'
        plot_time = time * 1_000
    else:
        xlabel = 'time (s)'
        plot_time = time
    plt.plot(plot_time, data, linewidth=1.025)
    plt.xlim([plot_time[0], plot_time[-1]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Insets
    x1, x2 = plot_time[0], plot_time[100_000]
    x3, x4 = plot_time[-100_000], plot_time[-1]
    y1, y2 = -15, 15
    # y1, y2 = (min(min(data[:100_000]), min(data[400_000:])), max(max(data[:100_000]), max(data[400_000:])))
    axins = ax.inset_axes([0.0625, 1-0.275, 0.425, 0.25], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[])
    axins_2 = ax.inset_axes([1-0.435, 1-0.275, 0.425, 0.25], xlim=(x3, x4), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.plot(plot_time, data, linewidth=1.025)
    axins_2.plot(plot_time, data, linewidth=1.025)
    _, lines = ax.indicate_inset_zoom(axins, edgecolor='black')
    lines[0].set_visible(True)
    lines[1].set_visible(False)
    _, lines_2 = ax.indicate_inset_zoom(axins_2, edgecolor='black')
    lines_2[2].set_visible(True)
    lines_2[3].set_visible(False)
    plt.savefig('figures/inset_signal.pdf')
    plt.savefig('figures/inset_signal.png', dpi=350)
    # plt.show()


def plot_power_subplot(ax, time, data):
    f, den = periodogram(data, 1/metrics.sampling_frequency(time))
    # ax.plot(f, den)
    ax.semilogy(f, den)
    ax.set_ylim([1e-14, 1e4])
    ax.grid(True)


def power_spectra_sections(time, data):
    """ """
    fig_size = (6.5, 3)
    fig, (ax_1, ax_2, ax_3) = plt.subplots(
        3, figsize=fig_size, sharex=True, layout='compressed')
    plot_power_subplot(ax_1, time[:100_000], data[:100_000])
    plot_power_subplot(ax_2, time[200_000:400_000], data[200_000:400_000])
    plot_power_subplot(ax_3, time[400_000:], data[400_000:])
    ax_1.set_xlabel('(a)')
    ax_2.set_xlabel('(b)')
    ax_3.set_xlabel('(c)\nfrequency (Hz)')
    ax_2.set_ylabel('PSD (V\u00b2/Hz)')
    # plt.savefig('figures/stacked_power_spectrum_fig.pdf')
    # plt.savefig('figures/stacked_power_spectrum_fig.jpg', dpi=350)
    # plt.savefig('figures/stacked_power_spectrum_fig.png', dpi=350)
    # plt.close()

