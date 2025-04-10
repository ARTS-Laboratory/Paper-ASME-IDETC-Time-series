from pathlib import Path

from matplotlib import pyplot as plt

from fig_funcs.signal_plots import plot_signal_for_display, plot_signal_fft, plot_signal_power_spectrum, \
    plot_signal_for_flowchart, power_spectra_sections, signal_with_inset_axes_for_paper
from fig_funcs.spectrogram_plots import get_spectrogram, plot_spectrogram_for_display, plot_spectrogram_for_paper
from utils.write_data import save_path


# Constants
SIGNAL_PLOT_SUB_DIR = 'signal_plots'

def make_signal_plots(time, data, save_root=None):
    """ """
    save_dir = Path(save_path(save_root), SIGNAL_PLOT_SUB_DIR)
    my_dpi = 350
    fig_size = (6.5, 2)
    # Plots for whole signal
    fig = plot_signal_for_display(time, data, ms=True, fig_size=fig_size)
    plt.savefig(Path(save_dir, 'signal_fig.png'), dpi=my_dpi)
    plt.close(fig)
    normal_fft_fig = plot_signal_fft(time, data)
    plt.savefig(Path(save_dir, 'fft_fig.png'))
    plt.close(normal_fft_fig)
    normal_per_fig = plot_signal_power_spectrum(time, data)
    plt.savefig(Path(save_dir, 'power_spectrum_fig.png'), dpi=my_dpi)
    plt.close(normal_per_fig)
    # Plots for safe section of signal
    stop = 100_000
    normal_fig = plot_signal_for_display(time[:stop], data[:stop], ms=True)
    plt.savefig(Path(save_dir, 'safe_signal_fig.png'), dpi=my_dpi)
    plt.close(normal_fig)
    normal_per_fig = plot_signal_power_spectrum(time[:stop], data[:stop])
    plt.savefig(Path(save_dir, 'safe_power_spectrum_fig.png'), dpi=my_dpi)
    plt.close(normal_per_fig)
    # Plots for shock section of signal
    start, stop = 200_000, 400_000
    shock_fig = plot_signal_for_display(time[start:stop], data[start:stop], ms=True)
    plt.savefig(Path(save_dir, 'shock_signal_fig.png'), dpi=my_dpi)
    plt.close(shock_fig)
    normal_per_fig = plot_signal_power_spectrum(time[start:stop], data[start:stop])
    plt.savefig(Path(save_dir, 'shock_power_spectrum_fig.png'), dpi=my_dpi)
    plt.close(normal_per_fig)
    # Plots for post shock section of signal
    start = 400_000
    post_shock_fig = plot_signal_for_display(time[start:], data[start:], ms=True)
    plt.savefig(Path(save_dir, 'post_shock_signal_fig.png'), dpi=my_dpi)
    plt.close(post_shock_fig)
    normal_per_fig = plot_signal_power_spectrum(time[start:], data[start:])
    plt.savefig(Path(save_dir, 'post_shock_power_spectrum_fig.png'), dpi=my_dpi)
    plt.close(normal_per_fig)
    start, stop = 100_000, 400_000
    flowchart_time_fig = plot_signal_for_flowchart(time[start:stop], data[start:stop], ms=True)
    plt.savefig(Path(save_dir, 'flowchart_time_fig.png'), dpi=my_dpi)
    plt.close(flowchart_time_fig)
    # normal_fft_fig_2 = plot_signal_fft(time[:100_000], data[:100_000])
    # normal_per_fig_2 = plot_signal_power_spectrum(time[:100_000], data[:100_000])
    # plt.show()
    # return normal_fig, normal_fft_fig


def make_spectrogram_plots(time, data, save_root=None):
    """ Make spectrogram figures for data."""
    save_dir = save_path(save_root)
    sxx, times, freqs = get_spectrogram(time, data)
    display_fig = plot_spectrogram_for_display(sxx, times, freqs, to_ms=True, to_db=True)
    display_fig.savefig(Path(save_dir, 'spectrogram.pdf'))
    display_fig.savefig(Path(save_dir, 'spectrogram.png'), dpi=350)
    plt.close(display_fig)
    paper_fig = plot_spectrogram_for_paper(sxx, times, freqs, to_ms=True, to_db=True)
    paper_fig.savefig(Path(save_dir, 'spectrogram_fig.pdf'))
    paper_fig.savefig(Path(save_dir, 'spectrogram_fig.png'), dpi=350)
    plt.close(paper_fig)

    # def make_stacked_power_spectrum_plot(time, data, save_root=None):
    #     """ Plot different sections of the power spectra for the data."""
    #     save_dir = save_path(save_root)
    #     power_spectra_sections(time, data)
    #     plt.savefig(Path(save_dir, 'stacked_power_spectrum.pdf'))
    #     plt.savefig(Path(save_dir, 'stacked_power_spectrum.png'), dpi=350)
    #     # plt.show()
    #
    # def make_signal_overlay_plot(time, data, save_root=None):
    #     """ """
    #     save_dir = save_path(save_root)
    #     fig = signal_with_inset_axes_for_paper(time, data, ms=True)
    #     plt.savefig(Path(save_dir, 'signal_plots', 'inset_signal.png'), dpi=350)
    #     plt.close(fig)
