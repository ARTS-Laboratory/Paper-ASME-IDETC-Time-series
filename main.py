import math

from matplotlib import pyplot as plt

from fig_funcs import rupture_changepoint_plots
from fig_funcs.detection_plots import plot_shock, interval_histogram, raw_histogram
from fig_funcs.signal_plots import plot_signal, plot_signal_fft, plot_signal_power_spectrum, signal_with_overlays, \
    power_spectra_sections, signal_with_inset_axes
from offline_detection import bottom_up, binary_segmentation, dynamic_programming
from online_detection.bocpd import bayesian_online_changepoint_detection, bayesian_online_changepoint_detection_v2, \
    get_plot_bocpd, get_bocpd, get_bocpd_windowed
from online_detection.cusum import get_plot_cusum, get_cusum_revised, cusum, cusum_alg
from online_detection.dni import get_plot_dni
from online_detection.expect_Max import get_plot_expectation_maximization, get_expectation_max, \
    get_expectation_maximization_model_from_generator
import cProfile
import numpy as np

from online_detection.grey_systems_model import get_plot_grey_model, get_grey_model_from_generator, get_grey_model
from utils.matplotlib_formatting import set_rc_params
from utils.read_data import get_data


def make_stacked_power_spectrum_plot(time, data):
    """ """
    power_spectra_sections(time, data)
    # plt.show()


def make_signal_overlay_plot(time, data):
    """ """
    signal_with_inset_axes(time, data, ms=True)
    # signal_with_overlays(time, data)


def make_signal_plots(file_path):
    """ """
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    plt.close('all')
    # Plots for whole signal
    fig = plot_signal(time, data, ms=True)
    plt.savefig('figures/signal_fig.jpg', dpi=350)
    plt.close()
    # normal_fft_fig = plot_signal_fft(time, data)
    # plt.savefig('figures/fft_fig.jpg', dpi=350)
    plt.close()
    normal_per_fig = plot_signal_power_spectrum(time, data)
    plt.savefig('figures/power_spectrum_fig.jpg', dpi=350)
    plt.close()
    # Plots for safe section of signal
    plt.close('all')
    normal_fig = plot_signal(time[:100_000], data[:100_000], ms=True)
    plt.savefig('figures/safe_signal_fig.jpg', dpi=350)
    normal_per_fig = plot_signal_power_spectrum(time[:100_000], data[:100_000])
    plt.savefig('figures/safe_power_spectrum_fig.jpg', dpi=350)
    plt.close()
    # Plots for shock section of signal
    shock_fig = plot_signal(time[200_000:400_000], data[200_000:400_000], ms=True)
    plt.savefig('figures/shock_signal_fig.jpg', dpi=350)
    normal_per_fig = plot_signal_power_spectrum(time[200_000:400_000], data[200_000:400_000])
    plt.savefig('figures/shock_power_spectrum_fig.jpg', dpi=350)
    plt.close()
    # Plots for post shock section of signal
    post_shock_fig = plot_signal(time[400_000:], data[400_000:], ms=True)
    plt.savefig('figures/post_shock_signal_fig.jpg', dpi=350)
    normal_per_fig = plot_signal_power_spectrum(time[400_000:], data[400_000:])
    plt.savefig('figures/post_shock_power_spectrum_fig.jpg', dpi=350)
    # normal_fft_fig_2 = plot_signal_fft(time[:100_000], data[:100_000])
    # normal_per_fig_2 = plot_signal_power_spectrum(time[:100_000], data[:100_000])
    # plt.show()
    # return normal_fig, normal_fft_fig


def plot_signals(file_path):
    """ """
    plt.close('all')
    make_signal_plots(file_path)


def plot_offline_detections(time, data):
    """ """
    num_bkps = 2
    bkps = binary_segmentation.get_breaks(np.abs(data), num_bkps, model_type='rank')
    binary_segmentation.plot_breaks(data, bkps)
    bkps = bottom_up.get_breaks(np.abs(data), num_bkps)
    bottom_up.plot_breaks(data, bkps)
    bkps = dynamic_programming.get_breaks(np.abs(data), num_bkps)
    rupture_changepoint_plots.plot_breaks(data, bkps, show=True)


def plot_bocpd(time, data, show_progress=False):
    """ """
    print('BOCPD detection')
    # Non-windowed version
    mu, kappa, alpha, beta, lamb = np.mean(data[:100]), 0.1, 0.00001, 0.00001, 100
    lamb = 100
    # alpha, beta = 0.1, 0.01
    shock_intervals, non_shock_intervals = get_bocpd(
        time, data, mu, kappa, alpha, beta, lamb, with_progress=show_progress)
    fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.savefig('figures/5-14-2024/bocpd_fig.png', dpi=350)
    plt.close(fig)
    # Windowed version
    mu, kappa, alpha, beta, lamb = np.mean(data[:100]), 0.1, 0.00001, 0.00001, 100
    # mu = 0
    alpha, beta = 1, 1
    # alpha, beta = 0.1, 0.01
    lamb = 100
    shock_intervals_w, non_shock_intervals_w = get_bocpd_windowed(
        time, data, mu, kappa, alpha, beta, lamb, window_size=10, with_progress=show_progress)
    fig_w = plot_shock(time, data, shock_intervals_w, non_shock_intervals_w)
    plt.savefig('figures/5-14-2024/bocpd_windowed_fig.png', dpi=350)
    plt.close(fig_w)
    # cusum_abs_hist = interval_histogram(time, data, cusum_shock, cusum_non_shock)
    # bocpd_raw_hist = raw_histogram(time, data, bocpd_shock, bocpd_non_shock)


def plot_cusum(time, data, show_progress=False):
    """ """
    print('Starting CUSUM Algorithms')
    # transformed_data =
    shock_intervals, non_shock_intervals = get_cusum_revised(time, data, len(data))
    fig_revised = plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.savefig('figures/5-14-2024/cusum_revised_fig.png', dpi=350)
    plt.close(fig_revised)
    mean, std = np.mean(data[:1_000]), np.std(data[:1_000])
    # shock_intervals, non_shock_intervals = simple_cusum(times, data, mean, std)
    # fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    # plt.savefig('figures/5-14-2024/simple_cusum_fig.jpg', dpi=350)
    # shock_intervals, non_shock_intervals = cusum(time, data, mean, std, alpha=0.025, beta=0.025)  # , alpha=0.025, beta=0.025
    # fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    # plt.savefig('figures/5-14-2024/cusum_fig.png', dpi=350)
    shock_intervals, non_shock_intervals = cusum_alg(time, data, mean, std, h=5, alpha=0.25)
    fig_alg = plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.savefig('figures/5-14-2024/cusum_alg_fig.jpg', dpi=350)
    plt.close(fig_alg)
    # cusum_abs_hist = interval_histogram(time, data, cusum_shock, cusum_non_shock)
    # cusum_abs_hist = raw_histogram(time, data, cusum_shock, cusum_non_shock)


def plot_expectation_maximization(time, data, show_progress=False):
    """ """
    print('Starting Expectation Maximization Algorithm')
    my_data = np.abs(data)
    mean_1 = np.mean(my_data[:100_000])
    var_1 = np.var(my_data[:100_000])
    print(np.mean(my_data[200_000:400_000]))
    print(np.std(my_data[200_000:400_000]))
    mean_2, var_2 = 20.0, 10.0
    safe = np.random.normal(mean_1, math.sqrt(var_1), 70)
    unsafe = np.random.normal(mean_2, math.sqrt(var_2), 30)
    shock_intervals_gen, non_shock_intervals_gen = get_expectation_maximization_model_from_generator(
        time, safe, unsafe, data, mean_1=mean_1, var_1=var_1,
        epochs=50, with_progress=show_progress)
    fig_gen = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
    plt.savefig('figures/5-14-2024/expectation_maximization_generator_fig.png', dpi=350)
    plt.close(fig_gen)
    shock_intervals, non_shock_intervals = get_expectation_max(
        time, safe, unsafe, data, mean_1=mean_1, var_1=var_1,
        epochs=100, with_progress=show_progress)
    fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.savefig('figures/5-14-2024/expectation_maximization_fig.png', dpi=350)
    plt.close(fig)


def plot_grey_model(time, data, show_progress=False):
    """ """
    print('Starting Grey Models')
    window_size, c, c_ratio = 100, 3.0, 0.01  # 1.0
    print('grey model generator')
    shock_intervals_gen, non_shock_intervals_gen = get_grey_model_from_generator(
        time, np.abs(data), window_size=window_size, c=c, c_ratio=10.0,
        with_progress=show_progress)
    gen_fig = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
    # plt.savefig('figures/5-14-2024/grey_model.png', dpi=350)
    print('Grey Systems Modelling')
    shock_intervals, non_shock_intervals = get_grey_model(
        time, np.abs(data), window_size=window_size, c=c, c_ratio=c_ratio,
        with_progress=show_progress)
    fig = plot_shock(time, data, shock_intervals, non_shock_intervals)


def plot_detections(file_path):
    """ """
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    # DNI
    # dni_shock, dni_non_shock, dni_fig = get_plot_dni(file_path)
    # dni_abs_hist = interval_histogram(time, data, dni_shock, dni_non_shock)
    # dni_abs_hist = raw_histogram(time, data, dni_shock, dni_non_shock)
    # BOCPD
    # plot_bocpd(time, data, True)
    # bocpd_shock, bocpd_non_shock, bocpd_fig = get_plot_bocpd(file_path, with_progress=True)
    # cusum_abs_hist = interval_histogram(time, data, cusum_shock, cusum_non_shock)
    # bocpd_raw_hist = raw_histogram(time, data, bocpd_shock, bocpd_non_shock)
    # plt.savefig('figures/bocpd_fig.png', dpi=350)
    # _, _, dni_fig = get_plot_dni(file_path)
    # plt.savefig('figures/dni_fig.jpg', dpi=350)
    # CUSUM
    # plot_cusum(time, data, True)
    # cusum_shock, cusum_non_shock, cusum_fig = get_plot_cusum(file_path)
    # cusum_abs_hist = interval_histogram(time, data, cusum_shock, cusum_non_shock)
    # cusum_abs_hist = raw_histogram(time, data, cusum_shock, cusum_non_shock)
    # plt.savefig('figures/cusum_revised_fig.png', dpi=350)
    # plt.savefig('figures/simple_cusum_fig.jpg', dpi=350)
    # plt.savefig('figures/cusum_fig.png', dpi=350)
    # plt.savefig('figures/cusum_alg_fig.jpg', dpi=350)
    # EM
    # plot_expectation_maximization(time, data, True)
    # em_shock, em_non_shock, exp_max_fig = get_plot_expectation_maximization(file_path, True)
    # em_abs_hist = interval_histogram(time, data, em_shock, em_non_shock)
    # em_abs_hist = raw_histogram(time, data, em_shock, em_non_shock)
    # plt.savefig('figures/expectation_maximization_fig.png', dpi=350)
    # Grey Systems
    plot_grey_model(time, data, True)
    # grey_shock, grey_non_shock, grey_fig = get_plot_grey_model(file_path, with_progress=True)
    # grey_fig = plot_shock(time, data, grey_shock, grey_non_shock)
    # grey_abs_hist = interval_histogram(time, data, grey_shock, grey_non_shock)
    # grey_abs_hist = raw_histogram(time, data, grey_shock, grey_non_shock)
    # plt.savefig('', dpi=350)


def profile():
    cProfile.run('main()', sort='ncalls')


def main():
    set_rc_params()
    file_path = './data/Dataset-7-forced-vibration-and-shock/data/dataset-A/inputData1_raw.txt'
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    plot_offline_detections(time, data)
    # make_signal_overlay_plot(time, data)
    # make_stacked_power_spectrum_plot(time, data)
    plt.show()
    # temp()
    # plot_signals(file_path)
    # plot_detections(file_path)
    plt.show()


if __name__ == '__main__':
    # profile()
    main()
