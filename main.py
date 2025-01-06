import math
from dataclasses import asdict

import scipy
import cProfile
import sklearn.metrics

import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt
from more_itertools import sliding_window

import Hyperparameters
import utils

from DetectionAlgorithm import DetectionAlgorithm
from fig_funcs import rupture_changepoint_plots
from fig_funcs.detection_plots import plot_shock, interval_histogram, raw_histogram  # , convert_intervals_to_time_memoized
from fig_funcs.histograms import plot_metric_histogram
from fig_funcs.rupture_changepoint_plots import plot_breaks
from fig_funcs.signal_plots import plot_signal, plot_signal_fft, plot_signal_power_spectrum, \
    power_spectra_sections, signal_with_inset_axes
from fig_funcs.spectrogram_plots import get_spectrogram, plot_spectrogram
from offline_detection import bottom_up, binary_segmentation, dynamic_programming
from online_detection.bocpd import get_bocpd_v5_from_generator
from online_detection.cusum import get_plot_cusum, get_cusum_revised, cusum, cusum_alg, cusum_alg_v1
from online_detection.dni import get_plot_dni
from online_detection.expect_Max import get_expectation_maximization_model_from_generator

from online_detection.grey_systems_model import get_grey_model_from_generator, get_grey_model
from online_detection.nonparametric_model import get_nonparametric_model_from_generator
from utils.detection_arr_helpers import convert_interval_indices_to_full_arr, intervals_to_dense_arr
from utils.matplotlib_formatting import set_rc_params
from utils.read_data import get_data
from utils.toml_utils import load_toml
from utils.write_data import save_path


def make_stacked_power_spectrum_plot(time, data, save_root=None):
    """ Plot different sections of the power spectra for the data."""
    save_dir = save_path(save_root)
    power_spectra_sections(time, data)
    plt.savefig(Path(save_dir, 'stacked_power_spectrum.pdf'))
    plt.savefig(Path(save_dir, 'stacked_power_spectrum.png'), dpi=350)
    # plt.show()


def make_signal_overlay_plot(time, data, save_root=None):
    """ """
    save_dir = save_path(save_root)
    fig = signal_with_inset_axes(time, data, ms=True)
    plt.savefig(Path(save_dir, 'signal_plots', 'inset_signal.png'), dpi=350)
    plt.close(fig)


def make_signal_plots(time, data, save_root=None):
    """ """
    save_dir = Path(save_path(save_root), 'signal_plots')
    # Plots for whole signal
    fig = plot_signal(time, data, ms=True)
    plt.savefig(Path(save_dir, 'signal_fig.png'), dpi=350)
    plt.close(fig)
    normal_fft_fig = plot_signal_fft(time, data)
    plt.savefig(Path(save_dir, 'fft_fig.png'))
    plt.close(normal_fft_fig)
    normal_per_fig = plot_signal_power_spectrum(time, data)
    plt.savefig(Path(save_dir, 'power_spectrum_fig.png'), dpi=350)
    plt.close(normal_per_fig)
    # Plots for safe section of signal
    normal_fig = plot_signal(time[:100_000], data[:100_000], ms=True)
    plt.savefig(Path(save_dir, 'safe_signal_fig.png'), dpi=350)
    plt.close(normal_fig)
    normal_per_fig = plot_signal_power_spectrum(time[:100_000], data[:100_000])
    plt.savefig(Path(save_dir, 'safe_power_spectrum_fig.png'), dpi=350)
    plt.close(normal_per_fig)
    # Plots for shock section of signal
    shock_fig = plot_signal(time[200_000:400_000], data[200_000:400_000], ms=True)
    plt.savefig(Path(save_dir, 'shock_signal_fig.png'), dpi=350)
    plt.close(shock_fig)
    normal_per_fig = plot_signal_power_spectrum(time[200_000:400_000], data[200_000:400_000])
    plt.savefig(Path(save_dir, 'shock_power_spectrum_fig.png'), dpi=350)
    plt.close(normal_per_fig)
    # Plots for post shock section of signal
    post_shock_fig = plot_signal(time[400_000:], data[400_000:], ms=True)
    plt.savefig(Path(save_dir, 'post_shock_signal_fig.png'), dpi=350)
    plt.close(post_shock_fig)
    normal_per_fig = plot_signal_power_spectrum(time[400_000:], data[400_000:])
    plt.savefig(Path(save_dir, 'post_shock_power_spectrum_fig.png'), dpi=350)
    plt.close(normal_per_fig)
    # normal_fft_fig_2 = plot_signal_fft(time[:100_000], data[:100_000])
    # normal_per_fig_2 = plot_signal_power_spectrum(time[:100_000], data[:100_000])
    # plt.show()
    # return normal_fig, normal_fft_fig


def make_spectrogram_plots(time, data, save_root=None):
    """ Make spectrogram figures for data."""
    save_dir = save_path(save_root)
    sxx, times, freqs = get_spectrogram(time, data)
    plot_spectrogram(sxx, times, freqs, to_ms=True, to_db=True)
    plt.savefig(Path(save_dir, 'spectrogram.pdf'))
    plt.savefig(Path(save_dir, 'spectrogram.png'), dpi=350)
    plt.close()


def plot_signals(file_path, save_root=None):
    """ """
    save_dir = save_path(save_root)
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    make_signal_plots(time, data, save_root=save_dir)
    make_spectrogram_plots(time, data, save_root=save_dir)


def plot_offline_detections(time, data, save_root=None):
    """ Plot figures for shock detection via offline detection algorithms."""
    save_dir = save_path(save_root)
    num_bkps = 2
    bkps = binary_segmentation.get_breaks(np.abs(data), num_bkps, model_type='rank')
    binary_segmentation.plot_breaks(data, bkps)
    # plt.savefig('./figures/offline_rank_bin-seg.jpg', dpi=350)
    # bkps = bottom_up.get_breaks(np.abs(data), num_bkps)
    # bottom_up.plot_breaks(data, bkps)
    # bkps = dynamic_programming.get_breaks(np.abs(data), num_bkps)
    # rupture_changepoint_plots.plot_breaks(data, bkps, show=True)
    # Make ground truth plot
    ground_shocks_idx, ground_nonshocks_idx = make_ground_truth(time, data)
    ground_shocks = [(time[start], time[stop - 1]) for start, stop in ground_shocks_idx]
    ground_nonshocks = [(time[start], time[stop - 1]) for start, stop in ground_nonshocks_idx]
    print('Shock event start and stop times')
    for start, stop in ground_shocks:
        print(f'Shock event start: {start}, shock event stop: {stop}')
    ground_truth_fig = plot_shock(time, data, ground_shocks, ground_nonshocks)
    plt.savefig(Path(save_dir, 'ground_truth_fig.pdf'))
    plt.savefig(Path(save_dir, 'ground_truth_fig.png'), dpi=350)
    plt.close(ground_truth_fig)


def write_metric_table(time, ground, predictions_list, algorithm_names, signal_names):
    """ """
    ids = (('signal_id', signal_names), ('algorithm', algorithm_names))
    metric_dict = dict(ids)
    # Calculate scores
    metric_names = ('accuracy', 'precision', 'recall', 'f1 score', 'detection delay')
    metric_funcs = (
        sklearn.metrics.accuracy_score, sklearn.metrics.precision_score,
        sklearn.metrics.recall_score, sklearn.metrics.f1_score)
    metrics = dict()
    for metric_name, metric_func in zip(metric_names, metric_funcs):
        scores = [metric_func(ground, predictions) for predictions in predictions_list]
        metrics[metric_name] = scores
    earliest_correct = [get_earliest_correct(time, ground, predictions) for predictions in predictions_list]
    delay = [get_detect_delay(time, ground, earliest) for earliest in earliest_correct]
    metrics['earliest correct'] = earliest_correct
    metrics['delay'] = delay
    # Combine scores and rest of table
    metric_dict.update(metrics)
    df = pd.DataFrame(metric_dict)
    print(df)
    return df


def get_earliest_correct(time, ground, predictions):
    """ """
    true_positive_indices = np.logical_and(ground.astype(bool), predictions.astype(bool))
    if true_positive_indices.any():
        earliest_correct = time[true_positive_indices][0]
    else:
        earliest_correct = np.inf
    return earliest_correct


def get_detect_delay(time, ground: np.ndarray, earliest_correct):
    """ """
    if ground.astype(bool).any():
        first = time[np.nonzero(ground.astype(bool))][0]
        return earliest_correct - first
    return np.inf


def get_scores(time, ground, predictions):
    """ Return metric score for predicted shock prediction given comparison."""
    # Calculate scores
    f1_score = sklearn.metrics.f1_score(ground, predictions)
    precision = sklearn.metrics.precision_score(ground, predictions)
    recall = sklearn.metrics.recall_score(ground, predictions)
    accuracy = sklearn.metrics.accuracy_score(ground, predictions)
    # Price is right score
    true_positive_indices = np.logical_and(ground.astype(bool), predictions.astype(bool))
    if true_positive_indices.any():
        earliest_correct = time[true_positive_indices][0]
        first_ground = time[np.nonzero(ground.astype(bool))][0]
        delay = earliest_correct - first_ground
        print(f'Shock first correctly detected at time: {earliest_correct}')
        print(f'Earliest fault from ground truth: {first_ground}. Found with delay: {delay}')
    else:
        earliest_correct = np.inf
        delay = np.inf
        print('No predictions aligned with ground truth.')
    scores = {
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'detection delay': delay,
        'earliest correct': earliest_correct
    }
    return scores


def print_scores(time, ground, predictions):
    """ Print metric scores for predicted shock prediction given comparison."""
    # Calculate scores
    f1_score = sklearn.metrics.f1_score(ground, predictions)
    precision = sklearn.metrics.precision_score(ground, predictions)
    recall = sklearn.metrics.recall_score(ground, predictions)
    accuracy = sklearn.metrics.accuracy_score(ground, predictions)
    # Print scores
    print(f'F1 score: {f1_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}')
    # Print confusion matrix
    # confusion = sklearn.metrics.confusion_matrix(ground, predictions)
    confusion = sklearn.metrics.confusion_matrix(ground, predictions, normalize='all')
    print(confusion)
    print(sklearn.metrics.classification_report(ground, predictions, digits=3))
    # Price is right score
    # todo we might need to assert that this is not empty before doing indexing, otherwise might break
    true_positive_indices = np.logical_and(ground.astype(bool), predictions.astype(bool))
    if true_positive_indices.any():
        earliest_correct = time[true_positive_indices][0]
        print(f'Shock first correctly detected at time: {earliest_correct}')
    else:
        print('No predictions aligned with ground truth.')


def make_ground_truth(time, data):
    """ Calculates ground truth and returns indices for intervals"""
    shocks, nonshocks = list(), list()
    num_bkps = 2
    bkps = binary_segmentation.get_breaks(np.abs(data), num_bkps, model_type='rank')
    begin = 0
    shocked = False
    print(f'number of breakpoints: {len(bkps)}')
    for bkp in bkps[:-1]:
        if shocked:
            shocks.append((begin, bkp + 1))
        else:
            nonshocks.append((begin, bkp + 1))
        shocked = not shocked
        begin = bkp
    if shocked:
        shocks.append((begin, bkps[-1]))
    else:
        nonshocks.append((begin, bkps[-1]))
    return shocks, nonshocks

    # binary_segmentation.plot_breaks(data, bkps)
    # plt.savefig('./figures/05-21-2024/offline_rank_bin-seg.jpg', dpi=350)


def plot_bocpd(time, data, show_progress=False, ground=None, save_root=None):
    """ """
    # todo make kwargs parser for run configs
    save_dir = save_path(save_root)
    # Generate segments to be used as ground truth
    # todo save ground truth as numpy array file and load it
    (true_shocks, true_nonshocks) = make_ground_truth(time, data)
    ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
    # BOCPD algorithms start running
    # new hyperparameters
    kappa = 100.0
    alpha = kappa * 0.5
    mu = np.mean(data[:100_000])
    beta = alpha * np.var(data[:100_000])
    #old hyperparameters
    # kappa = 50
    # alpha = kappa / 2.0
    # beta = kappa / np.var(data[:kappa])
    # mu = np.mean(data[:kappa])
    # alpha, beta, mu, kappa = 0.5, 1, 0.0, 1.0
    # cache = {time_val: idx for idx, time_val in time}
    print('BOCPD detection')
    # test
    print('Start of test')
    test_shocks, test_nonshocks = get_bocpd_v5_from_generator(
        time, data, mu, kappa, alpha, beta, 100, with_progress=True)
    test_fig_1 = plot_shock(time, data, test_shocks, test_nonshocks)
    plt.savefig(Path(save_dir, 'bocpd_fig.png'), dpi=350)
    # # Evaluation stuff
    pred = intervals_to_dense_arr(time, test_shocks, test_nonshocks)
    print_scores(time, ground, pred)
    # bayesian_online_changepoint_detection_v5(np.abs(data), np.mean(np.abs(data[:100])), 0.1, 1, 100, 100)
    test_shocks, test_nonshocks = get_bocpd_v5_from_generator(
        time, data, np.mean(np.abs(data[:100])), 0.1, 0.1, 0.01, 100,
        with_progress=True)
    test_fig_2 = plot_shock(time, data, test_shocks, test_nonshocks)
    plt.savefig(Path(save_dir, 'bocpd_2_fig.png'), dpi=350)
    # # Evaluation stuff
    pred = intervals_to_dense_arr(time, test_shocks, test_nonshocks)
    print_scores(time, ground, pred)
    print('End of test')
    # plt.show()


def plot_cusum(time, data, show_progress=False, save_root=None):
    """ """
    save_dir = save_path(save_root)
    print('Starting CUSUM Algorithms')
    # transformed_data =
    shock_intervals, non_shock_intervals = get_cusum_revised(time, data, len(data))
    fig_revised = plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.savefig(Path(save_dir, 'cusum_revised_fig.png'), dpi=350)
    plt.close(fig_revised)
    # Evaluation stuff
    (true_shocks, true_nonshocks) = make_ground_truth(time, data)
    pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
    ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
    print_scores(time, ground, pred)
    mean, std = np.mean(data[:100_000]), np.std(data[:100_000])
    # shock_intervals, non_shock_intervals = simple_cusum(times, data, mean, std)
    # fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    # plt.savefig('figures/5-14-2024/simple_cusum_fig.jpg', dpi=350)
    # shock_intervals, non_shock_intervals = cusum(time, data, mean, std, alpha=0.025, beta=0.025)  # , alpha=0.025, beta=0.025
    # fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    # plt.savefig('figures/5-14-2024/cusum_fig.png', dpi=350)
    shock_intervals, non_shock_intervals = cusum_alg(
        time, data, mean, std, h=5, alpha=0.95)  # 0.001
    fig_alg = plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.savefig(Path(save_dir, 'cusum_alg_fig.png'), dpi=350)
    plt.close(fig_alg)
    pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
    print_scores(time, ground, pred)

    shock_intervals, non_shock_intervals = cusum_alg_v1(
        time, data, mean, std, h=5, alpha=0.999)  # 0.001
    fig_alg = plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.savefig(Path(save_dir, 'cusum_alg_v1_fig.png'), dpi=350)
    plt.close(fig_alg)
    pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
    print_scores(time, ground, pred)


def plot_expectation_maximization(time, data, show_progress=False, save_root=None):
    """ """
    save_dir = save_path(save_root)
    print('Starting Expectation Maximization Algorithm')
    my_data = np.abs(data)
    mean_1 = np.mean(my_data[:100_000])
    var_1 = np.var(my_data[:100_000])
    print(np.mean(my_data[200_000:400_000]))
    print(np.std(my_data[200_000:400_000]))
    mean_2, var_2 = 20.0, 10.0
    rng = np.random.default_rng()
    safe = rng.normal(mean_1, math.sqrt(var_1), 70)
    unsafe = rng.normal(mean_2, math.sqrt(var_2), 30)
    # safe = np.random.normal(mean_1, math.sqrt(var_1), 70)
    # unsafe = np.random.normal(mean_2, math.sqrt(var_2), 30)
    shock_intervals_gen, non_shock_intervals_gen = get_expectation_maximization_model_from_generator(
        time, safe, unsafe, data, mean_1=mean_1, var_1=var_1,
        epochs=50, with_progress=show_progress)
    fig_gen = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
    plt.savefig(Path(save_dir, 'expectation_maximization_generator_fig.png'), dpi=350)
    plt.close(fig_gen)
    # Evaluation stuff
    (true_shocks, true_nonshocks) = make_ground_truth(time, data)
    pred = intervals_to_dense_arr(time, shock_intervals_gen, non_shock_intervals_gen)
    ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
    print_scores(time, ground, pred)
    # shock_intervals, non_shock_intervals = get_expectation_max(
    #     time, safe, unsafe, data, mean_1=mean_1, var_1=var_1,
    #     epochs=100, with_progress=show_progress)
    # fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    # plt.savefig(Path(save_dir, 'expectation_maximization_fig.png'), dpi=350)
    # plt.close(fig)
    # Evaluation stuff
    # pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
    # print_scores(time, ground, pred)


def plot_grey_model(time, data, show_progress=False, save_root=None):
    """ """
    save_dir = save_path(save_root)
    print('Starting Grey Models')
    window_size, c, c_ratio = 100, 3.0, 0.01  # 1.0
    print('grey model generator')
    shock_intervals_gen, non_shock_intervals_gen = get_grey_model_from_generator(
        time, np.abs(data), window_size=window_size, c=c, c_ratio=10.0,
        with_progress=show_progress)
    gen_fig = plot_shock(
        time, data, shock_intervals_gen, non_shock_intervals_gen)
    plt.savefig(Path(save_dir, 'grey_model_rel_dist.png'), dpi=350)
    # Evaluation stuff
    (true_shocks, true_nonshocks) = make_ground_truth(time, data)
    pred = intervals_to_dense_arr(time, shock_intervals_gen, non_shock_intervals_gen)
    ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
    print_scores(time, ground, pred)
    print('Grey Systems Modelling')
    shock_intervals, non_shock_intervals = get_grey_model(
        time, np.abs(data), window_size=window_size, c=c, c_ratio=c_ratio,
        with_progress=show_progress)
    fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.savefig(Path(save_dir, 'grey_model.png'), dpi=350)
    # Evaluation stuff
    pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
    print_scores(time, ground, pred)


def plot_nonparametric_model(time, data, show_progress=False, save_root=None):
    """ """
    save_dir = save_path(save_root)
    window_size, crit_value = 60, 1.965
    shock_intervals_gen, non_shock_intervals_gen = get_nonparametric_model_from_generator(
        time, data, window_size, crit_value=crit_value, with_progress=show_progress)
    fig = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
    plt.savefig(Path(save_dir, 'nonparametric.png'), dpi=350)
    # Evaluation stuff
    (true_shocks, true_nonshocks) = make_ground_truth(time, data)
    pred = intervals_to_dense_arr(time, shock_intervals_gen, non_shock_intervals_gen)
    ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
    print_scores(time, ground, pred)




def plot_detections(time, data, save_root=None):
    """ """
    save_dir = save_path(save_root)
    # DNI
    # dni_shock, dni_non_shock, dni_fig = get_plot_dni(file_path)
    # dni_abs_hist = interval_histogram(time, data, dni_shock, dni_non_shock)
    # dni_abs_hist = raw_histogram(time, data, dni_shock, dni_non_shock)
    # BOCPD
    plot_bocpd(time, data, True, save_root=save_dir)
    # CUSUM
    plot_cusum(time, data, True, save_root=save_dir)
    # EM
    plot_expectation_maximization(time, data, True, save_root=save_dir)
    # Grey Systems
    plot_grey_model(time, data, True, save_root=save_dir)
    # Nonparametric
    plot_nonparametric_model(time, data, True, save_root=save_dir)
    # Estimated Ground Truth
    # bkps = binary_segmentation.get_breaks(np.abs(data), 2, model_type='rank')
    # ground_fig = plot_breaks(data, bkps, show=False)
    # plt.savefig(Path(save_dir, 'ground_plot.png'), dpi=350)


def data_transformations(data):
    """ """
    bkps = binary_segmentation.get_breaks(np.abs(data), 2, model_type='rank')
    # code for different data transformations
    n_bins = 256
    left_data, right_data = data[:bkps[0]], data[bkps[0]:bkps[1]]
    raw_fig = plot_metric_histogram(left_data, right_data, num_bins=n_bins)
    abs_fig = plot_metric_histogram(np.abs(left_data), np.abs(right_data), num_bins=n_bins)
    diff_fig = plot_metric_histogram(np.ediff1d(left_data), np.ediff1d(right_data), num_bins=n_bins)
    plt.show()

    window_size = 10
    windows = np.array(list(sliding_window(data, window_size)))
    # windows = np.array(list(sliding_window(np.abs(np.ediff1d(data)), window_size)))

    left_windows = np.array(list(sliding_window(left_data, window_size)))
    right_windows = np.array(list(sliding_window(right_data, window_size)))
    transforms = [
        utils.metrics.abs_mean, utils.metrics.rms, utils.metrics.skewness,
        utils.metrics.kurtosis, utils.metrics.crest_factor,
        utils.metrics.impulse_factor, utils.metrics.shape_factor,
        scipy.stats.iqr
    ]
    for transform in transforms:
        print('Next Transform...')
        # transform_arr = np.array([transform(window) for window in windows])
        # left_transform = transform_arr[:bkps[0]]
        # right_transform = transform_arr[bkps[0]:bkps[1]]
        left_transform = np.array([transform(window) for window in left_windows])
        right_transform = np.array([transform(window) for window in right_windows])
        print(scipy.stats.describe(left_transform))
        print(scipy.stats.describe(right_transform))
        transform_hist_fig = plot_metric_histogram(left_transform, right_transform, num_bins=n_bins)
    print('End of iterations')
    window_itr = np.nditer(windows)
    # abs_arr = np.array([utils.metrics.abs_mean(window) for window in windows])
    # abs_fig = plot_metric_histogram(abs_arr[:bkps[0]], abs_arr[bkps[0]:bkps[1]])
    # rms_arr = np.array([utils.metrics.rms(window) for window in windows])
    # rms_fig = plot_metric_histogram(rms_arr[:bkps[0]], rms_arr[bkps[0]:bkps[1]])
    # skew_arr = np.array([utils.metrics.skewness(window) for window in windows])
    # skewness_fig = plot_metric_histogram(skew_arr[:bkps[0]], skew_arr[bkps[0]:bkps[1]])
    # kurtosis_arr = np.array([utils.metrics.kurtosis(window) for window in windows])
    # kurtosis_fig = plot_metric_histogram(kurtosis_arr[:bkps[0]], kurtosis_arr[bkps[0]:bkps[1]])
    # crest_arr = np.array([utils.metrics.crest_factor(window) for window in windows])
    # crest_fig = plot_metric_histogram(crest_arr[:bkps[0]], crest_arr[bkps[0]:bkps[1]])
    # impulse_arr = np.array([utils.metrics.impulse_factor(window) for window in windows])
    # impulse_fig = plot_metric_histogram(impulse_arr[:bkps[0]], impulse_arr[bkps[0]:bkps[1]])
    # shape_arr = np.array([utils.metrics.shape_factor(window) for window in windows])
    # shape_fig = plot_metric_histogram(shape_arr[:bkps[0]], shape_arr[bkps[0]:bkps[1]])
    ## Rejected metrics below
    # peaked_arr = np.array([np.max(np.abs(window))/utils.metrics.rms(window) for window in windows])
    # peaked_fig = plot_metric_histogram(peaked_arr[:bkps[0]], peaked_arr[bkps[0]:bkps[1]])
    # simp_arr = np.array([np.mean(np.abs(window)) + np.var(window) for window in windows])
    # simple_fig = plot_metric_histogram(simp_arr[:bkps[0]], simp_arr[bkps[0]:bkps[1]])
    # scalar = 1_000
    # rolling = np.array([np.var(np.ediff1d(window))/np.sqrt(np.mean(np.abs(window))) for window in windows])
    # rolling *= scalar
    # rolling_fig = plot_metric_histogram(rolling[:bkps[0]], rolling[bkps[0]:bkps[1]])
    range_arr = np.array([np.max(window) - np.min(window) for window in windows])
    print(scipy.stats.describe(range_arr[:bkps[0]]))
    print(scipy.stats.describe(range_arr[bkps[0]:bkps[1]]))
    range_fig = plot_metric_histogram(range_arr[:bkps[0]], range_arr[bkps[0]:bkps[1]])
    # iq_range_arr = np.array([scipy.stats.iqr(window) for window in windows])
    # iq_range_fig = plot_metric_histogram(iq_range_arr[:bkps[0]], iq_range_arr[bkps[0]:bkps[1]])


# def run_model(time: np.ndarray, data: np.ndarray, model: DetectionAlgorithm):
#     """ """
#     hp = model.hyperparameters
#     match model.name:
#         case 'bocpd':
#             get_bocpd_v5_from_generator(
#                 time, data, hp['mu'], hp['kappa'], hp['alpha'], hp['beta'],
#                 hp['lamb'], with_progress=model.show_progress)
#         case 'expectation maximization':
#             get_expectation_maximization_model_from_generator(
#                 time, data, with_progress=model.show_progress)
#         case 'cusum':
#             cusum_alg(time, data)
#         case 'grey':
#             get_grey_model_from_generator(time, data, with_progress=model.show_progress)
#         case 'nonparametric':
#             get_nonparametric_model_from_generator(time, data, with_progress=model.show_progress)


def plot_detection_1(time, data, models):
    """ """
    #todo save names should be part of config
    # Get ground truth here
    # Evaluation stuff
    (true_shocks, true_nonshocks) = make_ground_truth(time, data)
    ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
    predictions = list()
    algorithm_names = list()
    signal_names = list()
    signal_idx = 0
    for model in models:
        signal_idx += 1
        match model.name:
            case 'bocpd':
                shocks, non_shocks = get_bocpd_v5_from_generator(
                    time, data, **asdict(model.hyperparameters),
                    with_progress=model.with_progress)
                detection_fig = plot_shock(time, data, shocks, non_shocks)
                plt.savefig(Path(model.save_path, 'bocpd_fig.png'), dpi=350)
                plt.close(detection_fig)
                pred = intervals_to_dense_arr(time, shocks, non_shocks)
                print_scores(time, ground, pred)
            case 'expectation maximization':
                hp: Hyperparameters.EMHyperparams = model.hyperparameters
                rng = np.random.default_rng()
                safe = rng.normal(
                    hp.normal_mean, math.sqrt(hp.normal_var),
                    hp.normal_data_size)
                unsafe = rng.normal(
                    hp.abnormal_mean, math.sqrt(hp.abnormal_var),
                    hp.abnormal_data_size)
                shocks, non_shocks = get_expectation_maximization_model_from_generator(
                    time, safe, unsafe, data, hp.normal_mean,
                    hp.abnormal_mean, hp.normal_var, hp.abnormal_var, hp.pi,
                    hp.epochs, with_progress=model.with_progress)
                detection_fig = plot_shock(time, data, shocks, non_shocks)
                plt.savefig(Path(model.save_path, 'expectation_maximization_fig.png'), dpi=350)
                plt.close(detection_fig)
                pred = intervals_to_dense_arr(time, shocks, non_shocks)
                print_scores(time, ground, pred)
            case 'cusum':
                shocks, non_shocks = cusum_alg(
                    time, data, **asdict(model.hyperparameters))
                detection_fig = plot_shock(time, data, shocks, non_shocks)
                plt.savefig(Path(model.save_path, 'cusum_fig.png'), dpi=350)
                plt.close(detection_fig)
                pred = intervals_to_dense_arr(time, shocks, non_shocks)
                print_scores(time, ground, pred)
            case 'grey':
                hp: Hyperparameters.GreyHyperparams = model.hyperparameters
                shocks, non_shocks = get_grey_model_from_generator(
                    time, data, hp.window_size, hp.critical_value,
                    hp.critical_ratio_value,
                    with_progress=model.with_progress)
                detection_fig = plot_shock(time, data, shocks, non_shocks)
                plt.savefig(Path(model.save_path, 'grey_fig.png'), dpi=350)
                plt.close(detection_fig)
                pred = intervals_to_dense_arr(time, shocks, non_shocks)
                print_scores(time, ground, pred)
            case 'nonparametric':
                hp: Hyperparameters.NonparametricHyperparams = model.hyperparameters
                shocks, non_shocks = get_nonparametric_model_from_generator(
                    time, data, hp.window_size, hp.alpha, hp.critical_value,
                    with_progress=model.with_progress)
                detection_fig = plot_shock(time, data, shocks, non_shocks)
                plt.savefig(Path(model.save_path, 'nonparametric_fig.png'), dpi=350)
                plt.close(detection_fig)
                pred = intervals_to_dense_arr(time, shocks, non_shocks)
                print_scores(time, ground, pred)
            case _:
                raise NotImplementedError
        predictions.append(pred)
        algorithm_names.append(model.name)
        signal_names.append(signal_idx)
    df: pd.DataFrame = write_metric_table(time, ground, predictions, algorithm_names, signal_names)
    print(df.head())
    save_folder = './figures'
    tex_table: pd.DataFrame = df.iloc[:,1:]
    # convert the columns from s to ms
    tex_table.loc[:, 'earliest correct'] *= 1_000
    tex_table.loc[:, 'delay'] *= 1_000
    tex_table.rename({
        'earliest correct': 'earliest correct (ms)',
        'delay': 'detection delay (ms)'})
    write_frame_to_latex(df.iloc[:,1:], 'table.tex', save_folder)

def write_frame_to_latex(frame: pd.DataFrame, filename, folder=None):
    """ Writes reference frame and prediction frame to LaTeX files.

    Parameters
    ----------
    :param pd.DataFrame frame: Pandas Dataframe containing reference data metrics.
    :param filename: File name to write reference data LaTeX file.
    :type filename: str or None
    :param folder: Optional folder to prefix file names.
    :type folder: str or None


    """
    tmp_folder = folder if folder else ''
    frame.style.format(precision=3).hide(axis='index').to_latex(
        buf=Path(tmp_folder, filename), hrules=True)

def read_model_config(config_file):
    """ Parse config file for models."""
    config_table = load_toml(config_file)
    default_save_path = save_path(config_table['save-root'])
    models = config_table['models']
    algs = list()
    for model in models:
        hp = model['hyperparameters']
        if 'save-path' in model:
            save_name = save_path(model['save-path'])
        else:
            save_name = default_save_path
        if 'show-progress' in model:
            with_progress = model['show-progress']
        else:
            with_progress = False
        match model['name']:
            case 'bocpd':
                alg = DetectionAlgorithm(
                    name=model['name'], with_progress=with_progress,
                    save_path=save_name,
                    hyperparameters=Hyperparameters.BOCPDHyperparams(
                        alpha=hp['alpha'], beta=hp['beta'], mu=hp['mu'],
                        kappa=hp['kappa'], lamb=hp['lambda']))
            case 'expectation maximization':
                alg = DetectionAlgorithm(
                    name=model['name'], with_progress=with_progress,
                    save_path=save_name,
                    hyperparameters=Hyperparameters.EMHyperparams(
                        normal_data_size=hp['normal-data-size'],
                        abnormal_data_size=hp['abnormal-data-size'],
                        normal_mean=hp['normal-mean'],
                        abnormal_mean=hp['abnormal-mean'],
                        normal_var=hp['normal-variance'],
                        abnormal_var=hp['abnormal-variance'],
                        pi=hp['pi'], epochs=hp['epochs']))
            case 'cusum':
                alg = DetectionAlgorithm(
                    name=model['name'], with_progress=with_progress,
                    save_path=save_name,
                    hyperparameters=Hyperparameters.CUSUMHyperparams(
                        mean=hp['mean'], std_dev=hp['standard-deviation'], h=hp['h'],
                        alpha=hp['alpha']))
            case 'grey':
                alg = DetectionAlgorithm(
                    name=model['name'], with_progress=with_progress,
                    save_path=save_name,
                    hyperparameters=Hyperparameters.GreyHyperparams(
                        window_size=hp['window-size'],
                        critical_value=hp['critical-value'],
                        critical_ratio_value=hp['critical-ratio-value'],
                        alpha=hp['alpha']))
            case 'nonparametric':
                alg = DetectionAlgorithm(
                    name=model['name'], with_progress=with_progress,
                    save_path=save_name,
                    hyperparameters=Hyperparameters.NonparametricHyperparams(
                        window_size=hp['window-size'],
                        critical_value=hp['critical-value'], alpha=hp['alpha']
                    ))
            case _:
                raise NotImplementedError
        algs.append(alg)
    return algs



def plot_metric_boxplot(left_data, right_data):
    pass


# def plot_metric_pair_matrix(left_data, right_data, metrics):
#     num_metrics = len(metrics)
#     fig, axes = plt.subplots(ncols=num_metrics, nrows=num_metrics)
#     for idx in range(num_metrics):
#         row_metric = metrics[idx]
#         transform_arr = np.array([row_metric(window) for window in windows])
#         left_transform = transform_arr[:bkps[0]]
#         right_transform = transform_arr[bkps[0]:bkps[1]]
#
#         for jdx in range(num_metrics):
#             col_metric = metrics[jdx]
#             if idx != jdx:
#                 ax = axes[idx, jdx]
#                 ax.scatter()


def profile():
    cProfile.run('main()', sort='ncalls')


def run_from_config():
    """ """
    config_file = './src/configs/first_impact_config.toml'
    config_table = load_toml(config_file)
    my_data = get_data(config_table['file-path'])
    time, data = my_data[:, 0], my_data[:, 1]
    algs = read_model_config(config_file)
    plot_detection_1(time, data, algs)


def main():
    set_rc_params()
    file_path = './data/Dataset-7-forced-vibration-and-shock/data/dataset-A/inputData1_raw.txt'
    save_dir = Path('figures', '2024-09-18', 'signal-2')
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    # make_signal_plots(time, data, save_root=save_dir)
    # plot_offline_detections(time, data, save_dir)
    # plot_signal_fft(time, data)
    # make_signal_overlay_plot(time, data, save_root=save_dir)
    # make_stacked_power_spectrum_plot(time, data)
    # plt.show()
    # plot_signals(file_path, save_dir)
    # plot_detections(time, data, save_dir)
    # data_transformations(data)
    plt.show()


if __name__ == '__main__':
    run_from_config()
    exit()
    # profile()
    main()
