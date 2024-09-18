import math

from matplotlib import pyplot as plt

from fig_funcs import rupture_changepoint_plots
from fig_funcs.detection_plots import plot_shock, interval_histogram, raw_histogram
from fig_funcs.signal_plots import plot_signal, plot_signal_fft, plot_signal_power_spectrum, signal_with_overlays, \
    power_spectra_sections, signal_with_inset_axes
from fig_funcs.spectrogram_plots import get_spectrogram, plot_spectrogram
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
from online_detection.nonparametric_model import get_nonparametric_model_from_generator
from utils.matplotlib_formatting import set_rc_params
from utils.read_data import get_data
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


def plot_signals(file_path):
    """ """
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    make_signal_plots(time, data)
    make_spectrogram_plots(time, data)



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


def print_scores(time, ground, predictions):
    """ Print metric scores for predicted shock prediction given comparator."""
    # Calculate scores
    f1_score = sklearn.metrics.f1_score(ground, predictions)
    precision = sklearn.metrics.precision_score(ground, predictions)
    recall = sklearn.metrics.recall_score(ground, predictions)
    accuracy = sklearn.metrics.accuracy_score(ground, predictions)
    dice_similarity = 1.0 - scipy.spatial.distance.dice(ground, predictions)
    # Print scores
    print(f'F1 score: {f1_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}, Dice similarity: {dice_similarity:.3f}')
    # Print confusion matrix
    # confusion = sklearn.metrics.confusion_matrix(ground, predictions)
    confusion = sklearn.metrics.confusion_matrix(ground, predictions, normalize='all')
    print(confusion)
    print(sklearn.metrics.classification_report(ground, predictions, digits=3))
    # Price is right score
    # todo we might need to assert that this is not empty before doing indexing, otherwise might break
    true_positive_indices = np.logical_and(ground.astype(bool), predictions.astype(bool))
    earliest_correct = time[true_positive_indices][0]
    print(f'Shock first correctly detected at time: {earliest_correct}')


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
    # pred = intervals_to_dense_arr(time, test_shocks, test_nonshocks)
    # print_scores(time, ground, pred)
    # bayesian_online_changepoint_detection_v5(np.abs(data), np.mean(np.abs(data[:100])), 0.1, 1, 100, 100)
    test_shocks, test_nonshocks = get_bocpd_v5_from_generator(
        time, data, np.mean(np.abs(data[:100])), 0.1, 0.1, 0.01, 100,
        with_progress=True)
    test_fig_2 = plot_shock(time, data, test_shocks, test_nonshocks)
    plt.savefig(Path(save_dir, 'bocpd_2_fig.png'), dpi=350)
    # # Evaluation stuff
    # pred = intervals_to_dense_arr(time, test_shocks, test_nonshocks)
    # print_scores(time, ground, pred)
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
    window_size, crit_value = 100, 1.965
    shock_intervals_gen, non_shock_intervals_gen = get_nonparametric_model_from_generator(
        time, np.abs(data), window_size, crit_value=crit_value, with_progress=show_progress
    )
    fig = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
    plt.savefig(Path(save_dir, 'nonparametric.png'), dpi=350)
    # Evaluation stuff
    (true_shocks, true_nonshocks) = make_ground_truth(time, data)
    pred = intervals_to_dense_arr(time, shock_intervals_gen, non_shock_intervals_gen)
    ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
    print_scores(time, ground, pred)




def plot_detections(file_path, save_root=None):
    """ """
    save_dir = save_path(save_root)
    # save_dir = Path('figures', '2024-09-03', 'signal-1')
    my_data = get_data(file_path).astype('float32')
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
    # Nonparametric
    plot_nonparametric_model(time, data, True)


def profile():
    cProfile.run('main()', sort='ncalls')


def main():
    set_rc_params()
    file_path = './data/Dataset-7-forced-vibration-and-shock/data/dataset-A/inputData1_raw.txt'
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    # plot_offline_detections(time, data)
    # make_signal_overlay_plot(time, data)
    # make_stacked_power_spectrum_plot(time, data)
    # exit()
    plt.show()
    # temp()
    # plot_signals(file_path)
    plot_detections(file_path)
    plt.show()


if __name__ == '__main__':
    # profile()
    main()
