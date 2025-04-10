import math

from dataclasses import asdict
from pathlib import Path
from typing import TypeAlias

import numpy as np

from matplotlib import pyplot as plt

import Hyperparameters

from DetectionAlgorithm import ModelType, DetectionAlgorithm
from fig_funcs.detection_plots import plot_shock
# from main import make_ground_truth, print_scores, write_metric_table
from online_detection import bocpd, cusum, expect_Max, grey_systems_model, nonparametric_model
from online_detection.bocpd import get_bocpd_v5_from_generator
from online_detection.cusum import get_cusum_revised, cusum_alg, cusum_alg_v1
from online_detection.expect_Max import get_expectation_maximization_model_from_generator
from online_detection.grey_systems_model import get_grey_model_from_generator, get_grey_model
from online_detection.nonparametric_model import get_nonparametric_model_from_generator


# def plot_bocpd(time, data, show_progress=False, ground=None, save_root=None):
#     """ """
#     # todo make kwargs parser for run configs
#     save_dir = save_path(save_root)
#     # Generate segments to be used as ground truth
#     # todo save ground truth as numpy array file and load it
#     (true_shocks, true_nonshocks) = make_ground_truth(time, data)
#     ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
#     # BOCPD algorithms start running
#     # new hyperparameters
#     kappa = 100.0
#     alpha = kappa * 0.5
#     mu = np.mean(data[:100_000])
#     beta = alpha * np.var(data[:100_000])
#     #old hyperparameters
#     # kappa = 50
#     # alpha = kappa / 2.0
#     # beta = kappa / np.var(data[:kappa])
#     # mu = np.mean(data[:kappa])
#     # alpha, beta, mu, kappa = 0.5, 1, 0.0, 1.0
#     # cache = {time_val: idx for idx, time_val in time}
#     print('BOCPD detection')
#     # test
#     print('Start of test')
#     test_shocks, test_nonshocks = get_bocpd_v5_from_generator(
#         time, data, mu, kappa, alpha, beta, 100, with_progress=True)
#     test_fig_1 = plot_shock(time, data, test_shocks, test_nonshocks)
#     plt.savefig(Path(save_dir, 'bocpd_fig.png'), dpi=350)
#     # # Evaluation stuff
#     pred = intervals_to_dense_arr(time, test_shocks, test_nonshocks)
#     print_scores(time, ground, pred)
#     # bayesian_online_changepoint_detection_v5(np.abs(data), np.mean(np.abs(data[:100])), 0.1, 1, 100, 100)
#     test_shocks, test_nonshocks = get_bocpd_v5_from_generator(
#         time, data, np.mean(np.abs(data[:100])), 0.1, 0.1, 0.01, 100,
#         with_progress=True)
#     test_fig_2 = plot_shock(time, data, test_shocks, test_nonshocks)
#     plt.savefig(Path(save_dir, 'bocpd_2_fig.png'), dpi=350)
#     # # Evaluation stuff
#     pred = intervals_to_dense_arr(time, test_shocks, test_nonshocks)
#     print_scores(time, ground, pred)
#     print('End of test')
#     # plt.show()
#
#
# def plot_cusum(time, data, show_progress=False, save_root=None):
#     """ """
#     save_dir = save_path(save_root)
#     print('Starting CUSUM Algorithms')
#     # transformed_data =
#     shock_intervals, non_shock_intervals = get_cusum_revised(time, data, len(data))
#     fig_revised = plot_shock(time, data, shock_intervals, non_shock_intervals)
#     plt.savefig(Path(save_dir, 'cusum_revised_fig.png'), dpi=350)
#     plt.close(fig_revised)
#     # Evaluation stuff
#     (true_shocks, true_nonshocks) = make_ground_truth(time, data)
#     pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
#     ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
#     print_scores(time, ground, pred)
#     mean, std = np.mean(data[:100_000]), np.std(data[:100_000])
#     # shock_intervals, non_shock_intervals = simple_cusum(times, data, mean, std)
#     # fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
#     # plt.savefig('figures/5-14-2024/simple_cusum_fig.jpg', dpi=350)
#     # shock_intervals, non_shock_intervals = cusum(time, data, mean, std, alpha=0.025, beta=0.025)  # , alpha=0.025, beta=0.025
#     # fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
#     # plt.savefig('figures/5-14-2024/cusum_fig.png', dpi=350)
#     shock_intervals, non_shock_intervals = cusum_alg(
#         time, data, mean, std, h=5, alpha=0.95)  # 0.001
#     fig_alg = plot_shock(time, data, shock_intervals, non_shock_intervals)
#     plt.savefig(Path(save_dir, 'cusum_alg_fig.png'), dpi=350)
#     plt.close(fig_alg)
#     pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
#     print_scores(time, ground, pred)
#
#     shock_intervals, non_shock_intervals = cusum_alg_v1(
#         time, data, mean, std, h=5, alpha=0.999)  # 0.001
#     fig_alg = plot_shock(time, data, shock_intervals, non_shock_intervals)
#     plt.savefig(Path(save_dir, 'cusum_alg_v1_fig.png'), dpi=350)
#     plt.close(fig_alg)
#     pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
#     print_scores(time, ground, pred)
#
#
# def plot_expectation_maximization(time, data, show_progress=False, save_root=None):
#     """ """
#     save_dir = save_path(save_root)
#     print('Starting Expectation Maximization Algorithm')
#     my_data = np.abs(data)
#     mean_1 = np.mean(my_data[:100_000])
#     var_1 = np.var(my_data[:100_000])
#     print(np.mean(my_data[200_000:400_000]))
#     print(np.std(my_data[200_000:400_000]))
#     mean_2, var_2 = 20.0, 10.0
#     rng = np.random.default_rng()
#     safe = rng.normal(mean_1, math.sqrt(var_1), 70)
#     unsafe = rng.normal(mean_2, math.sqrt(var_2), 30)
#     # safe = np.random.normal(mean_1, math.sqrt(var_1), 70)
#     # unsafe = np.random.normal(mean_2, math.sqrt(var_2), 30)
#     shock_intervals_gen, non_shock_intervals_gen = get_expectation_maximization_model_from_generator(
#         time, safe, unsafe, data, mean_1=mean_1, var_1=var_1,
#         epochs=50, with_progress=show_progress)
#     fig_gen = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
#     plt.savefig(Path(save_dir, 'expectation_maximization_generator_fig.png'), dpi=350)
#     plt.close(fig_gen)
#     # Evaluation stuff
#     (true_shocks, true_nonshocks) = make_ground_truth(time, data)
#     pred = intervals_to_dense_arr(time, shock_intervals_gen, non_shock_intervals_gen)
#     ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
#     print_scores(time, ground, pred)
#     # shock_intervals, non_shock_intervals = get_expectation_max(
#     #     time, safe, unsafe, data, mean_1=mean_1, var_1=var_1,
#     #     epochs=100, with_progress=show_progress)
#     # fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
#     # plt.savefig(Path(save_dir, 'expectation_maximization_fig.png'), dpi=350)
#     # plt.close(fig)
#     # Evaluation stuff
#     # pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
#     # print_scores(time, ground, pred)
#
#
# def plot_grey_model(time, data, show_progress=False, save_root=None):
#     """ """
#     save_dir = save_path(save_root)
#     print('Starting Grey Models')
#     window_size, c, c_ratio = 100, 3.0, 0.01  # 1.0
#     print('grey model generator')
#     shock_intervals_gen, non_shock_intervals_gen = get_grey_model_from_generator(
#         time, np.abs(data), window_size=window_size, c=c, c_ratio=10.0,
#         with_progress=show_progress)
#     gen_fig = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
#     plt.savefig(Path(save_dir, 'grey_model_rel_dist.png'), dpi=350)
#     # Evaluation stuff
#     (true_shocks, true_nonshocks) = make_ground_truth(time, data)
#     pred = intervals_to_dense_arr(time, shock_intervals_gen, non_shock_intervals_gen)
#     ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
#     print_scores(time, ground, pred)
#     print('Grey Systems Modelling')
#     shock_intervals, non_shock_intervals = get_grey_model(
#         time, np.abs(data), window_size=window_size, c=c, c_ratio=c_ratio,
#         with_progress=show_progress)
#     fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
#     plt.savefig(Path(save_dir, 'grey_model.png'), dpi=350)
#     # Evaluation stuff
#     pred = intervals_to_dense_arr(time, shock_intervals, non_shock_intervals)
#     print_scores(time, ground, pred)
#
#
# def plot_nonparametric_model(time, data, show_progress=False, save_root=None):
#     """ """
#     save_dir = save_path(save_root)
#     window_size, crit_value = 60, 1.965
#     shock_intervals_gen, non_shock_intervals_gen = get_nonparametric_model_from_generator(
#         time, data, window_size, crit_value=crit_value, with_progress=show_progress)
#     fig = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
#     plt.savefig(Path(save_dir, 'nonparametric.png'), dpi=350)
#     # Evaluation stuff
#     (true_shocks, true_nonshocks) = make_ground_truth(time, data)
#     pred = intervals_to_dense_arr(time, shock_intervals_gen, non_shock_intervals_gen)
#     ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
#     print_scores(time, ground, pred)


# def plot_detections(time, data, save_root=None):
#     """ """
#     save_dir = save_path(save_root)
#     # DNI
#     # dni_shock, dni_non_shock, dni_fig = get_plot_dni(file_path)
#     # dni_abs_hist = interval_histogram(time, data, dni_shock, dni_non_shock)
#     # dni_abs_hist = raw_histogram(time, data, dni_shock, dni_non_shock)
#     # BOCPD
#     plot_bocpd(time, data, True, save_root=save_dir)
#     # CUSUM
#     plot_cusum(time, data, True, save_root=save_dir)
#     # EM
#     plot_expectation_maximization(time, data, True, save_root=save_dir)
#     # Grey Systems
#     plot_grey_model(time, data, True, save_root=save_dir)
#     # Nonparametric
#     plot_nonparametric_model(time, data, True, save_root=save_dir)
#     # Estimated Ground Truth
#     # bkps = binary_segmentation.get_breaks(np.abs(data), 2, model_type='rank')
#     # ground_fig = plot_breaks(data, bkps, show=False)
#     # plt.savefig(Path(save_dir, 'ground_plot.png'), dpi=350)

RegionType: TypeAlias = tuple[float, float]
ResultType: TypeAlias = tuple[DetectionAlgorithm, RegionType, RegionType]

def run_online_models(
        time: np.ndarray, data: np.ndarray,
        models: [DetectionAlgorithm]) ->[ResultType]:
    """ Run each model in the models list on given data and return results.

        :param time: Array of time data
        :param data: Array of data used for detection.
        :param models: Iterable of model information for models to run.
        :returns: A list of tuples consisting of: model, detected shock
         regions, detected non-shock regions.
    """
    results = list()
    signal_idx = 0
    # Start iteration
    for model in models:
        signal_idx += 1
        match model.type:
            case ModelType.BOCPD:
                shocks, non_shocks = bocpd.get_bocpd_v5_from_generator(
                    time, data, **asdict(model.hyperparameters),
                    with_progress=model.with_progress)
            case ModelType.EM:
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
            case ModelType.CUSUM:
                shocks, non_shocks = cusum.cusum_alg(
                    time, data, **asdict(model.hyperparameters))
            case ModelType.GREY_MODEL:
                hp: Hyperparameters.GreyHyperparams = model.hyperparameters
                shocks, non_shocks = get_grey_model_from_generator(
                    time, data, hp.window_size, hp.critical_value,
                    hp.critical_ratio_value,
                    with_progress=model.with_progress)
            case ModelType.NON_PARAMETRIC:
                hp: Hyperparameters.NonparametricHyperparams = model.hyperparameters
                shocks, non_shocks = get_nonparametric_model_from_generator(
                    time, data, hp.window_size, hp.alpha, hp.critical_value,
                    with_progress=model.with_progress)
            case _:
                if isinstance(model.type, ModelType):
                    raise NotImplementedError(f'{model.type} does not have a detection algorithm implemented yet.')
                else:
                    raise ValueError(f'{model.type} is not one of the algorithms currently available.')
        results.append((model, shocks, non_shocks))
    return results

def plot_detection_online_models(time: np.ndarray, data: np.ndarray, results: ResultType):
    """ Plot detection figures for each model."""
    # set some defaults here
    dpi = 350
    to_ms = True
    for (model, shocks, non_shocks) in results:
        detection_fig = plot_shock(time, data, shocks, non_shocks, to_ms=to_ms)
        plt.savefig(Path(model.save_path).with_suffix('.png'),dpi=dpi)
        plt.close(detection_fig)
