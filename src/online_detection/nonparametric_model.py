import math
import scipy.stats

import numpy as np

from math import sqrt
from scipy.stats import rankdata
from tqdm import tqdm

from online_detection.model_helpers import detection_to_intervals_for_generator_v1
from src.utils import metrics
from src.online_detection.grey_systems_model import get_rolling_window, accumulation_sequence, mean_sequence


def window_to_factors(window):
    """ Return array of metrics from window."""
    funcs = [
        metrics.crest_factor, metrics.shape_factor, metrics.impulse_factor,
        metrics.abs_mean, metrics.rms, metrics.skewness, metrics.kurtosis]
    return np.array([func(window) for func in funcs])


def test_statistic(r_value, n, crit_value=1.965):
    if n == 0:
        return np.inf
    n_n_plus_1 = n * (n + 1)
    denom = sqrt(n_n_plus_1 * (2 * n + 1) / 24)
    return abs((r_value - n_n_plus_1/4)/denom)


def test_statistic_2(mean_seq_1, mean_seq_2, window_size):
    """

    This version accounts for ties and zeros in the data
    """
    seq_diff = mean_seq_1 - mean_seq_2
    num_zeros = len(seq_diff[seq_diff == 0.0])
    seq_diff = seq_diff[seq_diff != 0]
    all_ranked = rankdata(seq_diff, method='average')
    pos_idx = seq_diff > 0.0
    neg_idx = seq_diff < 0.0
    pos_rank_value = np.sum(all_ranked[pos_idx])
    neg_rank_value = np.sum(all_ranked[neg_idx])
    r_value = min(pos_rank_value, neg_rank_value)
    # pos_diffs = seq_diff[seq_diff > 0.0]
    # neg_diffs = seq_diff[seq_diff < 0.0]
    # ranks = rankdata(pos_diffs, method='average')
    # neg_ranks = rankdata(neg_diffs, method='average')
    # neg_r_value = np.dot(neg_diffs, neg_ranks)
    # r_value = np.dot(pos_diffs, ranks)
    if window_size == num_zeros:
        return np.inf
    if num_zeros != 0:
        c = tie_correction(all_ranked[pos_idx]) if r_value == pos_rank_value else all_ranked[neg_idx]
        mean = (window_size * (window_size + 1)) / 4 - (num_zeros * (num_zeros + 1)) / 4
        var = (window_size * (window_size + 1) * (2 * window_size + 1) - (num_zeros * (num_zeros + 1) * (2 * num_zeros + 1)) - (c / 2)) / 24
    else:
        mean = (window_size * (window_size + 1)) / 4
        var = (window_size * (window_size + 1) * (2 * window_size + 1)) / 24
    return abs((r_value - mean) / sqrt(var))


def norm_cdf(val):
    """ Calculate the CDF of the normal function.

    """
    return 0.5 * (1 + math.erf(val / sqrt(2)))
# def test_statistic_r_2(r_value, n, crit_value):
#     """ """
#     pass


def tie_correction(rank_arr):
    """ """
    # could also do this
    # len(rank_arr) == 0
    if rank_arr.size == 0:
        return 0
    repeated_ranks, counts = scipy.stats.find_repeats(rank_arr)
    return np.sum(counts**3 - counts)


def r_func(mean_seq_1, mean_seq_2):
    """ """
    seq_diff = mean_seq_1 - mean_seq_2
    pos_diffs = seq_diff[seq_diff > 0.0]
    return np.dot(pos_diffs, np.arange(1, len(pos_diffs) + 1, 1)), len(pos_diffs)
    # return np.dot(seq_diff, np.arange(1, len(seq_diff) + 1, 1))


def rank_func_2(mean_seq_1, mean_seq_2):
    """ """
    # Wilcoxon diff rank test
    seq_diff = mean_seq_1 - mean_seq_2
    # seq_diff = seq_diff[seq_diff != 0.0]
    signs = np.sign(seq_diff)
    abs_diff = np.abs(seq_diff)
    ranked_abs = rankdata(abs_diff, method='average')
    r_value = np.dot(signs, ranked_abs)
    return r_value, len(ranked_abs)


def nonparametric_model(data, window_size=1, alpha=0.5, crit_value=1.965):
    """ Return Grey Model predictions for given data.

        :param np.ndarray data: Array of data. Data expected to be non-negative.
        :param int window_size: Size of window to iterate over array.
        :param float c: Constant multiplier for degree of grey incidence.
        :param float c_ratio: Constant multiplier for degree of ratio grey incidence.
        :returns:
        :rtype: None"""
    cps = 0
    attack = False
    window_1 = get_rolling_window(data, 0, window_size)
    x_1 = accumulation_sequence(window_1)
    z_1 = mean_sequence(x_1, alpha=alpha)
    for idx in tqdm(range(0, len(data) - window_size)):
        window_2 = get_rolling_window(data, idx, window_size)
        x_2 = accumulation_sequence(window_2)
        z_2 = mean_sequence(x_2, alpha=alpha)
        r_val, n = r_func(z_1, z_2)
        t_val = test_statistic(r_val, n)
        attack_likely = t_val <= crit_value
        r_val_2, n = rank_func_2(z_1, z_2)
        t_val_2 = test_statistic(r_val_2, n)
        attack_likely = t_val_2 <= crit_value
        if attack_likely:
            cps += 1
            attack = False
    print(cps)
    return


def nonparametric_model_generator(data, window_size=1, alpha=0.5, crit_value=1.965):
    """ """
    # neg_crit, pos_crit = -1.64, 1.64
    neg_crit, pos_crit = -1.96, 1.96
    # neg_crit, pos_crit = -1.99, 1.96
    cps = 0
    attacks, z_attacks, a_attacks = 0, 0, 0
    too_many_zeros = 0
    t_crit = 0.975
    half_of_ranks = (window_size * (window_size + 1)) / 2
    window_1 = get_rolling_window(data, 0, window_size)
    x_1 = accumulation_sequence(np.abs(window_1)) # todo remove the abs after this test
    z_1 = mean_sequence(x_1, alpha=alpha)
    # a_1 = window_to_factors(window_1)
    for idx in range(0, len(data) - window_size):
        window_2 = get_rolling_window(data, idx, window_size)
        x_2 = accumulation_sequence(np.abs(window_2)) # todo remove the abs after this test
        z_2 = mean_sequence(x_2, alpha=alpha)
        # a_2 = window_to_factors(window_2)
        if np.all(window_1 == window_2):
            attack_likely = False
        else:
            window_diff = np.around(window_1 - window_2, decimals=3)
            nonzeros = np.count_nonzero(window_diff)
            if nonzeros <= window_size // 4:
                too_many_zeros += 1
            # r_val, n = r_func(z_1, z_2)
            # t_val = test_statistic(r_val, n)
            # attack_likely = t_val <= crit_value
            # r_val_2, n = rank_func_2(z_1, z_2)
            # t_val_2 = test_statistic(r_val_2, n)
            # # attack_likely = t_val_2 <= crit_value
            # res = scipy.stats.wilcoxon(np.around(window_1 - window_2, decimals=3), zero_method='zsplit', method='approx')
            # attack_likely = not (res.zstatistic <= neg_crit or res.zstatistic > pos_crit)
            # if attack_likely:
            #     attacks += 1
            z_res = scipy.stats.wilcoxon(np.around(z_1 - z_2, decimals=3), zero_method='zsplit', method='approx')
            if not (z_res.zstatistic <= neg_crit or z_res.zstatistic > pos_crit):
                z_attacks += 1
            attack_likely = z_res.zstatistic <= neg_crit or z_res.zstatistic > pos_crit
            # a_res = scipy.stats.wilcoxon(np.around(a_1 - a_2, decimals=3), zero_method='zsplit', method='approx')
            # if not (a_res.zstatistic <= neg_crit or a_res.zstatistic > pos_crit):
            #     a_attacks += 1
            # attack_likely = (a_res.zstatistic <= neg_crit or a_res.zstatistic > pos_crit)
        if attack_likely:
            cps += 1
        yield attack_likely
    print(f'Times wilcoxon z score test: {attacks}, wilcoxon z transform z score test: {z_attacks}, metric score test: {a_attacks}')
    print(f'Times we had too many zeros in array: {too_many_zeros}')
    print(cps)


def get_nonparametric_model_from_generator(
        time, data, window_size=1, alpha=0.5, crit_value=1.965,
        shock_intervals=None, non_shock_intervals=None, with_progress=False):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    my_data = data
    begin = 0
    shock = False
    attacks, not_attacks = 0, 0
    # # This is a generator
    # nonparametric_model_gen = nonparametric_model_generator(my_data, window_size, alpha, crit_value)
    # items = tqdm(nonparametric_model_gen, total=len(data)-window_size) if with_progress else nonparametric_model_gen
    # for idx, is_attack in enumerate(items, start=window_size):
    #     if is_attack and not shock:  # If detected attack and not in shock state, change state
    #         non_shocks.append((time[begin], time[idx]))
    #         shock = True
    #         begin = idx
    #         attacks += 1
    #     elif not is_attack and shock:
    #         shocks.append((time[begin], time[idx]))
    #         shock = False
    #         begin = idx
    #         not_attacks += 1
    #     # attack = False
    # print(f'Safe points: {attacks}, change points: {not_attacks}')
    # # Check if remaining segment is shock or not
    # if shock:
    #     shocks.append((time[begin], time[-1]))
    # else:
    #     non_shocks.append((time[begin], time[-1]))
    if with_progress:
        nonparametric_model_gen = tqdm(
            nonparametric_model_generator(
                my_data, window_size, alpha, crit_value),
            total=len(data) - window_size)
    else:
        nonparametric_model_gen = nonparametric_model_generator(
            my_data, window_size, alpha, crit_value)
    shocks, non_shocks = detection_to_intervals_for_generator_v1(
        time, begin, nonparametric_model_gen, start_offset=window_size)
    return shocks, non_shocks
