import itertools
from functools import cache

import numpy as np
from numba import jit
from tqdm import tqdm

from fig_funcs.detection_plots import plot_shock
from utils.read_data import get_data


# def grey_model(data, window_size=1, c=3):
#     normal = window_size * 2
#     attacks = list()
#     s_1 = behavior_sequence(data, window_size, normal, window_size)
#     for idx in range(window_size, len(data)):
#         # # Equivalent to following 2 lines
#         # degree_old = get_grey_incidence(data, window_size, normal, idx, window_size, c=c)
#         # # Equivalent lines
#         s_2 = behavior_sequence(data, window_size, idx, window_size)
#         degree = grey_incidence(s_1, s_2, c=c)
#         # if degree_old != degree:
#         #     print('Degrees not equal!!!')
#         #
#         attack = is_attack(grey_incidence_degree=degree)
#         attacks.append(attack)
#     return attacks
#
#
# def grey_model_2(data, window_size=1, c=3):
#     # This section of time from 0 to normal is assumed safe
#     normal = window_size * 2
#     attacks = list()
#     window_1 = get_rolling_window(data, window_size, window_size)
#     s_1 = behavior_sequence(data, window_size, normal, window_size)
#     accum = np.zeros_like(window_1)
#     for idx in range(window_size, len(data)):
#         # Get our behavior sequences
#         s_2, accum = behavior_sequence_2(data, window_1, window_size, idx, window_size, accum)
#         # Use our behavior sequences to compute grey incidence degree
#         degree = grey_incidence(s_1, s_2, c=c)
#         attack = is_attack(grey_incidence_degree=degree)
#         attacks.append(attack)
#     return attacks


def grey_model_v1(data, window_size=1, c=3, c_ratio=300):
    """ """
    cps = 0
    attack = False
    window_1 = get_rolling_window(data, 0, window_size)
    x_1 = accumulation_sequence(window_1)
    z_1 = mean_sequence(x_1, alpha=0.5)
    s_1 = behavioral_sequence(z_1)
    s_1_ratio = behavioral_sequence_ratio(z_1)
    for idx in tqdm(range(0, len(data) - window_size)):
        window_2 = get_rolling_window(data, idx, window_size)
        x_2 = accumulation_sequence(window_2)
        z_2 = mean_sequence(x_2, alpha=0.5)
        s_2 = behavioral_sequence(z_2)
        s_2_ratio = behavioral_sequence_ratio(z_2)
        degree = grey_incidence_degree(s_1, s_2, c=c)
        degree_ratio = grey_incidence_degree_ratio(s_1_ratio, s_2_ratio, c=c_ratio)
        attack_likely = degree + degree_ratio <= 0.5
        if degree <= 0.5 and degree_ratio <= 0.5:
            attack = True
        if attack:
            cps += 1
            attack = False
    print(cps)
    return


def accumulation_sequence(window):
    return np.add.accumulate(window)


def mean_sequence(window, alpha=0.5):
    transformed = [window[0]]
    # transformed.extend(
    #     [prev * alpha + (1.0 - alpha) * curr] for (prev, curr) in itertools.pairwise(window))
    # for (prev, curr) in itertools.pairwise(window):
    #     transformed.append(prev * alpha + (1.0 - alpha) * curr)
    transformed.extend(
        [window[idx - 1] * alpha + (1 - alpha) * val for idx, val in enumerate(window[1:], start=1)]
    )
    # for idx, val in enumerate(window[1:], start=1):
    #     new_val = window[idx - 1] * alpha + (1 - alpha) * val
    #     transformed.append(new_val)
    return np.array(transformed)


def behavioral_sequence(window):
    s_0 = np.sum(window[:-1] - window[0]) + 0.5 * (window[-1] - window[0])
    return s_0


def behavioral_sequence_ratio(window, offset=1):
    s_0 = np.sum((window[:-1] + offset) / (window[0] + offset)) + ((0.5 * window[-1] + offset) / (0.5 * window[0] + offset))
    return s_0


def grey_incidence_degree(val_1, val_2, c=3.0):
    num = 1.0 + abs(val_1) + abs(val_2)
    return num / (num + c * abs(val_1 - val_2))
    # return 1 / ( 1.0 + 0.01 * abs(val_1 - val_2)/val_1)


def grey_incidence_degree_ratio(val_1, val_2, c=1.0):
    return 1.0 / (1.0 + c * abs(val_1 - val_2))


def get_rolling_window(obs, idx, n):
    """ Get window of size n with idx being start (inclusive)."""
    # this can be either way, we look back but can look forward
    # return obs[idx - n:idx]
    return obs[idx:idx + n]
#
#
# @jit(cache=True)
# def window_diff(window_1, window_2):
#     """ Calculate the difference between two windows."""
#     if len(window_1) != len(window_2):
#         print('sizes not equal')
#     return window_2 - window_1


# @jit(cache=True)
# def get_window_diff(obs, idx, jdx, n):
#     """ Calculate the two windows and return their difference.
#
#         idx is safe window, jdx is next one.
#     """
#     window_1 = get_rolling_window(obs, idx, n)
#     window_2 = get_rolling_window(obs, jdx, n)
#     return window_diff(window_1, window_2)


# @jit
# def behavior_sequence(obs, idx, jdx, n):
#     """ Calculate behavior sequence.
#
#         idx is start,
#     """
#     window_1 = get_rolling_window(obs, idx, n)
#     window_2 = get_rolling_window(obs, jdx, n)
#     diff = 0.5 * window_diff(window_1, window_2)
#     window_sum = sum_of_windows(obs, diff, window_1, idx, jdx, n)
#     return window_sum
    # if isinstance(obs, np.ndarray):
    #     result = np.sum([window_diff(window_1, get_rolling_window(obs, kdx, n)) for kdx in range(idx, jdx)])
    # else:
    #     result = sum([window_diff(window_1, get_rolling_window(obs, kdx, n)) for kdx in range(idx, jdx)])
    # return result + diff

    # # old code
    # diff = get_window_diff(obs, idx, jdx, n)
    # val = 0.5 * diff
    # if isinstance(obs, np.ndarray):
    #     result = np.sum([get_window_diff(obs, idx, kdx, n) for kdx in range(idx, jdx)])
    # else:
    #     result = sum([get_window_diff(obs, idx, kdx, n) for kdx in range(idx, jdx)])
    # return result + val


# @jit
# def behavior_sequence_2(obs, window_1, idx, jdx, n, last_windows=None):
#     """ """
#     window_2 = get_rolling_window(obs, jdx, n)
#     # windows = map(lambda x: window_diff(window_1, get_rolling_window(obs, x, n)), range(idx, jdx))
#     if jdx > n:
#         windows = get_rolling_window(obs, jdx - 1, n)
#         window_sum, new_accum = sum_of_windows_2(windows, window_1, window_2, accum=last_windows)
#     elif jdx == n:
#         diff = 0.5 * window_diff(window_1, window_2)
#         window_sum, new_accum = diff, np.zeros_like(window_1)
#         # return diff, np.zeros_like(window_1)
#     else:
#         print('The index jdx is less than window size')
#         window_sum, new_accum = np.zeros_like(window_1), np.zeros_like(window_1)
#     # window_sum = sum_of_windows(obs, diff, window_1, idx, jdx, n)
#     return window_sum, new_accum


# @jit
# def sum_of_windows(items, val, window, idx, jdx, n):
#     result = val
#     for kdx in range(idx, jdx):
#         result = result + window_diff(window, get_rolling_window(items, kdx, n))
#     return result
#
#
# def sum_of_windows_1(items, item_dict):
#     """ """
#     pass


# @jit
# def sum_of_windows_2(windows, safe_window, current_window, accum):
#     """
#
#     windows is either an iterable of windows from 1 to n-1 or the n-1th window
#     """
#     current = 0.5 * (current_window - safe_window)
#     next_accum = accum + (windows - safe_window)
#     return next_accum + current, next_accum


# @jit
# def sum_of_windows_2_no_accum(windows, safe_window, current_window):
#     """
#
#     windows is either an iterable of windows from 1 to n-1 or the n-1th window
#     """
#     current = 0.5 * (current_window - safe_window)
#     return np.sum([window - safe_window for window in windows]) + current


# def get_grey_incidence(obs, idx, jdx, kdx, n, c=3):
#     """ Grey incidence for two behavior sequences.
#
#         idx is start, jdx is safe, kdx is second.
#     """
#     s_1, s_2 = behavior_sequence(obs, idx, jdx, n), behavior_sequence(obs, idx, kdx, n)
#     return grey_incidence(s_1, s_2, c=c)
    # s_diff = s_1 - s_2
    # numerator = 1 + np.dot(s_1, s_1) + np.dot(s_2, s_2)
    # denominator = numerator + c * np.dot(s_diff, s_diff)
    # return numerator/denominator


# @jit
# def grey_incidence(s_1, s_2, c):
#     """
#
#     This version of grey incidence assumes you have the behavior sequences already.
#     """
#     s_diff = s_1 - s_2
#     numerator = 1 + np.dot(s_1, s_1) + np.dot(s_2, s_2)
#     denominator = numerator + c * np.dot(s_diff, s_diff)
#     return numerator / denominator


# Old code below
# def x_0(obs, n):
#     return obs[0:n]
#
#
# def x_1(obs, n):
#     values = [sum(obs[0:idx]) for idx in range(n)]
#     # for idx in range(n):
#     #     values[idx] = sum(obs[0:idx])
#     return values
#
#
# def z_1(obs, n):
#     x_1s = x_1(obs, n)
#     values = [(x_1s[idx - 1] + x_1s[idx]) / 2 for idx in range(1, n)]
#     return values
#
#
# def x_i(obs, n):
#     """ """
#     # vals = x_0(obs, n)
#     # return vals[n - 1] - vals[0]
#
#     # The two lines above are equivalent to the line below
#     return obs[n - 1] - obs[0]
#
#
# def s_i(obs, n):
#     """ """
#     summation = sum(lambda k: x_i(obs, k), range(0, n - 1))
#     val = 0.5 * x_i(obs, n)
#     return summation + val


# def grey_incidence(i_obs, j_obs, n, c=3):
#     """ """
#     numerator = 1 + np.abs(s_i(i_obs, n)) + np.abs(s_i(j_obs, n))
#     # denominator = 1 + np.abs(s_i(i_obs, n)) + np.abs(s_i(j_obs, n))
#     denominator = numerator + c * np.abs(s_i(i_obs, n) - s_i(j_obs, n))
#     return numerator/denominator


# @jit
# def is_attack(grey_incidence_degree):
#     """ """
#     return grey_incidence_degree <= 0.5


# def get_grey_model_old(time, data, window_size=1, c=3, shock_intervals=None, non_shock_intervals=None, with_progress=False):
#     """ """
#     # Instantiating variables
#     shocks = [] if shock_intervals is None else shock_intervals
#     non_shocks = [] if non_shock_intervals is None else non_shock_intervals
#     begin = 0
#     shock = False
#     # This is supposed to be a normal example we can use to compare
#     normal = window_size + 1
#     my_data = np.asarray(data)
#     items = tqdm(range(window_size, len(data))) if with_progress else range(window_size, len(data))
#     for idx in items:
#         degree = grey_incidence(my_data, window_size, normal, idx, window_size, c=c)
#         attack = is_attack(grey_incidence_degree=degree)
#         if attack and not shock:  # If detected attack and not in shock state, change state
#             non_shocks.append((time[begin], time[idx - 1]))
#             shock = True
#             begin = idx
#         elif not attack and shock:
#             shocks.append((time[begin], time[idx - 1]))
#             shock = False
#             begin = idx
#     # Check if remaining segment is shock or not
#     if shock:
#         shocks.append((time[begin], time[-1]))
#     else:
#         non_shocks.append((time[begin], time[-1]))
#     return shocks, non_shocks


def get_grey_model(time, data, window_size=1, c=3, c_ratio=3, shock_intervals=None, non_shock_intervals=None, with_progress=False):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    my_data = np.abs(data)
    begin = 0
    shock = False
    attacks, not_attacks = 0, 0
    cps = 0
    attack = False
    window_1 = get_rolling_window(my_data, 0, window_size)
    x_1 = accumulation_sequence(window_1)
    z_1 = mean_sequence(x_1, alpha=0.5)
    s_1 = behavioral_sequence(z_1)
    s_1_ratio = behavioral_sequence_ratio(z_1)
    items = tqdm(range(0, len(data) - window_size)) if with_progress else range(0, len(data) - window_size)
    for idx in items:
        window_2 = get_rolling_window(my_data, idx, window_size)
        x_2 = accumulation_sequence(window_2)
        z_2 = mean_sequence(x_2, alpha=0.5)
        s_2 = behavioral_sequence(z_2)
        s_2_ratio = behavioral_sequence_ratio(z_2)
        degree = grey_incidence_degree(s_1, s_2, c=c)
        degree_ratio = grey_incidence_degree_ratio(s_1_ratio, s_2_ratio, c=c_ratio)
        attack_likely = degree + degree_ratio <= 0.5
        if attack_likely:  # degree <= 0.5 and degree_ratio <= 0.5:
            attack = True
            cps += 1
        if attack and not shock:  # If detected attack and not in shock state, change state
            non_shocks.append((time[begin], time[idx + window_size]))
            shock = True
            begin = idx + window_size
            attacks += 1
        elif not attack and shock:
            shocks.append((time[begin], time[idx + window_size]))
            shock = False
            begin = idx + window_size
            not_attacks += 1
        attack = False
    print(f'Change points: {cps}')
    print(f'Safe points: {attacks}, change points: {not_attacks}')
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def get_plot_grey_model(file_path, with_progress=False):
    """ """
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    print('Grey Systems Modelling')
    shock_intervals, non_shock_intervals = get_grey_model(
        time, data, window_size=1000, c=3.0, c_ratio=0.5,
        with_progress=with_progress)
    fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    # return None, None, None
    return shock_intervals, non_shock_intervals, fig

