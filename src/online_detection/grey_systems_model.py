import itertools
from functools import cache

import numpy as np
from numba import jit
from tqdm import tqdm

from fig_funcs.detection_plots import plot_shock
from utils.read_data import get_data


def grey_model(data, window_size=1, c=3, c_ratio=300):
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


@jit
def grey_model_generator(data, window_size=1, c=3, c_ratio=300):
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
    z_1 = mean_sequence(x_1, alpha=0.5)
    s_1 = behavioral_sequence(z_1)
    s_1_ratio = behavioral_sequence_ratio(z_1)
    s_1_ratio = behavioral_sequence_ratio_2(z_1)
    for idx in range(0, len(data) - window_size):
        window_2 = get_rolling_window(data, idx, window_size)
        x_2 = accumulation_sequence(window_2)
        z_2 = mean_sequence(x_2, alpha=0.5)
        s_2 = behavioral_sequence(z_2)
        s_2_ratio = behavioral_sequence_ratio(z_2)
        s_2_ratio = behavioral_sequence_ratio_2(z_2)
        degree = grey_incidence_degree(s_1, s_2, c=c)
        degree_ratio = grey_incidence_degree_ratio(s_1_ratio, s_2_ratio, c=c_ratio)
        attack_likely = degree + degree_ratio <= 0.5
        yield attack_likely
    print(cps)


@jit
def accumulation_sequence(window):
    """ Return the accumulation over the window."""
    # return np.add.accumulate(window)
    output = np.cumsum(window)
    output[-1] = np.sum(window)
    return output


@jit
def mean_sequence(window, alpha=0.5):
    """ Return the running average over a window."""
    transformed = np.empty_like(window)
    transformed[0] = window[0]
    val = window[0]
    for idx in range(1, len(window)):
        val = mean_helper(val, window[idx], alpha)
        transformed[idx] = val
    return transformed
    # transformed = [window[0]]
    # transformed.extend(
    #     [prev * alpha + (1.0 - alpha) * curr] for (prev, curr) in itertools.pairwise(window))
    # transformed.extend(
    #     [window[idx - 1] * alpha + (1 - alpha) * val for idx, val in enumerate(window[1:], start=1)]
    # )
    # transformed.extend(
    #     [window[idx - 1] * alpha + (1 - alpha) * window[idx] for idx in range(start=1, stop=len(window))]
    # )

    # return np.array(transformed)


@jit
def mean_helper(val_1, val_2, alpha):
    return (val_1 * alpha) + (val_2 * (1 - alpha))

@jit
def behavioral_sequence(window):
    s_0 = np.sum(window[:-1] - window[0]) + 0.5 * (window[-1] - window[0])
    return s_0


@jit
def behavioral_sequence_ratio(window, offset_1=1e-32, offset_2=1.0):
    # v - vref / math.abs(vref)
    # math.abs(x - y)/(math.abs(x) + math.abs(y))/2, (0, 0) = 0
    # This assumes window is composed only of nonnegative numbers
    s_0 = np.sum((window[:-1] + offset_2) / (window[0] + offset_2)) + 0.5 * ((window[-1] + offset_2) / (window[0] + offset_2))
    # s_0 = ((0.5 * window[-1] + offset_2) / (0.5 * window[0] + offset_2))
    # for item in window[:-1]:
    #     if item != 0.0 or window[0] != 0.0:

    return s_0


@jit
def behavioral_sequence_ratio_2(window):
    """ """
    head = window[0]
    # s_0 = 0.5 * (window[-1] - head) / (np.abs(window[-1]) + np.abs(head))
    s_0 = 0.5 * (window[-1] - head + 1) / (0.5 * (np.abs(window[-1]) + np.abs(head) + 1))
    for item in window[:-1]:
        if item != 0.0 or window[0] != 0.0:
            s_0 += (item - head)/((np.abs(item) + np.abs(head)) * 0.5)  #
    return s_0


@jit
def grey_incidence_degree(val_1, val_2, c=3.0):
    num = 1.0 + abs(val_1) + abs(val_2)
    return num / (num + c * abs(val_1 - val_2))
    # return 1 / ( 1.0 + 0.01 * abs(val_1 - val_2)/val_1)


@jit
def grey_incidence_degree_ratio(val_1, val_2, c=1.0):
    return 1.0 / (1.0 + c * abs(val_1 - val_2))


@jit
def get_rolling_window(obs, idx, n):
    """ Get window of size n with idx being start (inclusive)."""
    # this can be either way, we look forward but can look back
    # return obs[idx - n:idx]
    return obs[idx:idx + n]


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


def get_grey_model_from_generator(time, data,  window_size=1, c=3, c_ratio=3, shock_intervals=None, non_shock_intervals=None, with_progress=False):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    my_data = np.abs(data)
    begin = 0
    shock = False
    attacks, not_attacks = 0, 0
    # This is a generator
    grey_model_gen = grey_model_generator(my_data, window_size, c=c, c_ratio=c_ratio)
    items = tqdm(grey_model_gen) if with_progress else grey_model_gen
    for idx, is_attack in enumerate(items, start=window_size):
        # if is_attack:
        #     attack = True
        if is_attack and not shock:  # If detected attack and not in shock state, change state
            non_shocks.append((time[begin], time[idx]))
            shock = True
            begin = idx
            attacks += 1
        elif not is_attack and shock:
            shocks.append((time[begin], time[idx]))
            shock = False
            begin = idx
            not_attacks += 1
        # attack = False
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
    print('grey model generator')
    shock_intervals_gen, non_shock_intervals_gen = get_grey_model_from_generator(
        time, data, window_size=1000, c=3.0, c_ratio=0.5,
        with_progress=with_progress)
    gen_fig = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
    print('Grey Systems Modelling')
    shock_intervals, non_shock_intervals = get_grey_model(
        time, data, window_size=1000, c=3.0, c_ratio=0.5,
        with_progress=with_progress)
    fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    # return None, None, None
    return shock_intervals, non_shock_intervals, fig

