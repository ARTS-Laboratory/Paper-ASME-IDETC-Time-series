from functools import cache

import numpy as np
from numba import njit
from tqdm import tqdm

from fig_funcs.detection_plots import plot_shock
from online_detection.model_helpers import detection_to_intervals_for_generator_v1
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


@njit
def grey_model_generator(data, window_size=1, c=3, c_ratio=300):
    """ Return Grey Model predictions for given data.

        :param np.ndarray data: Array of data. Data expected to be non-negative.
        :param int window_size: Size of window to iterate over array.
        :param float c: Constant multiplier for degree of grey incidence.
        :param float c_ratio: Constant multiplier for degree of ratio grey incidence.
        :returns:
        :rtype: None"""
    sample_arr = np.empty((3, window_size))
    window_1 = sample_arr[0]
    x_1 = sample_arr[1]
    z_1 = sample_arr[2]
    get_rolling_window_inplace(data, 0, window_size, window_1)
    accumulation_sequence_inplace(window_1, x_1)
    mean_sequence_inplace(x_1, z_1, alpha=0.5)
    s_1 = behavioral_sequence(z_1)
    # s_1_ratio = behavioral_sequence_ratio(z_1)
    # s_1_ratio = behavioral_sequence_ratio_2(z_1)
    s_1_rel_dist = behavior_relative_difference(z_1)
    out_arr = np.empty((3, window_size))
    window_2 = out_arr[0]
    x_2 = out_arr[1]
    z_2 = out_arr[2]
    for idx in range(0, len(data) - window_size):
        get_rolling_window_inplace(data, idx, window_size, window_2)
        accumulation_sequence_inplace(window_2, x_2)
        mean_sequence_inplace(x_2, z_2, alpha=0.5)
        s_2 = behavioral_sequence(z_2)
        s_2_ratio = behavioral_sequence_ratio(z_2)
        s_2_ratio = behavioral_sequence_ratio_2(z_2)
        degree = grey_incidence_degree(s_1, s_2, c=c)
        degree_ratio = grey_incidence_degree_ratio(s_1_ratio, s_2_ratio, c=c_ratio)
        attack_likely = degree + degree_ratio <= 0.5
        yield attack_likely


@njit
def accumulation_sequence(window: np.ndarray):
    """ Return the accumulation over the window."""
    # return np.add.accumulate(window)
    output = window.cumsum()
    output[-1] = window.sum()
    return output


@njit
def accumulation_sequence_inplace(window: np.ndarray, out: np.ndarray):
    """ Return the accumulation over the window."""
    # return np.add.accumulate(window)
    out[:] = window.cumsum()
    out[-1] = window.sum()


@njit
def mean_sequence(window, alpha=0.5):
    """ Return the running average over a window."""
    transformed = np.empty_like(window)
    transformed[0] = window[0]
    val = window[0]
    neg_alpha = 1 - alpha
    for idx, item in enumerate(window[1:]):
        val = (val * alpha) + (item * neg_alpha)
        transformed[idx] = val
    return transformed


@njit
def mean_sequence_inplace(window, out, alpha=0.5):
    """ Return the running average over a window."""
    out[0] = window[0]
    val = window[0]
    neg_alpha = 1 - alpha
    for idx in range(1, len(window)):
        out[idx] = (window[idx - 1] * alpha) + (window[idx] * neg_alpha)


@njit
def behavioral_sequence(window):
    s_0 = (window[:-1] - window[0]).sum() + 0.5 * (window[-1] - window[0])
    return s_0


@njit
def behavioral_sequence_ratio(window, offset_2=1.0):
    # This assumes window is composed only of nonnegative numbers
    # new code
    s_0 = (window[:-1] + offset_2).sum()
    s_0 += 0.5 * (window[-1] + offset_2)
    s_0 /= window[0] + offset_2
    # old code
    # s_0 = np.sum((window[:-1] + offset_2) / (window[0] + offset_2)) + 0.5 * ((window[-1] + offset_2) / (window[0] + offset_2))
    return s_0


@njit
def behavioral_sequence_ratio_2(window):
    """ """
    head = window[0]
    s_0 = 0.5 * (window[-1] - head + 1) / (0.5 * (np.abs(window[-1]) + np.abs(head) + 1))
    for item in window[:-1]:
        if item != 0.0 or window[0] != 0.0:
            s_0 += (item - head)/((np.abs(item) + np.abs(head)) * 0.5)  #
    return s_0


@njit
def behavior_relative_difference(window):
    """ """
    head = window[0]
    tail = window[-1]
    if head == 0 and tail == 0:
        s_0 = 0
    else:
        s_0 = 0.5 * abs(tail - head) / ( 0.5 * (abs(tail) + abs(head)))
    if head == 0.0:
        for item in window[1:]:
            if item != 0.0:
                s_0 += abs(item - head) / (0.5 * (abs(item) + abs(head)))
            # if head and item are zero, no difference
    else:
        for item in window[1:]:
            s_0 += abs(item - head) / (0.5 * (abs(item) + abs(head)))
    return s_0


@njit
def behavior_log_difference(window):
    """ """
    head = window[0]
    tail = window[-1]
    match (head, tail):
        case (0, 0):
            s_0 = 0.0
        case (0, tail):
            s_0 = math.inf
        case (head, 0):
            s_0 = -math.inf
        case (head, tail):
            s_0 = 0.5 * math.log(tail / head)
        case _:
            raise NotImplementedError
    for item in window[1:]:
        match (head, tail):
            case (0, 0):
                s_0 += 0.0
            case (0, tail):
                s_0 += math.inf
            case (head, 0):
                s_0 += -math.inf
            case (head, tail):
                s_0 += math.log(item / head)
            case _:
                raise NotImplementedError
    return s_0


@njit
def grey_incidence_degree(val_1, val_2, c=3.0):
    num = 1.0 + abs(val_1) + abs(val_2)
    return num / (num + c * abs(val_1 - val_2))
    # return 1 / ( 1.0 + 0.01 * abs(val_1 - val_2)/val_1)


@njit
def grey_incidence_degree_ratio(val_1, val_2, c=1.0):
    return 1.0 / (1.0 + c * abs(val_1 - val_2))


@njit
def get_rolling_window(obs, idx, n):
    """ Get window of size n with idx being start (inclusive)."""
    # this can be either way, we look forward but can look back
    # return obs[idx - n:idx]
    return obs[idx:idx + n]


# @njit
def get_rolling_window_inplace(obs, idx, n, out):
    """ """
    out[:] = obs[idx:idx + n]


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
    if with_progress:
        grey_model_gen = tqdm(
            grey_model_generator(
                my_data, window_size, c=c, c_ratio=c_ratio),
            total=len(data)-window_size)
    else:
        grey_model_gen = grey_model_generator(
            my_data, window_size, c=c, c_ratio=c_ratio)
    shocks, non_shocks = detection_to_intervals_for_generator_v1(
        time, begin, grey_model_gen, start_offset=window_size)
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

