import itertools
import matplotlib.collections

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from utils import metrics
from utils.detection_arr_helpers import get_mean_array, convert_intervals_to_time, get_deviation_array


def interval_histogram(time, data, shock_intervals, non_shock_intervals, title=True):
    """ """
    shock_times = convert_intervals_to_time(time, shock_intervals)
    non_shock_times = convert_intervals_to_time(time, non_shock_intervals)
    fig, ax = plt.subplots(ncols=2)
    ax[0].hist([metrics.abs_mean(data[start:stop]) for start, stop in shock_times])
    ax[1].hist([metrics.abs_mean(data[start:stop]) for start, stop in non_shock_times])
    return fig


def raw_histogram(time, data, shock_intervals, non_shock_intervals, title=True):
    """ """
    n_bins = 50
    shock_times = convert_intervals_to_time(time, shock_intervals)
    non_shock_times = convert_intervals_to_time(time, non_shock_intervals)
    # These flatten the list of lists into one big list
    # shock_vals = [point for start, stop in shock_times for point in data[start:stop]]
    # non_shock_vals = [point for start, stop in non_shock_times for point in data[start:stop]]
    fig, ax = plt.subplots(ncols=2, sharey=True)
    ax[0].hist(np.array(list(itertools.chain([data[start:stop] for start, stop in shock_times]))), bins=n_bins)
    ax[1].hist(np.array(list(itertools.chain([data[start:stop] for start, stop in non_shock_times]))), bins=n_bins)
    return fig


def plot_shock_helper(ax, interval, color, alpha):
    """ Helper function to plot shading rectangles as collection.

        :param plt.Axes ax: Axes to plot rectangles.
        :param List[tuple[float, float]] interval: Iterable of tuples containing start and stop times.
        :param str color: Color of rectangles.
        :param float alpha: Transparency of rectangles.
    """
    ax.add_collection(matplotlib.collections.PatchCollection(
        (Rectangle((start, 0), stop - start, 1.0) for start, stop in interval),
        alpha=alpha, facecolor=color, transform=ax.get_xaxis_transform()))


def plot_shock(time, data, shock_intervals, non_shock_intervals, to_ms=False):
    # fig = plt.figure(figsize=(6.5, 2), layout='compressed')
    # ax = plt.gca()
    fig, ax = plt.subplots(figsize=(6.5, 2), layout='compressed')
    safe_color = 'blue'
    unsafe_color = 'red'
    shade_alpha = 0.3
    y_min = -200
    y_max = 200
    if to_ms:
        scalar = 1_000
        ms_time = time * scalar
        # Plot acceleration
        ax.plot(ms_time, data, color='black')
        ax.set_xlim((ms_time[0], ms_time[-1]))
        new_shocks = ((start * scalar, stop * scalar) for start, stop in shock_intervals)
        new_non_shocks = ((start * scalar, stop * scalar) for start, stop in non_shock_intervals)
        plot_shock_helper(ax=ax, interval=new_shocks, color=unsafe_color, alpha=shade_alpha)
        plot_shock_helper(ax=ax, interval=new_non_shocks, color=safe_color, alpha=shade_alpha)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('acceleration (m/s\u00b2)')
    else:
        # Plot acceleration
        ax.plot(time, data, color='black')
        ax.set_xlim((time[0], time[-1]))
        # Shade regions
        plot_shock_helper(ax=ax, interval=shock_intervals, color=unsafe_color, alpha=shade_alpha)
        plot_shock_helper(ax=ax, interval=non_shock_intervals, color=safe_color, alpha=shade_alpha)
        # Setting plot details
        ax.set_xlabel('time (s)')
        ax.set_ylabel('acceleration (m/s\u00b2)')
    ax.set_ylim((y_min, y_max))
    safe_rect = Rectangle((0, 0), 1, 1, facecolor=safe_color, alpha=shade_alpha)
    unsafe_rect = Rectangle((0, 0), 1, 1, facecolor=unsafe_color, alpha=shade_alpha)
    ax.legend((safe_rect, unsafe_rect), ('normal region', 'shock region'), bbox_to_anchor=(1, 1), loc='lower right', ncol=2)
    # plt.tight_layout()
    return fig


def plot_shock_w_mean_std(time, data, shock_intervals, non_shock_intervals):
    means = get_mean_array(time, data, shock_intervals, non_shock_intervals)
    devs = get_deviation_array(time, data, shock_intervals, non_shock_intervals)
    fig = plt.figure()
    # Plot acceleration
    plt.plot(time, data, color='black')
    plt.plot(time, means, '--', color='black')
    plt.plot(time, means + devs, ':', color='black')
    plt.plot(time, means - devs, ':', color='black')
    # Shade regions
    ax = plt.gca()
    plot_shock_helper(ax, shock_intervals, 'red', 0.3)
    plot_shock_helper(ax, non_shock_intervals, 'blue', 0.3)
    # Setting plot details
    plt.xlim((time[0], time[-1]))
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (m/s\u00b2)')
    plt.title('Forced Vibration And Shock (Blue=Shock, Red=Non-shock)')
    plt.tight_layout()
    return fig

