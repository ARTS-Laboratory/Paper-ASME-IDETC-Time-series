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


def plot_shock(time, data, shock_intervals, non_shock_intervals, title=True):
    fig = plt.figure(figsize=(6.5, 2), layout='compressed')
    # Plot acceleration
    plt.plot(time, data, color='black')
    plt.xlim((time[0], time[-1]))
    ax = plt.gca()
    # Shade regions
    plot_shock_helper(ax=ax, interval=shock_intervals, color='red', alpha=0.3)
    plot_shock_helper(ax=ax, interval=non_shock_intervals, color='blue', alpha=0.3)
    # Setting plot details
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (m/s\u00b2)')
    if title:
        plt.title('Forced Vibration And Shock (Blue=Non-shock, Red=Shock)')
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

