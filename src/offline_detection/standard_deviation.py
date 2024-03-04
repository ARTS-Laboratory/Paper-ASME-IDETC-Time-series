# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:24:29 2023

@author: localuser
"""

import numpy as np

from fig_funcs.detection_plots import plot_shock
from utils.read_data import read_data_from_file


def find_change_interval(data):
    return np.std(data)


def get_stdev_intervals_v2(
        time, data, num_samples, interval_size=2_000, stddev_shock=5, deviation=None,
        shock_intervals=None, non_shock_intervals=None):
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    devs = [] if deviation is None else deviation
    # Algorithm
    num_intervals = int(num_samples / interval_size)
    for idx in range(0, num_samples, interval_size):
        low, high = idx, idx + interval_size
        interval_stddev = find_change_interval(data[low:high])
        deviation.append(interval_stddev)
        if interval_stddev > stddev_shock:
            shock_intervals.append((time[low], time[high - 1]))
        else:
            non_shock_intervals.append((time[low], time[high - 1]))
    return shocks, non_shocks, devs


def get_stdev_intervals(
        time, data, num_samples, interval_size=2_000, stddev_shock=5, deviation=None,
        shock_intervals=None, non_shock_intervals=None):
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    devs = [] if deviation is None else deviation
    # interval_size = 2000
    # stddev_shock = 5
    num_intervals = int(num_samples / interval_size)
    for idx in range(num_intervals):
        low = interval_size*idx
        high = (interval_size-1)*(idx+1)+idx
        interval_stddev = find_change_interval(data[low:high])
        deviation.append(interval_stddev)
        if interval_stddev > stddev_shock:
            shock_intervals.append((time[low], time[high - 1]))
        else:
            non_shock_intervals.append((time[low], time[high - 1]))
    return shocks, non_shocks, devs


def get_plot_std_dev(file_path):
    times, data = read_data_from_file(file_path)
    num_samples = 500000  # Temporary magic number for data length
    shock_intervals, non_shock_intervals, deviation = get_stdev_intervals_v2(times, data, num_samples)
    fig = plot_shock(times, data, shock_intervals, non_shock_intervals)
    return shock_intervals, non_shock_intervals, fig
