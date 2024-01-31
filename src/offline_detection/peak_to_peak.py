# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:32:16 2023

@author: goshorna
"""

import numpy as np

from fig_funcs.detection_plots import plot_shock
from utils.read_data import read_data_from_file


def find_change_interval_v2(data):
    maximum = max(data)
    minimum = min(data)
    return maximum - minimum > 40


def find_change_interval(data):
    if data[0] > data[1]:
        minimum = data[1]
        maximum = data[0]
    else:
        minimum = data[0]
        maximum = data[1]
    for datum in data:
        if maximum < datum:
            maximum = datum
        if minimum > datum:
            minimum = datum
    if maximum - minimum > 40:
        return True
    else:
        return False


def get_peak_to_peak(
        time, data, interval_size=2_000, shock_intervals=None, non_shock_intervals=None):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    for idx in range(0, len(data), interval_size):
        low, high = idx, idx + interval_size
        if find_change_interval(data[low:high]):
            shock_intervals.append((time[low], time[high - 1]))
        else:
            non_shock_intervals.append((time[low], time[high - 1]))
    return shocks, non_shocks


def get_peak_to_peak(
        time, data, shock_intervals=None, non_shock_intervals=None):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    for i in range(500):
        low = 1000 * i
        high = 999 * (i + 1) + i
        if find_change_interval(data[low:high]):
            shock_intervals.append((time[low], time[high - 1]))
        else:
            non_shock_intervals.append((time[low], time[high - 1]))
    return shocks, non_shocks


def get_plot_peak_to_peak(file_path):
    times, data = read_data_from_file(file_path)
    shock_intervals, non_shock_intervals = get_peak_to_peak(times, data)
    plot_shock(times, data, shock_intervals, non_shock_intervals)
