# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:15:54 2024

@author: localuser
"""
import numpy as np

from fig_funcs.detection_plots import plot_shock
from utils.read_data import read_data_from_file


def get_dni(
        times, data, num_samples, d_threshold, i_threshold,
        shock_intervals=None, non_shock_intervals=None):
    """ Determine if points are in state of shock based on difference between adjacent points and point threshold.

    """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    begin, shock = 0, False
    for i in range(num_samples):
        # TODO change this so we don't accidentaly check the end of list
        if ((np.abs(data[i]-data[i-1]) > d_threshold) or np.abs(data[i]) > i_threshold) and not shock:
            non_shocks.append((times[begin], times[i-1]))
            begin = i
            shock = True
        elif ((np.abs(data[i]-data[i-1]) < d_threshold) and np.abs(data[i]) < i_threshold) and shock:
            shocks.append((times[begin], times[i-1]))
            begin = i
            shock = False
    if not shock:
        non_shocks.append((times[begin], times[-1]))
    else:
        shocks.append((times[begin], times[-1]))
    return shocks, non_shocks


def get_plot_dni(file_path):
    times, data = read_data_from_file(file_path)
    num_samples = 500000  # Temporary magic number for data length
    d_threshold = 3  # Temporary magic number for parameter
    i_threshold = 15  # Temporary magic number for parameter
    shock_intervals, non_shock_intervals = get_dni(
        times, data, num_samples, d_threshold, i_threshold)
    fig = plot_shock(times, data, shock_intervals, non_shock_intervals)
    return shock_intervals, non_shock_intervals, fig
