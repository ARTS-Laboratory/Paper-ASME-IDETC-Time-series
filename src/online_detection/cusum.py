# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:24:29 2023

@author: localuser
"""
import numpy as np
import matplotlib.pyplot as plt

from fig_funcs.detection_plots import plot_shock
from utils.read_data import read_data_from_file


def get_cusum_revised(time, data, num_samples, shock_intervals=None, non_shock_intervals=None):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    avg, cusum, cusum_prev = 0, 0, 0
    begin = 0
    shock = False
    # Start of algorithm
    for i in range(num_samples):
        avg = ((avg * i) + np.abs(data[i])) / (i + 1)
        cusum += np.abs(data[i]) - avg
        if (cusum > cusum_prev+1.5) and not shock:
            non_shocks.append((time[begin], time[i - 1]))
            begin = i
            shock = True
        elif (cusum < cusum_prev+1.5) and shock:
            shocks.append((time[begin], time[i - 1]))
            begin = i
            shock = False
        cusum_prev = cusum
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def old_get_cusum(times, data, num_samples, shock_intervals=None, non_shock_intervals=None):
    """ """
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    cusum = 0
    cusum_prev = 0
    shock = 0
    begin = 0
    n = 1
    avg = 0
    # Variables instantiated
    for i in range(num_samples):
        avg = ((avg * (n - 1)) + np.abs(data[i])) / n
        cusum += np.abs(data[i]) - avg
        n += 1
        if (cusum > cusum_prev+1.5) and shock==0:
            non_shock_intervals.append((times[begin], times[i-1]))
            begin = i
            shock = 1
        elif (cusum < cusum_prev+1.5) and shock==1:
            shock_intervals.append((times[begin], times[i-1]))
            begin = i
            shock = 0
        cusum_prev = cusum
    if shock:
        shock_intervals.append((times[begin], times[times.__len__()-1]))
    else:
        non_shock_intervals.append((times[begin], times[times.__len__()-1]))


def get_plot_cusum(file_path):
    times, data = read_data_from_file(file_path)
    num_samples = 500000  # Temporary magic number for data length
    shock_intervals, non_shock_intervals = get_cusum_revised(times, data, num_samples)
    fig = plot_shock(times, data, shock_intervals, non_shock_intervals)
    return shock_intervals, non_shock_intervals, fig
