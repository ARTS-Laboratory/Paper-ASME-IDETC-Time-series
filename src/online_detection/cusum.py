# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:24:29 2023

@author: localuser
"""
import numpy as np

from fig_funcs.detection_plots import plot_shock
from utils.read_data import read_data_from_file


def cusum(time, data, mean, sigma, alpha, beta, shock_intervals=None, non_shock_intervals=None):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    shock = False
    begin = 0
    # next_data = list()
    m_bar = [mean]
    h = 5 * sigma
    cp, cn = [0], [0]
    var = sigma**2
    accumulator = data[0]
    for idx, val in enumerate(data[1:], start=1):
        accumulator += val
        # m_bar.append(beta*m_bar[idx - 1] + (1 - beta) * data[idx])
        m_bar.append(beta * m_bar[idx - 1] - (1 - beta) * val)
        m_bar[0], m_bar[1] = mean, mean
        mean_p = accumulator / (idx + 1)
        # next_data.append(abs(data[idx] - data[idx - 1]) - m_bar[idx - 1])
        # These are the original lines
        # cp.append(max(0, cp[idx - 1] + (alpha*(m_bar[idx] - mean)/sigma**2)*(data[idx]-(m_bar[idx]-mean)-alpha*(m_bar[idx]-mean)*0.5)))
        # cn.append(max(0, cn[idx - 1] - (alpha*(m_bar[idx] - mean)/sigma**2)*(data[idx]+(m_bar[idx]-mean)+alpha*(m_bar[idx]-mean)*0.5)))
        # The following  5 lines should be equivalent to the 2 above
        # diff = m_bar[idx - 1] - mean
        diff = m_bar[idx - 1] - mean_p
        alpha_diff = alpha * diff
        alpha_diff_var, alpha_diff_half = alpha_diff / var, alpha_diff * 0.5
        cp.append(max(0, cp[idx - 1] + alpha_diff_var * (val - diff - alpha_diff_half)))
        cn.append(max(0, cn[idx - 1] - alpha_diff_var * (val + diff + alpha_diff_half)))
        # End of conversion
        # attack_likely = (np.abs(cn[idx]) > h or cp[idx] > h)

        attack_likely = (cp[idx] > h or cn[idx] > h)
        if attack_likely:
            cp[idx], cn[idx] = 0, 0
            if shock:
                shocks.append((time[begin], time[idx]))
            else:
                non_shocks.append((time[begin], time[idx]))
            begin = idx
            shock = not shock
        # if np.abs(cn[idx]) > h or cp[idx] > h:
        #     cp[idx] = 0
        #     cn[idx] = 0
        # if attack_likely and not shock:
        #     non_shocks.append((time[begin], time[idx - 1]))
        #     begin = idx
        #     shock = True
        # elif not attack_likely and shock:
        #     shocks.append((time[begin], time[idx - 1]))
        #     begin = idx
        #     shock = False
    # print(m_bar)
    # print(cp), print(cn)
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def simple_cusum(time, data, mean, std_dev, k=1, h=5, shock_intervals=None, non_shock_intervals=None):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    cp, cn = [0], [0]
    values = (data - mean) / std_dev
    shock = False
    begin = 0
    for idx, value in enumerate(values[1:], start=1):
        cp.append(max(0, value - k + cp[idx - 1]))
        cn.append(max(0, -value - k + cn[idx - 1]))
        attack_likely = (cn[idx] > h or cp[idx] > h)
        if attack_likely and not shock:
            non_shocks.append((time[begin], time[idx - 1]))
            begin = idx
            shock = True
        elif not attack_likely and shock:
            shocks.append((time[begin], time[idx - 1]))
            begin = idx
            shock = False
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def cusum_alg(time, data, mean, std_dev, h, alpha, shock_intervals=None, non_shock_intervals=None):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    cp, cn, d, mu = [0], [0], [0], [data[0]]
    begin = 0
    shock = False
    # mu = None
    h_val = h * std_dev
    variance = std_dev**2
    # values = (data - mean) / std_dev
    accumulator = data[0]
    for idx, event in enumerate(data[1:], start=1):
        # Continue updating mean
        accumulator += event
        mean_p = accumulator / (idx + 1)

        weight = alpha * d[idx - 1] / variance
        cp.append(max(0, cp[idx - 1] + weight * (event - d[idx - 1] - alpha * d[idx - 1] * 0.5)))
        cn.append(max(0, cn[idx - 1] - weight * (event + d[idx - 1] + alpha * d[idx - 1] * 0.5)))
        # d.append(mu[idx - 1] - mean)  # Original: get difference
        # d.append(max(0, np.abs(mu[idx - 1] - mean)))  # most current, absolute diff
        d.append(mu[idx - 1] - mean_p)  # Based on accumulator, use updated mean
        mu.append(alpha * mu[idx - 1] - (1 - alpha) * event)
        # mu.append(alpha * mu[idx - 1] - (1 - alpha) * event)
        # # d.append(mu[idx - 1] - mean)
        # d.append(max(0, np.abs(mu[idx - 1] - mean)))
        # weight = alpha * d[idx] / variance
        # cp.append(max(0, cp[idx - 1] + weight * (event - d[idx] - alpha * d[idx] * 0.5)))
        # cn.append(max(0, cn[idx - 1] - weight * (event + d[idx] + alpha * d[idx] * 0.5)))
        # attack_likely = (cp[idx] > h or cn[idx] > h)
        attack_likely = (cp[idx] > h_val or cn[idx] > h_val)
        if attack_likely:
            cp[idx], cn[idx] = 0, 0
            if shock:
                shocks.append((time[begin], time[idx - 1]))
            else:
                non_shocks.append((time[begin], time[idx - 1]))
            begin = idx
            shock = not shock
        # old style marking
        # if attack_likely and not shock:
        #     cp[idx], cn[idx] = 0, 0
        #     # event is an attack
        #     non_shocks.append((time[begin], time[idx - 1]))
        #     begin = idx
        #     shock = True
        # elif not attack_likely and shock:
        #     shocks.append((time[begin], time[idx - 1]))
        #     begin = idx
        #     shock = False
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


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
            cusum = 0.0
        elif (cusum < cusum_prev+1.5) and shock:
            shocks.append((time[begin], time[i - 1]))
            begin = i
            shock = False
            cusum = 0.0
        cusum_prev = cusum
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def get_plot_cusum(file_path):
    times, data = read_data_from_file(file_path)
    # shock_intervals, non_shock_intervals = get_cusum_revised(times, data, num_samples)
    mean, std = np.mean(data[:1_000]), np.std(data[:1_000])
    # shock_intervals, non_shock_intervals = simple_cusum(times, data, mean, std)
    shock_intervals, non_shock_intervals = cusum(times, data, mean, std, alpha=0.025, beta=0.025)  # , alpha=0.025, beta=0.025
    # shock_intervals, non_shock_intervals = cusum_alg(times, data, mean, std, h=5, alpha=0.25)
    fig = plot_shock(times, data, shock_intervals, non_shock_intervals)
    return shock_intervals, non_shock_intervals, fig
