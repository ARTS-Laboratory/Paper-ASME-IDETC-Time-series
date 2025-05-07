import numpy as np
import matplotlib.pyplot as plt


def plot_vibration(time_vec, response):
    """ Plot vibration response over time."""
    fig, ax = plt.subplots(layout='constrained')
    ax.plot(time_vec, response)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_xlim((time_vec[0], time_vec[-1]))
    return fig, ax

def plot_vibration_with_force_input(time_vec, response, force_input):
    """ """
    fig, ax = plot_vibration(time_vec, response)
    ax.plot(time_vec, force_input, linestyle='dashdot')
    return fig, ax


def plot_vibration_and_shock_region_from_difference(time_vec, response_1, response_2):
    """ plot figure of response with and without impulse."""
    threshold = 0.1
    region_indices = np.argwhere(np.abs(response_2 - response_1) >= threshold)
    region_start, region_end = region_indices[0, 0], region_indices[-1, 0]
    fig, ax = plt.subplots(layout='constrained')
    ax.plot(time_vec, response_1)
    # ax.plot(time_vec, response_2)
    ax.axvspan(time_vec[region_start], time_vec[region_end], color='orange', linewidth=0, alpha=0.25)
    ax.set_xlim((time_vec[0], time_vec[-1]))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    return fig, ax
