import numpy as np
from matplotlib import pyplot as plt


def get_mean_array(time, data, shock_intervals, non_shock_intervals):
    """ """
    mean_levels = np.zeros_like(data) # time or data, shouldn't matter
    for (start, stop) in shock_intervals:
        mean_levels[start:stop] = np.mean(data[start:stop])
    for (start, stop) in non_shock_intervals:
        mean_levels[start:stop] = np.mean(data[start:stop])
    return mean_levels


def get_deviation_array(time, data, shock_intervals, non_shock_intervals):
    """ """
    dev_levels = np.zeros_like(data) # time or data, shouldn't matter
    for (start, stop) in shock_intervals:
        dev_levels[start:stop] = np.std(data[start:stop])
    for (start, stop) in non_shock_intervals:
        dev_levels[start:stop] = np.std(data[start:stop])
    return dev_levels


def get_std_ratio(time, data, shock_intervals, non_shock_intervals):
    """ """
    devs = get_deviation_array(time, data, shock_intervals, non_shock_intervals)
    base_val = devs[0]
    return devs / base_val


def plot_shock(time, data, shock_intervals, non_shock_intervals):
    fig, ax = plt.subplots()

    ax.plot(time, data, color='black')
    for start, end in shock_intervals:
        ax.axvspan(start, end, facecolor='green', alpha=0.3)
    for start, end in non_shock_intervals:
        ax.axvspan(start, end, facecolor='red', alpha=0.3)
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Acceleration(m/s\u00b2)')
    ax.set_title('Forced Vibration And Shock (Green=Shock, Red=Non-shock)')
    plt.tight_layout()
    return fig, ax


def plot_shock_v2(time, data, shock_intervals, non_shock_intervals):
    fig, ax = plt.subplots()

    ax.plot(time, data, color='black')
    for start, end in shock_intervals:
        ax.axvspan(start, end, facecolor='blue', alpha=0.3)
    for start, end in non_shock_intervals:
        ax.axvspan(start, end, facecolor='orange', alpha=0.3)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('acceleration (m/s\u00b2)')
    ax.set_title('Forced Vibration And Shock')
    plt.legend()
    plt.tight_layout()
    return fig, ax

