import matplotlib.pyplot as plt
import numpy as np


def abs_mean_shaded(time, data, threshold):
    fig = plt.figure()
    plt.plot(time, data)
    max_y = np.max(data)
    min_y = np.min(data)
    plt.fill_between(time, min_y, max_y, where=data >= threshold, color='orange', alpha=0.2)
    plt.fill_between(time, min_y, max_y, where=data < threshold, color='blue', alpha=0.2)
    return fig

# def abs_mean_shaded(time, data, threshold):
#     fig = plt.figure()
#     plt.plot(time, data)
#     max_y = np.max(data)
#     min_y = np.min(data)
#     plt.fill_between(time, min_y, max_y, where=data >= threshold, color='orange', alpha=0.2)
#     plt.fill_between(time, min_y, max_y, where=data < threshold, color='blue', alpha=0.2)
#     return fig
#
# def abs_mean_shaded(time, data, threshold):
#     fig = plt.figure()
#     plt.plot(time, data)
#     max_y = np.max(data)
#     min_y = np.min(data)
#     plt.fill_between(time, min_y, max_y, where=data >= threshold, color='orange', alpha=0.2)
#     plt.fill_between(time, min_y, max_y, where=data < threshold, color='blue', alpha=0.2)
#     return fig


def frequency_center_shaded(time, data, threshold):
    fig = plt.figure()
    plt.plot(time, data)
    max_y = np.max(data)
    min_y = np.min(data)
    plt.fill_between(time, min_y, max_y, where=data >= threshold, color='orange', alpha=0.2)
    plt.fill_between(time, min_y, max_y, where=data < threshold, color='blue', alpha=0.2)
    return fig
