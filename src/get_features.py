import numpy as np
import scipy as sp
from utils import metrics
import matplotlib.pyplot as plt

from offline_detection import binary_segmentation
from src.fig_funcs import shaded_plots, signal_plots
from utils.read_data import get_data


def feature_cloud(data):
    fig = plt.figure()
    plt.plot(data)


def get_slides(data_vec, vec_length):
    sliced_data = [data_vec[i: i + vec_length] for i in range(len(data_vec) - vec_length)]
    return sliced_data


def slide_metrics(metric, data_vec, vec_length):
    sliced_data = get_slides(data_vec, vec_length)
    metric_vals = list(map(metric, sliced_data))
    return metric_vals


def temp_ruptures():
    file_1 = '../data/dataset-A/inputData1_raw.txt'
    # vec_length = 50
    my_data = get_data(file_1)
    time, data = my_data[:, 0], my_data[:, 1]
    time_step = metrics.sampling_frequency(data)
    sig_fig = signal_plots.plot_signal(time, data)
    fft_fig = signal_plots.plot_signal_fft(time, data)
    # plt.show()
    # exit()
    # res = bottom_up.get_breaks(data, 3)
    # bottom_up.plot_breaks(data, 3, res)

    res = binary_segmentation.guess_breaks(data)
    binary_segmentation.plot_breaks(data, res)


def get_show_features():
    file_1 = '../data/dataset-A/inputData1_raw.txt'
    vec_length = 500
    my_data = get_data(file_1)
    time, data = my_data[:, 0], my_data[:, 1]
    time_step = metrics.sampling_frequency(data)
    abs_val = metrics.abs_mean(data)
    # Let's look at the data
    sig_fig = signal_plots.plot_signal(time, data)

    fft_fig = signal_plots.plot_signal_fft(time, data)
    # plt.show()
    print(my_data.shape)
    abs_mean_slid = slide_metrics(metrics.abs_mean, data, vec_length)
    time_slide = np.asarray(get_slides(time, vec_length))
    slided_data = np.asarray(get_slides(data, vec_length))
    abs_shade = np.zeros_like(data)
    for idx in range(vec_length, len(abs_shade)):
        abs_shade[idx] = metrics.abs_mean(my_data[idx - vec_length: idx, 1])
    fc_shade = np.zeros_like(data)
    for idx in range(vec_length, len(fc_shade)):
        fc_shade[idx] = metrics.frequency_center(my_data[idx - vec_length: idx, 1], time_step)
    fig_shaded_1 = shaded_plots.abs_mean_shaded(time, abs_shade, 5)
    fig_shaded_8 = shaded_plots.frequency_center_shaded(time, fc_shade, 13_000)
    # fig_1 = histograms.abs_mean_hist(list(map(metrics.abs_mean, slided_data)))
    # fig_2 = histograms.rms_hist(list(map(metrics.rms, slided_data)))
    # fig_3 = histograms.skewness_hist(list(map(metrics.skewness, slided_data)))
    #
    # fig_8 = histograms.freq_center_hist([metrics.frequency_center(vec, time_step) for vec in slided_data])
    plt.show()
    # print(abs_mean_slid)
    print(sp.stats.describe(data))
    obj = sp.stats.describe(abs_mean_slid)
    print(obj)


if __name__ == '__main__':
    alt_main()
