import numpy as np
import scipy as sp
import metrics
import matplotlib.pyplot as plt


def get_data(filename):
    my_data = np.loadtxt(filename)
    return my_data


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

def main():
    file_1 = '../data/dataset-A/inputData1_raw.txt'
    vec_length = 50
    my_data = get_data(file_1)
    time_step = metrics.sampling_frequency(my_data[:, 1])
    abs_val = metrics.abs_mean(my_data[:, 1])
    print(my_data.shape)
    abs_mean_slid = slide_metrics(metrics.abs_mean, my_data[:, 1], vec_length)
    slided_data = np.asarray(get_slides(my_data[:, 1], vec_length))
    fig_1 = src.fig_funcs.histograms.abs_mean_hist(list(map(metrics.abs_mean, slided_data)))
    fig_2 = src.fig_funcs.histograms.rms_hist(list(map(metrics.rms, slided_data)))
    fig_3 = src.fig_funcs.histograms.skewness_hist(list(map(metrics.skewness, slided_data)))

    fig_8 = src.fig_funcs.histograms.freq_center_hist([metrics.frequency_center(vec, time_step) for vec in slided_data])
    plt.show()
    # print(abs_mean_slid)
    print(sp.stats.describe(my_data[:, 1]))
    obj = sp.stats.describe(abs_mean_slid)
    print(obj)


if __name__ == '__main__':
    main()
