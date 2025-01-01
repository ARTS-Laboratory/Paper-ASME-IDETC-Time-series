import matplotlib.pyplot as plt


def abs_mean_hist(data):
    # figsize = (,)
    xlabel, ylabel = 'absolute mean (m/s^2)', 'count'
    fig = plt.figure()
    plt.hist(data, bins=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig


def rms_hist(data):
    # figsize = (,)
    xlabel, ylabel = 'rms (m/s^2)', 'count'
    fig = plt.figure()
    plt.hist(data, bins=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig


def skewness_hist(data):
    # figsize = (,)
    xlabel, ylabel = 'skewness (m/s^2)', 'count'
    fig = plt.figure()
    plt.hist(data, bins=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig


def freq_center_hist(data):
    # figsize = (,)
    xlabel, ylabel = 'frequency center Hz', 'count'
    fig = plt.figure()
    plt.hist(data, bins=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig


def plot_metric_histogram(left_data, right_data, num_bins=32):
    fig, ax = plt.subplots(ncols=2, sharey=True)
    ax[0].hist(left_data, bins=num_bins)
    ax[1].hist(right_data, bins=num_bins)
    return fig
