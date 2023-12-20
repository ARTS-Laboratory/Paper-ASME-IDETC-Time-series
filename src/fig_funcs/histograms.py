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
