from matplotlib import pyplot as plt

from fig_funcs.signal_plots import plot_signal, plot_signal_fft, plot_signal_power_spectrum
from online_detection.cusum import get_plot_cusum
from online_detection.dni import get_plot_dni
from online_detection.expect_Max import get_plot_expectation_maximization
import cProfile

from utils.matplotlib_formatting import set_rc_params
from utils.read_data import get_data


def make_signal_plots(file_path):
    """ """
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    fig = plot_signal(time, data, ms=True)
    normal_fft_fig = plot_signal_fft(time, data)
    normal_per_fig = plot_signal_power_spectrum(time, data)

    # plt.savefig('figures/signal_fig.jpg', dpi=350)
    normal_fig_2 = plot_signal(time[:100_000], data[:100_000], ms=True)
    normal_fft_fig_2 = plot_signal_fft(time[:100_000], data[:100_000])
    normal_per_fig_2 = plot_signal_power_spectrum(time[:100_000], data[:100_000])
    plt.show()
    # return normal_fig, normal_fft_fig


def profile():
    cProfile.run('main()', sort='ncalls')


def main():
    set_rc_params()
    file_path = './data/Dataset-7-forced-vibration-and-shock/data/dataset-A/inputData1_raw.txt'
    # make_signal_plots(file_path)
    # plt.show()
    # file_path_2 = './data/Dataset-7-forced-vibration-and-shock/data/dataset-A/outData1_filtered.txt'
    # make_signal_plots(file_path_2)
    # plt.show()
    # exit()
    # my_data = get_data(file_path)
    # time, data = my_data[:, 0], my_data[:, 1]
    # fig = plot_signal(time, data, ms=True)
    # plt.savefig('figures/signal_fig.jpg', dpi=350)
    # plt.show()
    # _, _, dni_fig = get_plot_dni(file_path)
    # plt.savefig('figures/dni_fig.jpg', dpi=350)
    _, _, cusum_fig = get_plot_cusum(file_path)
    # plt.savefig('figures/cusum_revised_fig.jpg', dpi=350)
    # plt.savefig('figures/simple_cusum_fig.jpg', dpi=350)
    # plt.savefig('figures/cusum_fig.jpg', dpi=350)
    # plt.savefig('figures/cusum_alg_fig.jpg', dpi=350)
    _, _, exp_max_fig = get_plot_expectation_maximization(file_path, True)
    # plt.savefig('figures/expectation_maximization_fig.jpg', dpi=350)
    plt.show()


if __name__ == '__main__':
    # profile()
    main()
