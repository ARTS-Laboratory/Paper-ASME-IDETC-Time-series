import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np


def guess_breaks(signal, model_type='l2'):
    model = model_type  # "l2"
    noise = 10
    alg = rpt.Binseg(model=model).fit(signal)
    my_bkps_pen = alg.predict(epsilon=3.5 * len(signal) * noise ** 2)
    return my_bkps_pen


def get_breaks(signal, num_bkps, model_type='l2'):
    model = model_type  # "l2"
    alg = rpt.Binseg(model=model).fit(signal)
    my_bkps = alg.predict(n_bkps=num_bkps)
    return my_bkps


def plot_breaks(signal, num_bkps):
    fig, axes = rpt.show.display(signal, num_bkps)
    # plt.show()
