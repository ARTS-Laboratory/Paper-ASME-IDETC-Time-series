import ruptures as rpt
import matplotlib.pyplot as plt


def get_breaks(signal, num_bkps, model_type='l2'):
    model = model_type  # "l2"
    alg = rpt.BottomUp(model=model).fit(signal)
    my_bkps = alg.predict(num_bkps)
    return my_bkps


def plot_breaks(signal, my_bkps):
    rpt.show.display(signal, my_bkps)
    plt.show()
