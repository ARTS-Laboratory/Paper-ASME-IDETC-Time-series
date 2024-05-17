import ruptures as rpt
import matplotlib.pyplot as plt


def plot_breaks(signal, bkps, show=False):
    fig, axes = rpt.show.display(signal, bkps)
    if show:
        plt.show()
    return fig
