from matplotlib import pyplot as plt


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

