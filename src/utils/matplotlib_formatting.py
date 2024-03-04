import matplotlib.pyplot as plt


def set_rc_params():
    """ Set default rcparams for matplotlib.pyplot."""
    # set default fonts and plot colors
    plt.rcParams.update({'text.usetex': False})
    plt.rcParams.update({'image.cmap': 'viridis'})
    plt.rcParams.update({'font.serif': [
        'Times New Roman', 'Times', 'DejaVu Serif',
        'Bitstream Vera Serif', 'Computer Modern Roman',
        'New Century Schoolbook', 'Century Schoolbook L', 'Utopia',
        'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino',
        'Charter', 'serif']})
    plt.rcParams.update({'font.family': 'serif'})
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'mathtext.rm': 'serif'})
    # plt.rcParams.update({'mathtext.fontset': 'custom'}) # I don't think I need this as its set to 'stixsans' above.

    cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
