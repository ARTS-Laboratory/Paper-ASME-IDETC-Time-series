import math
import os

import numpy as np

from pathlib import Path

from examples import vibration_funcs

# Some constants
DATA_DIR = Path(os.pardir, 'data', 'example-data', '')
DATA_NAME = 'example-1.npy'

TIME_START = 0
TIME_END = 20
TIME_NUM_POINTS = 500

DEFAULT_C = 2
DEFAULT_K = 60
DEFAULT_M = 5
DEFAULT_CRIT_DAMP_RATIO = 0.1
DEFAULT_FORCE_FREQ_COEFF = 1.5

def get_time_vector():
    """ Generate and return time vector."""
    start = TIME_START
    end = TIME_END
    num_points = TIME_NUM_POINTS
    time_vec = np.linspace(start, end, num_points)
    return time_vec


def make_simulated_data() -> tuple[np.ndarray, np.ndarray]:
    """ """
    # parameters for time vector
    # start, end, num_points = 0, 20, 500
    # damping coefficient, spring constant, mass
    c, k, m = 2, 60, 5
    # calculated values - natural frequency, critical damping ratio, damped natural frequency
    nat_freq = vibration_funcs.natural_frequency(k, m)
    crit_damp_ratio = 0.1  # vibration_funcs.critical_damping_ratio(c, m, nat_freq)
    damped_nat_freq = vibration_funcs.damped_natural_frequency(
        nat_freq, crit_damp_ratio)
    # vectors
    time_vec = get_time_vector()
    tau_vec = get_time_vector()
    # time_vec = np.linspace(start, end, num_points)
    # tau_vec = np.linspace(start, end, num_points)
    # make forcing vector
    force_freq = 1.5 * nat_freq
    force_vec_2 = 2 * m / (nat_freq ** 2 - force_freq ** 2) * np.cos(force_freq * tau_vec)
    force_vec_2[100:100 + math.ceil(25 * 2 * math.pi/nat_freq)] += 100 * m / 2
    # generate vibration response
    impulsed_vibe = vibration_funcs.arbitrary_input_response_arr(
        time_vec, tau_vec, force_vec_2, damped_nat_freq, nat_freq, m, crit_damp_ratio)
    return time_vec, impulsed_vibe

def make_vibration_signal():
    """ Generate and return vibration signal."""
    pass

def save_signal(signal: np.ndarray, save_path: os.PathLike):
    """ Save given signal to path provided."""
    np.save(save_path, signal)

def generate_data():
    data_dir = DATA_DIR
    file_name = DATA_NAME
    data_path = Path(data_dir, file_name)
    time, data = make_simulated_data()
    combined = np.column_stack((time, data))
    save_signal(combined, data_path)

if __name__ == '__main__':
    generate_data()