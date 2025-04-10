from pathlib import Path

import numpy as np


def read_data_from_file(file_path):
    times, data = [], []
    with open(file_path, 'r') as file:
        for line in file:
            time, value = line.strip().split('\t')
            times.append(float(time))
            data.append(float(value))
    return times, data


def get_data(filename):
    my_data = np.loadtxt(filename)
    return my_data


def load_signals(file_path: (Path | str)) -> tuple[np.ndarray, np.ndarray]:
    """ Load signals in from file."""
    if isinstance(file_path, Path):
        my_path = file_path
    elif isinstance(file_path, str):
        my_path = Path(file_path)
    else:
        raise TypeError('file_path must be a string or Path')
    match my_path.suffix:
        case '.txt':
            my_data = get_data(file_path)
        case '.npy':
            my_data = np.load(file_path)
        case _:
            raise NotImplementedError
    time, data = my_data[:, 0], my_data[:, 1]
    return time, data
