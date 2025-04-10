import numpy as np


def get_earliest_correct(time, ground, predictions):
    """ """
    true_positive_indices = np.logical_and(
        ground.astype(bool), predictions.astype(bool))
    if true_positive_indices.any():
        earliest_correct = time[true_positive_indices][0]
    else:
        earliest_correct = np.inf
    return earliest_correct


def get_detect_delay(time, ground: np.ndarray, earliest_correct):
    """ """
    if ground.astype(bool).any():
        first = time[np.nonzero(ground.astype(bool))][0]
        return earliest_correct - first
    return np.inf
