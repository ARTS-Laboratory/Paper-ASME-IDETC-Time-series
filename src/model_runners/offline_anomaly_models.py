from enum import Enum
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from fig_funcs.detection_plots import plot_shock
# from main import print_scores
from offline_detection.isolation_forest import get_iso_forest
from offline_detection.svm import get_svm_model


# from utils.detection_arr_helpers import intervals_to_dense_arr

class AnomalyType(Enum):
    SVM = 'svm'
    ISO_FOREST = 'iso_forest'


def run_offline_anomaly_models(time, safe, unsafe, data, models):
    """ """
    sample_data = np.concatenate((safe, unsafe), axis=0)
    sample_labels = np.concatenate((np.ones_like(safe), -np.ones_like(unsafe)), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(sample_data, sample_labels)
    results = list()
    for model in models:
        match model: # this is the type of model, will change later
            case AnomalyType.SVM:
                shocks, non_shocks = get_svm_model(x_train, y_train, time, data)
            case AnomalyType.ISO_FOREST:
                shocks, non_shocks = get_iso_forest(x_train, y_train, time, data)
            case _:
                raise ValueError
        results.append((model, shocks, non_shocks))
    return results


def plot_detection_anomaly_models(time, data, results, save_dir, save_name):
    """ """
    dpi = 350
    to_ms = True
    for (model, shocks, non_shocks) in results:
        detection_fig = plot_shock(time, data, shocks, non_shocks, to_ms=to_ms)
        plt.savefig(Path(save_dir, Path(save_name).with_suffix('.png')), dpi=dpi)
        plt.close(detection_fig)

# def plot_offline_anomaly_detection(time, safe, unsafe, data):
#     """ """
#     dpi = 350
#     to_ms = True
#     sample_data = np.concatenate((safe, unsafe), axis=0)
#     sample_labels = np.concatenate((np.ones_like(safe), -np.ones_like(unsafe)), axis=0)
#     x_train, x_test, y_train, y_test = train_test_split(sample_data, sample_labels)
#     shocks, non_shocks = get_iso_forest(x_train, y_train, time, data)
#     detection_fig = plot_shock(time, data, shocks, non_shocks, to_ms=to_ms)
#     plt.savefig(Path(save_path, 'nonparametric_fig.png'), dpi=dpi)
#     plt.close(detection_fig)
#     pred = intervals_to_dense_arr(time, shocks, non_shocks)
#     print_scores(time, ground, pred)
#     return pred
