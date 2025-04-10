from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from utils.read_data import load_signals

class GroundSource(Enum):
    SAVE = 'save'
    GENERATE = 'generate'

class OfflineAlgorithm(Enum):
    BIN_SEG = 'binary_segmentation'
    BOTTOM_UP = 'bottom_up'
    DYN_PROG = 'dynamic_programming'

class MetricOptions(Enum):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1-score'
    EARLIEST_CORRECT = 'earliest-correct'
    DETECT_DELAY = 'delay'

@dataclass
class GroundSaveData:
    time: np.ndarray
    data: np.ndarray

@dataclass
class GroundGenerateData:
    algorithm: OfflineAlgorithm
    extras: dict | None
    name: str
    save: bool = False
    save_location: Path = None

def parse_eval_ground_config(ground_config: dict):
    """ Parse evaluation config and return evaluation config object."""
    source_type_name = ground_config['what']
    # todo this can fail if source_type_name is not in enum, among other things.
    source_type = GroundSource(source_type_name)
    match source_type:
        case GroundSource.SAVE:
            config = ground_config['save']
            save_dir = config['dir']
            save_name = config['name']
            time, data = load_signals(Path(save_dir, save_name))
            ground = GroundSaveData(time, data)
        case GroundSource.GENERATE:
            config = ground_config['generate']
            alg = OfflineAlgorithm(config['alg'])
            name: str = config['name']
            extras = config['extras'] if 'extras' in config else None
            save_ground: bool = config['save']
            if save_ground:
                save_dir = config['dir']
                save_name = config['save-name']
                save_loc = Path(save_dir, save_name)
            else:
                save_loc = None
            ground = GroundGenerateData(alg, extras, name, save_ground, save_loc)
        case _:
            raise NotImplementedError(f'Cannot take eval data of type {source_type}')
    return ground

@dataclass
class Metric:
    func: MetricOptions
    # name: str
    table_name: str
    # units:


def parse_eval_metrics_config(metrics_config: dict):
    """ Parse evaluation config and return evaluation config object."""
    metric_funcs = metrics_config['functions']
    scores = metrics_config['scores']
    unit_conversions = metrics_config['unit_conversions']
    metrics = list()
    for score in scores:
        match score:
            case 'accuracy':
                func = MetricOptions.ACCURACY
                metric_func = metric_funcs['accuracy']
            case 'precision':
                func = MetricOptions.PRECISION
                metric_func = metric_funcs['precision']
            case 'f1-score':
                func = MetricOptions.F1_SCORE
                metric_func = metric_funcs['f1-score']
            case 'earliest-correct':
                func = MetricOptions.EARLIEST_CORRECT
                metric_func = metric_funcs['earliest-correct']
            case 'delay':
                func = MetricOptions.DETECT_DELAY
                metric_func = metric_funcs['delay']
            case _:
                raise NotImplementedError(f'Cannot take eval data of type {score}')
        table_name = metric_func['table-name']
        units = metric_func['units']
        metric = Metric(func=func, table_name=table_name)
        metrics.append(metric)
    return metrics

@dataclass
class Save:
    save: bool


# def parse_eval_saving_config(saving_config: dict):
#     """ Parse saving subdirectory of evaluation config."""
#     save_eval: bool = saving_config['save']

