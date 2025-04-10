from collections.abc import Callable

import pandas as pd
import sklearn.metrics

from config_parsers.eval_parsers import Metric, MetricOptions
from evals.metrics import get_earliest_correct, get_detect_delay


def eval_metrics(time, ground, predictions_list, metrics: [Metric], algorithm_names):
    """ """
    # this section is still a copy and paste from somewhere else
    # maybe we can get it from a config somehow?
    signal_names = [f'signal{idx + 1}' for idx in range(len(ground))]
    ids = (('signal_id', signal_names), ('algorithm', algorithm_names))
    # End of section
    metric_dict = dict(ids)
    for metric in metrics:
        # use func to get function, name to store in dict
        match metric.func:
            case MetricOptions.ACCURACY:
                scores = eval_metrics_helper_1(
                    ground, predictions_list, sklearn.metrics.accuracy_score)
            case MetricOptions.PRECISION:
                scores = eval_metrics_helper_1(
                    ground, predictions_list, sklearn.metrics.precision_score)
            case MetricOptions.RECALL:
                scores = eval_metrics_helper_1(
                    ground, predictions_list, sklearn.metrics.recall_score)
            case MetricOptions.F1_SCORE:
                scores = eval_metrics_helper_1(
                    ground, predictions_list, sklearn.metrics.f1_score)
            case MetricOptions.EARLIEST_CORRECT:
                [
                    get_earliest_correct(time, ground, predictions)
                    for predictions in predictions_list]
            case MetricOptions.DETECT_DELAY:
                earliest_correct = (get_earliest_correct(time, ground, predictions)
                    for predictions in predictions_list)
                scores = [
                    get_detect_delay(time, ground, earliest)
                    for earliest in earliest_correct]
            case _:
                raise ValueError(f'Metric {metric.name} not supported')
        metric_dict[metric.name] = scores
    df = pd.DataFrame(metric_dict)
    return df

def eval_metrics_helper_1(ground, predictions_list, metric_func: Callable):
    """ """
    scores = [
        metric_func(ground, predictions)
        for predictions in predictions_list]
    return scores
