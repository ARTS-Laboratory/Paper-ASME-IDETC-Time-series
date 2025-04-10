import pandas as pd
import plotly.graph_objects as go

from utils.plotly_formatting import update_font


def plot_radar_single(fig: go.Figure, values, theta, name, fill):
    """ Add radar plot for a single metric table to figure."""
    fig.add_trace(
        go.Scatterpolar(r=values, theta=theta, fill = fill, name=name))
    return fig


def plot_metric_scores(fig: go.Figure, data_collection, axis_names, labels, fill=None):
    """ """
    fill_type = fill if fill is not None else 'toself'
    # append beginning to end to close chart
    # make chart
    update_font(fig)
    for (data, label) in zip(data_collection, labels):
        plot_radar_single(fig, data, axis_names, label, fill_type)
    return fig

def plot_metric_scores_for_paper(df: pd.DataFrame):
    """ """
    fig = go.Figure()
    _metric_names = (
        'accuracy', 'precision', 'recall', 'f1 score', 'earliest correct',
        'detection delay')
    metric_pretty_names = (
        'Accuracy', 'Precision', 'Recall', 'F1 score',
        'Earliest Correct (ms)', 'Detection Delay (ms)')
    # todo confirm metrics exist in table
    # todo confirm algorithm names match
    alg_pretty_names = ('BOCPD', 'EM', 'GM', 'CUSUM')
    skip_idx = 2
    data = df.itertuples(index=False)
    data_collection = tuple(item[skip_idx:] for item in data)
    plot_metric_scores(fig, data_collection, metric_pretty_names, alg_pretty_names)
    return fig

