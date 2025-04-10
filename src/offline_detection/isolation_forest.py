"""

    Modified from code by Justice Adjei Owusu 2025
"""

from sklearn.ensemble import IsolationForest

from offline_detection.offline_model_helpers import dense_to_time_intervals, estimator_prediction, \
    format_single_feature_data


def get_iso_forest(train_data, _train_labels, time_vec, data):
    """ """
    model = IsolationForest(n_estimators=300, contamination=0.201)
    reshaped_training, reshaped_data = format_single_feature_data(train_data, data)
    labels = None
    predictions = estimator_prediction(
        model, reshaped_training, labels, reshaped_data)
    # 1.0 is normal, -1.0 is abnormal
    out = (prediction != 1.0 for prediction in predictions)
    shocks, non_shocks = dense_to_time_intervals(time_vec, out)
    return shocks, non_shocks
