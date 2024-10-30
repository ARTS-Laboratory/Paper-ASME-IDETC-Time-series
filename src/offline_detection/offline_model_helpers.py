

def estimator_prediction(estimator, fitting_data, data):
    """ """
    # todo finish this function
    estimator.fit(fitting_data[0], fitting_data[1])
    predictions = [estimator.predict(data_point) for data_point in data]
    # for data_point in data:
    #     estimator.predict(data_point)
    return predictions


def dense_to_time_intervals(time_vec, predictions):
    """ Convert dense array of predictions to time interval"""
    shocks = list()
    steady = list()
    begin = 0
    shock = False
    for idx, is_change in enumerate(predictions):
        if is_change and not shock:
            steady.append((time_vec[begin], time_vec[idx]))
            shock = True
            begin = idx
        elif not is_change and shock:
            shocks.append((time_vec[begin], time_vec[idx]))
            shock = False
            begin = idx
    if shock:
        shocks.append((time_vec[begin], time_vec[-1]))
    else:
        steady.append((time_vec[begin], time_vec[-1]))
    return shocks, steady
