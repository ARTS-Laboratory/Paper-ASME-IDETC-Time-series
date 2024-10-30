

def detection_to_intervals_for_generator_v1(
        time_vec, begin, model_generator, start_offset=0):
    """ Convert detections from generator to time intervals

        This version is for detecting when a measure deviates from expected.
    """
    shock = False
    shocks = list()
    nonshocks = list()
    for idx, is_change in enumerate(model_generator, start=start_offset):
        if is_change and not shock:
            nonshocks.append((time_vec[begin], time_vec[idx]))
            shock = True
            begin = idx
        elif not is_change and shock:
            shocks.append((time_vec[begin], time_vec[idx]))
            shock = False
            begin = idx
    if shock:
        shocks.append((time_vec[begin], time_vec[-1]))
    else:
        nonshocks.append((time_vec[begin], time_vec[-1]))
    return shocks, nonshocks


def detection_to_intervals_for_generator_v2(time_vec, begin, model_generator):
    """

        This version is for detecting when a change has occurred.
    """
    shock = False
    shocks = list()
    nonshocks = list()
    for idx, is_change in enumerate(model_generator):
        if is_change:
            if shock:
                shocks.append((time_vec[begin], time_vec[idx]))
            else:
                nonshocks.append((time_vec[begin], time_vec[idx]))
            begin = idx
            shock = not shock
    if shock:
        shocks.append((time_vec[begin], time_vec[-1]))
    else:
        nonshocks.append((time_vec[begin], time_vec[-1]))
    return shocks, nonshocks
