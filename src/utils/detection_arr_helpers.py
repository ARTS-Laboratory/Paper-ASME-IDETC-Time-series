import numpy as np


def convert_interval_indices_to_full_arr(shocks, nonshocks, arr_size):
    """ Converts intervals of indices to filled array"""
    arr = np.empty(arr_size)
    for start, end in shocks:
        arr[start: end] = 1
    for start, end in nonshocks:
        arr[start: end] = 0
    return arr


def intervals_to_dense_arr(time, shock_intervals, non_shock_intervals):
    """ Converts intervals of time to filled array"""
    cache = make_time_index_cache(time)
    shock_times = convert_intervals_to_time_memoized(time, shock_intervals, cache)
    nonshock_times = convert_intervals_to_time_memoized(time, non_shock_intervals, cache)
    dense_arr = convert_interval_indices_to_full_arr(shock_times, nonshock_times, len(time))
    return dense_arr


def make_time_index_cache(time_arr):
    """ """
    return {time_val: idx for idx, time_val in enumerate(time_arr)}


def get_mean_array(time, data, shock_intervals, non_shock_intervals):
    """ """
    mean_levels = np.zeros_like(data)  # time or data, shouldn't matter
    mean_array_helper(time, data, shock_intervals, mean_levels)
    mean_array_helper(time, data, non_shock_intervals, mean_levels)
    return mean_levels


def convert_intervals_to_time(time, intervals):
    if isinstance(time, list):
        return [(time.index(start), time.index(stop)) for start, stop in intervals]
    elif isinstance(time, np.ndarray):
        return [(np.where(time == start)[0][0], np.where(time == stop)[0][0]) for start, stop in intervals]


def convert_intervals_to_time_v2(time, intervals, cache_dict=None):
    if cache_dict is not None:
        convert_intervals_to_time_memoized(time, intervals, cache_dict)
    else:
        return convert_intervals_to_time(time, intervals)


def convert_intervals_to_time_memoized(time, intervals, cache_dict):
    if isinstance(time, list):
        return [
            conversion_list_helper(time, start, stop, cache_dict)
            for start, stop in intervals]
    elif isinstance(time, np.ndarray):
        return [
            conversion_array_helper(time, start, stop, cache_dict)
            for start, stop in intervals]


def conversion_list_helper(time, start, stop, cache):
    # idx_0, idx_1 = None, None
    if start in cache:
        idx_0 = cache[start]
    else:
        idx_0 = time.index(start)
        cache[start] = idx_0
    if stop in cache:
        idx_1 = cache[stop]
    else:
        idx_1 = time.index(stop)
        cache[stop] = idx_1
    return idx_0, idx_1


def conversion_array_helper(time, start, stop, cache):
    if start in cache:
        idx_0 = cache[start]
    else:
        idx_0 = np.where(time == start)[0][0]
        cache[start] = idx_0
    if stop in cache:
        idx_1 = cache[stop]
    else:
        idx_1 = np.where(time == stop)[0][0]
        cache[stop] = idx_1
    return idx_0, idx_1


def mean_array_helper(time, data, intervals, mean_levels):
    """ """
    for (start_time, stop_time) in convert_intervals_to_time(time, intervals):
        if start_time == stop_time:
            mean_levels[start_time] = data[start_time]
        else:
            mean_levels[start_time:stop_time] = np.mean(data[start_time:stop_time])


def dev_array_helper(time, data, intervals, dev_levels):
    """ """
    for (start_time, stop_time) in convert_intervals_to_time(time, intervals):
        if start_time == stop_time:
            dev_levels[start_time] = 0.0
        else:
            dev_levels[start_time:stop_time] = np.std(data[start_time:stop_time])


def get_deviation_array(time, data, shock_intervals, non_shock_intervals):
    """ """
    dev_levels = np.zeros_like(data) # time or data, shouldn't matter
    dev_array_helper(time, data, shock_intervals, dev_levels)
    dev_array_helper(time, data, non_shock_intervals, dev_levels)
    return dev_levels


def get_std_ratio(time, data, shock_intervals, non_shock_intervals):
    """ """
    devs = get_deviation_array(time, data, shock_intervals, non_shock_intervals)
    base_val = devs[0]
    return devs / base_val

