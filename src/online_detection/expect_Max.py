import math

import numpy as np

from line_profiler import profile
from numba import njit
from typing import List, Optional, Iterable
from tqdm import tqdm

from online_detection.model_helpers import detection_to_intervals_for_generator_v1, \
    detection_to_intervals_for_generator_v1_with_progress
from src.fig_funcs.detection_plots import plot_shock
from src.utils.read_data import get_data


def expectation_maximization(
        safe, not_safe, unknown, mean_1, mean_2, var_1,
        var_2, pi, epochs=1, converge_threshold=1e-3):
    """ Perform expectation maximization on one unknown.

        :param safe: Data that is known to be safe.
        :param not_safe: Data that is not known to be safe.
        :param unknown: Data that needs to be classified.
        :param float mean_1: Estimated mean for safe data.
        :param float mean_2: Estimated mean for unsafe data.
        :param float var_1: Estimated variance for safe data.
        :param float var_2: Estimated variance for unsafe data.
        :param float pi: Estimated probability that an attack has occurred.
        :param int epochs: Number of epochs to update parameters.
        :returns: Tuple of (attack classification, updated mean 1,
        updated mean 2, updated variance 1, updated variance 2,
         updated attack probability.
        :rtype: (bool, float, float, float, float, float)
        """
    data = np.concatenate((safe, not_safe, (unknown,)))
    # todo this may need to loop till convergence
    # Variable initialization
    size = len(data)
    mu1_hat, mu2_hat = mean_1, mean_2
    sig1_hat, sig2_hat = var_1, var_2
    pi_hat = pi
    # For some number of epochs, iterate over given data until convergence
    last_attack_prob = np.full_like(data, fill_value=-1e99)
    for idx in range(epochs):
        # Expectation
        # TODO confirm that the following 3 lines are equivalent
        # attack_prob = posterior_probs(data, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat)
        # inverse = posterior_probs(data, 1 - pi_hat, mu1_hat, sig1_hat, mu2_hat, sig2_hat)
        attack_prob, inverse = posterior_probs_v2(
            data, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat)
        mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat = maximization(
            data, attack_prob, inverse, mu1_hat, mu2_hat, sig1_hat, sig2_hat,
            pi_hat, size)
        if close_enough(attack_prob, last_attack_prob):
            break
        last_attack_prob[:] = attack_prob
    is_attack = posterior_prob(
        data[-1], pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat) > 0.01
    return is_attack, mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat


# @njit
def expectation_maximization_generator(
        safe, not_safe, unknowns, mean_1, mean_2, var_1,
        var_2, pi, epochs=1):
    """ Perform expectation maximization on one unknown.

        :param safe: Data that is known to be safe.
        :param not_safe: Data that is not known to be safe.
        :param unknowns: Collection of data that needs to be classified.
        :param float mean_1: Estimated mean for safe data.
        :param float mean_2: Estimated mean for unsafe data.
        :param float var_1: Estimated variance for safe data.
        :param float var_2: Estimated variance for unsafe data.
        :param float pi: Estimated probability that an attack has occurred.
        :param int epochs: Number of epochs to update parameters.
        :returns: Tuple of (attack classification, updated mean 1,
        updated mean 2, updated variance 1, updated variance 2,
         updated attack probability.
        :rtype: (bool, float, float, float, float, float)
        """
    data = np.concatenate((safe, not_safe, np.empty(1)))
    # Variable initialization
    size = len(data)
    mu1_hat, mu2_hat = mean_1, mean_2
    sig1_hat, sig2_hat = var_1, var_2
    pi_hat = pi
    last_attack_prob = np.empty_like(data)
    last_attack_prob[:] = -1e99
    out = np.empty((2, size))
    attack_prob, inverse = out[0], out[1]
    for unknown in unknowns:
        data[-1] = unknown # reassign last value to our new unknown
        # For some number of epochs, iterate over given data until convergence
        for idx in range(epochs):
            # Expectation
            posterior_probs_v2_inplace(
                data, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat, out)
            # attack_prob, inverse = posterior_probs_v2(
            #     data, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat)
            # attack_prob, inverse = posterior_probs_v3(
            #     data, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat)
            # Maximization
            mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat = maximization(
                data, attack_prob, inverse, mu1_hat, mu2_hat,
                sig1_hat, sig2_hat, pi_hat, size)
            if close_enough(attack_prob, last_attack_prob):
                break
            last_attack_prob[:] = attack_prob
        is_attack = posterior_prob(
            unknown, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat) > 0.01
        yield is_attack


@njit
def close_enough(a, b):
    return np.allclose(a, b) and np.allclose(b, a)


@njit
def maximization(
        data, attack_prob, inverse, mu1_hat, mu2_hat, sig1_hat, sig2_hat,
        pi_hat, size):
    """
    """
    density, inverse_density = attack_prob.sum(), inverse.sum()
    # If all probabilities are zero for attack or not attack, no need to update
    if not (density == 0 or inverse_density == 0):
        new_mu1_hat, new_mu2_hat = update_means(
            attack_prob, inverse, density, inverse_density, data)
        new_sig1_hat, new_sig2_hat = update_variances(
            attack_prob, inverse, density, inverse_density, data, mu1_hat,
            mu2_hat)
        new_pi_hat = update_attack_prob(density, size)
        return new_mu1_hat, new_mu2_hat, new_sig1_hat, new_sig2_hat, new_pi_hat
    return mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat


# @njit
# def phi_v2(value, mean, variance):
#     """ Return the probability density function for value.
#
#         :param float value: Value to get function for.
#         :param float mean: Population mean.
#         :param float variance: Population variance.
#         :returns: PDF of value given.
#         :rtype: float
#     """
#     if variance == 0.0:
#         return np.where(value == mean, 1.0, 0.0)
#     ex = np.exp(-0.5*(value - mean)**2/variance)
#     ex /= np.sqrt(variance*2*np.pi)
#     return ex


@njit
def phi_single(value, mean, variance):
    """ Return the probability density function for a single value.

        :param float value: Value to get function for.
        :param float mean: Population mean.
        :param float variance: Population variance.
        :returns: PDF of value given.
        :rtype: float
    """
    if variance == 0.0:
        return 1.0 if value == mean else 0.0
    denom = math.sqrt(variance*2*np.pi)
    ex = math.exp(-0.5*(value - mean)**2/variance)
    return ex/denom


# # @profile
# @njit
# def phi(values, mean, variance):
#     """ Return the probability density function for some values.
#
#         :param ndarray values: Value to get function for.
#         :param float mean: Population mean.
#         :param float variance: Population variance.
#         :returns: PDF of value given.
#         :rtype: float
#     """
#     if variance == 0.0:
#         return np.where(values == mean, 1.0, 0.0)
#     ex = values - mean
#     ex **= 2
#     ex *= -0.5 / variance
#     np.exp(ex, ex)
#     return ex / math.sqrt(variance)


# @profile
@njit
def phi_inplace(values, mean, variance, out):
    """ Return the probability density function for some values.

        :param ndarray values: Value to get function for.
        :param float mean: Population mean.
        :param float variance: Population variance.
        :returns: PDF of value given.
        :rtype: float
    """
    if variance == 0.0:
        for idx, value in enumerate(values):
            out[idx] = 1.0 if value == mean else 0.0
        return
    out[:] = values
    out -= mean
    out **= 2
    out *= -0.5 / variance
    np.exp(out, out)
    out /= math.sqrt(variance)


@njit
def posterior_prob(point, attack_prob, attack_mean, attack_var, normal_mean, normal_var):
    """ Calculate probability of latent variable for given data point."""
    # Probability of attack * Probability of point occurring if it was an attack
    # Divided by probability of point occurring
    num = attack_prob * phi_single(point, attack_mean, attack_var)
    denom = num + (1 - attack_prob) * phi_single(point, normal_mean, normal_var)
    if denom == 0.0:
        # print(f'There exists a zero here: {num}')
        return num
    return num / denom


# @profile
@njit
def posterior_probs_v2(points, attack_prob, attack_mean, attack_var, normal_mean, normal_var):
    """ Calculate probabilities of each latent variable for each data point."""
    num_1 = np.empty_like(points)
    phi_inplace(points, attack_mean, attack_var, num_1)
    # num_1 = phi(points, attack_mean, attack_var)
    num_1 *= attack_prob
    num_2 = np.empty_like(points)
    phi_inplace(points, normal_mean, normal_var, num_2)
    num_2 *= (1 - attack_prob)
    denom = num_1 + num_2
    normalize_probs(num_1, num_2, denom)
    return num_1, num_2


@njit
def normalize_probs(probs, inverse, denom):
    for idx, value in enumerate(denom):
        if value != 0.0:
            probs[idx] /= value
            inverse[idx] /= value


# @profile
@njit
def posterior_probs_v2_inplace(points, attack_prob, attack_mean, attack_var, normal_mean, normal_var, out):
    """ Calculate probabilities of each latent variable for each data point."""
    num_1 = out[0]
    num_2 = out[1]
    phi_inplace(points, attack_mean, attack_var, out[0])
    num_1 *= attack_prob
    # new lines
    phi_inplace(points, normal_mean, normal_var, num_2)
    num_2 *= (1 - attack_prob)
    denom = num_1 + num_2
    normalize_probs(num_1, num_2, denom)


@njit
def update_means(probs, inverse, density, inverse_density, events):
    """ Return updated values for means.

        :param probs: List of probabilities of attack.
        :type probs: List[float] or np.ndarray
        :param inverse: List of probabilities of safe.
        :type inverse: List[float] or np.ndarray
        :param float density: Probability density for mean 1.
        :param float inverse_density: Probability density for mean 1.
        :param events: List of events corresponding to probs.
        :rtype: (float, float)
        :returns: tuple of updated means (mean 1, mean 2)"""
    mean_1 = np.dot(inverse, events) / inverse_density
    mean_2 = np.dot(probs, events) / density
    return mean_1, mean_2
    # mean_1_denom = inverse_density
    # mean_2_denom = density
    # mean_1_num = np.dot(inverse, events)
    # mean_2_num = np.dot(probs, events)
    # mean_1 = mean_1_num / mean_1_denom
    # mean_2 = mean_2_num / mean_2_denom
    # return mean_1, mean_2


@njit
def variance_helper(probs, events, mean):
    """ """
    return np.dot(probs, np.square(events - mean))


@njit
def update_variances(probs, inverse, density, inverse_density, events, mean_1, mean_2):
    """ Return updated variances.


        :param probs: List of probabilities of attack.
        :type probs: List[float] or np.ndarray
        :param inverse: List of probabilities of safe.
        :type inverse: List[float] or np.ndarray
        :param float density: Probability density for mean 1.
        :param float inverse_density: Probability density for mean 1.
        :param events: List of events corresponding to probs.
        :param float mean_1: Approximate mean for safe data.
        :param float mean_2: Approximate mean for unsafe data.
        :returns: Tuple of mean for safe data and mean for unsafe data.
        :rtype: (float, float)
    """
    var_1 = variance_helper(inverse, events, mean_1) / inverse_density
    var_2 = variance_helper(probs, events, mean_2) / density
    return var_1, var_2


@njit
def update_attack_prob(density, size):
    """ Return updated attack probability.

        :param float density: Probability density of data.
        :param int size: Length of data vector.
        :returns: Updated attack probability.
        :rtype: float
    """
    return density / size


def get_expectation_max(
        time, normal_obs, abnormal_obs, unknowns, mean_1=None, mean_2=None, var_1=None, var_2=None, pi=None,
        shock_intervals=None, non_shock_intervals=None, epochs=1, with_progress=False):
    """ Get and return probable shock events computed by expectation maximization.

        :rtype: (List[(int, int)], List[(int, int)])
        :returns: (list of shocks, list of non-shocks)
        """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    begin = 0
    shock = False
    # get params theta = mu , mu2, sig, sig2, pi
    mean_1_p = mean_1 if mean_1 is not None else np.mean(normal_obs)
    mean_2_p = mean_2 if mean_2 is not None else np.mean(abnormal_obs)
    var_1_p = var_1 if var_1 is not None else np.var(normal_obs)
    var_2_p = var_2 if var_2 is not None else np.var(abnormal_obs)
    if pi is not None:
        pi_p = pi
    else:
        normal_size, ab_size = len(normal_obs), len(abnormal_obs)
        pi_p = ab_size / (normal_size + ab_size)
    # Begin algorithm loop
    elapsed = 0
    my_normal_obs, my_abnormal_obs, my_unknowns = np.asarray(normal_obs), np.asarray(abnormal_obs), np.asarray(unknowns)
    items = tqdm(enumerate(my_unknowns), total=len(unknowns)) if with_progress else enumerate(my_unknowns)
    for idx, unknown in items:
        # start = perf_counter()
        attack, mean_1_p, mean_2_p, var_1_p, var_2_p, pi_p = expectation_maximization(
            normal_obs, abnormal_obs, unknown, mean_1_p, mean_2_p, var_1_p,
            var_2_p, pi_p, epochs=epochs)
        # stop = perf_counter()
        # elapsed += stop - start
        # if idx % 10_000 == 9_999:
        #     print(f'Function took {elapsed / 10_000} sec to run on average over last 10,000 loops, {elapsed} sec total.')
        #     elapsed = 0
        if attack and not shock:  # If detected attack and not in shock state, change state
            non_shocks.append((time[begin], time[idx]))
            shock = True
            begin = idx
        elif not attack and shock:
            shocks.append((time[begin], time[idx]))
            shock = False
            begin = idx
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def get_expectation_maximization_model_from_generator(
        time, normal_obs, abnormal_obs, unknowns, mean_1=None, mean_2=None,
        var_1=None, var_2=None, pi=None, shock_intervals=None,
        non_shock_intervals=None, epochs=1, with_progress=False):
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    begin = 0
    shock = False
    # get params theta = mu , mu2, sig, sig2, pi
    mean_1_p = mean_1 if mean_1 is not None else np.mean(normal_obs)
    mean_2_p = mean_2 if mean_2 is not None else np.mean(abnormal_obs)
    var_1_p = var_1 if var_1 is not None else np.var(normal_obs)
    var_2_p = var_2 if var_2 is not None else np.var(abnormal_obs)
    if pi is not None:
        pi_p = pi
    else:
        normal_size, ab_size = len(normal_obs), len(abnormal_obs)
        pi_p = ab_size / (normal_size + ab_size)
    # Begin algorithm loop
    my_normal_obs, my_abnormal_obs, my_unknowns = np.asarray(normal_obs), np.asarray(abnormal_obs), np.asarray(unknowns)
    em_model_gen = expectation_maximization_generator(
        my_normal_obs, my_abnormal_obs, my_unknowns, mean_1_p, mean_2_p,
        var_1_p, var_2_p, pi_p, epochs)
    if with_progress:
        shocks, non_shocks = detection_to_intervals_for_generator_v1_with_progress(
            time, begin, em_model_gen, len(unknowns))
    else:
        shocks, non_shocks = detection_to_intervals_for_generator_v1(
        time, begin, em_model_gen)
    # if with_progress:
    #     em_model_gen = tqdm(expectation_maximization_generator(
    #     my_normal_obs, my_abnormal_obs, my_unknowns, mean_1_p, mean_2_p,
    #     var_1_p, var_2_p, pi_p, epochs), total=len(unknowns))
    # else:
    #     em_model_gen = expectation_maximization_generator(
    #     my_normal_obs, my_abnormal_obs, my_unknowns, mean_1_p, mean_2_p,
    #     var_1_p, var_2_p, pi_p, epochs)
    # shocks, non_shocks = detection_to_intervals_for_generator_v1(
    #     time, begin, em_model_gen)
    return shocks, non_shocks


def get_plot_expectation_maximization(file_path, with_progress=False):
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    print(len(my_data[:, 0]))
    mean_1 = np.mean(data[:100_000])
    var_1 = np.var(data[:100_000])
    mean_2 = np.mean(data[200_000:400_000])
    var_2 = np.var(data[200_000:400_000])
    safe = np.random.normal(mean_1, var_1, 64)
    unsafe = np.random.normal(mean_2, var_2, 64)
    print(f'Means: {mean_1}, {mean_2}')
    print(f'Variances: {var_1}, {var_2}')
    # shock_intervals, non_shock_intervals = get_expectation_max(
    #     time, data[:7], data[7:10], data, mean_1=mean_1, var_1=var_1,
    #     mean_2=mean_2, var_2=var_2, epochs=50, with_progress=with_progress)
    shock_intervals_gen, non_shock_intervals_gen = get_expectation_maximization_model_from_generator(
        time, safe, unsafe, data, mean_1=mean_1, var_1=var_1,
        epochs=50, with_progress=with_progress)
    fig_gen = plot_shock(time, data, shock_intervals_gen, non_shock_intervals_gen)
    shock_intervals, non_shock_intervals = get_expectation_max(
        time, safe, unsafe, data, mean_1=mean_1, var_1=var_1,
        epochs=50, with_progress=with_progress)
    fig = plot_shock(time, data, shock_intervals, non_shock_intervals)
    return shock_intervals, non_shock_intervals, fig
