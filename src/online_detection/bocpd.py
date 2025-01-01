import dataclasses
import math
import scipy

import numpy as np

from line_profiler import profile
from collections import deque
from numba import jit, njit, guvectorize, float64, vectorize
from numba.core.extending import overload
from tqdm import tqdm

from fig_funcs.detection_plots import plot_shock
try:
    from online_detection.bocpd_cy import (calculate_prior_arr_cy, calculate_prior_arr_inplace_cy,
        bayesian_online_changepoint_detection_deque_cy)
except ImportError:
    raise Warning('Expected Cython imports to be available.')
from online_detection.model_helpers import detection_to_intervals_for_generator_v1
from utils.read_data import get_data


def bayesian_online_changepoint_detection_v6_generator(data, mu, kappa, alpha, beta, lamb):
    """ """
    my_data = np.asarray(data)
    run_length = 1  # Iterations since last changepoint
    maxes = deque((0,), maxlen=2)
    run_length_arr = np.array([0], dtype=np.uint32)
    probabilities = np.array([1.0])
    accumulator = 0.0
    alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
    for idx, event in enumerate(my_data):
        probabilities, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr = calculate_probabilities_v3(
            event, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr, probabilities, lamb, trunc_threshold=1e-32)
        max_idx = find_max_cp(probabilities)
        maxes.append(run_length_arr[max_idx])
        if maxes[-1] < maxes[0]:
            # reset run length and accumulator
            run_length, accumulator = update_attack(event)
            # reset params
            probabilities = np.asarray([probabilities.sum()])
            run_length_arr = np.asarray([0], dtype=np.uint32)
            # maxes = [0]
            alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
        else:
            # update
            cp = probabilities[0]
            run_length, accumulator, alpha_arr, beta_arr, mu_arr, kappa_arr = update_no_attack_arr(
                event, run_length, cp, accumulator, alpha_arr, beta_arr, mu_arr, kappa_arr, alpha, beta, mu, kappa)
        # Calculate probability of change point
        attack_probs = calculate_prior_arr(event, alpha_arr, beta_arr, mu_arr, kappa_arr)
        attack_probs *= probabilities
        val_prob = attack_probs.sum()
        is_attack = val_prob <= 0.05
        yield is_attack


def bayesian_online_changepoint_detection_deque_generator_v2(data, mu, kappa, alpha, beta, lamb):
    """ """
    predictions = bayesian_online_changepoint_detection_deque_cy(data, mu, kappa, alpha, beta, lamb)
    print(predictions.shape)
    for prediction in predictions:
        yield prediction <= 0.05


@profile
# @njit
def calculate_probabilities_v3(
        event, alpha, beta, mu, kappa, run_lengths, probabilities, lamb, trunc_threshold=1e-16):
    """ """
    hazard = hazard_function(lamb)
    priors = calculate_prior_arr_v1(event, alpha, beta, mu, kappa)
    new_probabilities = np.zeros(probabilities.size + 1)
    # here we define the type as uint32, this is arbitrary and might need to be changed later
    new_run_lengths = np.zeros(run_lengths.size + 1, dtype=np.uint32)
    # Multiply probabilities by their priors
    priors *= probabilities
    new_probabilities[1:] += priors
    # should be fine to multiply entire vector if first element is zero
    new_probabilities *= (1 - hazard)
    new_probabilities[0] += priors.sum()
    new_probabilities[0] *= hazard
    # Normalize probabilities
    prob_sum = new_probabilities.sum()
    if prob_sum != 0.0:
        new_probabilities /= prob_sum
        # new_probabilities /= new_probabilities.sum()
    # Match the run length values with the probabilities
    # new_run_lengths[0] = 0  # don't need this line since array initialized to zeros
    new_run_lengths[1:] += run_lengths
    new_run_lengths[1:] += 1
    # Truncate near zero values
    # trunc = new_probabilities < trunc_threshold
    # new_probabilities[trunc] = 0.0
    threshold_filter = new_probabilities > trunc_threshold
    threshold_filter[0] = True
    new_probabilities = new_probabilities[threshold_filter]
    new_run_lengths = new_run_lengths[threshold_filter]
    threshold_filter = threshold_filter[1:]
    new_alpha, new_beta, new_mu, new_kappa = alpha[threshold_filter], beta[threshold_filter], mu[threshold_filter], kappa[threshold_filter]
    # new_alpha, new_beta, new_mu, new_kappa = alpha, beta, mu, kappa
    return new_probabilities, new_alpha, new_beta, new_mu, new_kappa, new_run_lengths


def update_attack(event: float):
    """ """
    # event is an attack
    accum = event
    run_length_p = 1
    return run_length_p, accum


# @njit
# def update_no_attack_v5(
#         event, run_length, cp, accumulator,
#         mu, kappa, alpha, beta):
#     """ """
#     # update
#     kappa_plus_one = kappa + 1
#     run_length_p = run_length + 1
#     kappa_p = kappa_plus_one
#     alpha_p = alpha + 0.5
#     new_accumulator = event + accumulator
#     mu_p = (kappa * mu + event) / kappa_plus_one
#     beta_p = beta + kappa * np.square(event - mu) / (2 * kappa_plus_one)
#
#     return run_length_p, new_accumulator, mu_p, kappa_p, alpha_p, beta_p


@njit
def update_no_attack_arr(
        event, run_length, cp, accumulator, alpha_arr, beta_arr, mu_arr,
        kappa_arr, alpha, beta, mu, kappa):
    """ """
    size = alpha_arr.size + 1
    # update
    mu_p = np.empty(shape=size)
    kappa_p = np.empty(shape=size)
    alpha_p = np.empty(shape=size)
    beta_p = np.empty(shape=size)

    new_accumulator = event + accumulator
    run_length_p = run_length + 1
    kappa_p[1:] = kappa_arr + 1
    alpha_p[1:] = alpha_arr + 0.5
    kappa_plus = kappa_arr + 1
    mu_p[1:] = (kappa_arr * mu_arr + event) / kappa_plus
    beta_p[1:] = beta_arr + kappa_arr * np.square(event - mu_arr) / (2 * kappa_plus)
    mu_p[0] = mu
    kappa_p[0] = kappa
    alpha_p[0] = alpha
    beta_p[0] = beta
    return run_length_p, new_accumulator, alpha_p, beta_p, mu_p, kappa_p


@njit
def calculate_prior_arr(point, alphas, betas, mus, kappas):
    """ Return student's T distribution PDF given parameters of inverse gamma distribution."""
    return t_func_arr(point, mus, ((betas * (kappas + 1.0)) / (alphas * kappas)), 2 * alphas)


@njit
def calculate_prior_arr_v1(point, alphas, betas, mus, kappas):
    """ Return student's T distribution PDF for given parameters of inverse gamma distribution."""
    t_values = calculate_prior_helper(point, alphas, betas, mus, kappas)
    t_values /= beta_numba(0.5, alphas)
    return t_values

    # # Old code
    # denom = 2 * betas * (kappas + 1.0) / kappas
    # t_values = (point - mus)**2 / denom
    # t_values += 1.0
    # # t_values **= -(alphas + 0.5)
    # exponent = -(alphas + 0.5)
    # t_values **= exponent
    # beta_result = scipy.special.beta(0.5, alphas)
    # # beta_result = beta_numba(0.5, alphas)
    # beta_result *= np.sqrt(denom)
    # t_values /= beta_result
    # return t_values


# @profile
@njit
def calculate_prior_helper(point, alphas, betas, mus, kappas):
    """ """
    denom = 2 * betas * (kappas + 1.0) / kappas
    t_values = (point - mus)**2 / denom
    t_values += 1.0
    # t_values **= -(alphas + 0.5)
    exponent = -(alphas + 0.5)
    t_values **= exponent
    t_values /= np.sqrt(denom)
    return t_values


# @profile
@njit
def calculate_prior_arr_inplace(point, alphas, betas, mus, kappas, out):
    """ """
    calculate_prior_helper_inplace(point, alphas, betas, mus, kappas, out)
    out /= beta_numba(0.5, alphas)
    # out /= scipy.special.beta(0.5, alphas)


# @njit
def calculate_prior_deque_ndarray(point, params, out):
    for idx, param in enumerate(params):
        out[idx] = calculate_prior_helper(point, param.alpha, param.beta, param.mu, param.kappa) / scipy.special.beta(0.5, param.alpha)


# @overload(scipy.special.beta)
# def scipy_beta(a, b):
#     beta_vals = scipy.special.beta(a, b)
#     def scipy_beta_fn(a, b):
#         return beta_vals
#     return scipy_beta_fn


# @njit
def find_max_cp(probs):
    return np.argmax(probs)


# @njit
def hazard_function(lam: float):
    return 1 / lam


@njit
def t_func_arr(x_bar, mu_arr, s_arr, n_arr):
    """ """
    # t_values = np.zeros_like(mu_arr)
    s_n_arr = s_arr * n_arr
    n_half = n_arr * 0.5
    t_values = ((x_bar - mu_arr)**2 / s_n_arr + 1.0) ** (-(n_half + 0.5))

    t_values /= (np.sqrt(s_n_arr) * beta_numba(0.5, n_arr / 2))
    return t_values
    # old code
    # t_values = (x_bar - mu_arr) / np.sqrt(s_arr)
    # t_values = (1.0 + np.square(t_values) / n_arr) ** (-(n_arr + 1) / 2)
    # # t_values /= (np.sqrt(n_arr) * t_func_arr_helper_beta(n_arr)) * np.sqrt(s_arr)
    # t_values /= (np.sqrt(n_arr) * beta_numba(0.5, n_arr / 2)) * np.sqrt(s_arr)
    # # t_values /= (np.sqrt(n_arr) * scipy.special.beta(0.5, n_arr / 2)) * np.sqrt(s_arr)
    # return t_values
    # coeffs = (np.sqrt(dfs) * scipy.special.beta(0.5, dfs / 2))
    # return exponentials / (coeffs * np.sqrt(s_arr))


@vectorize(['float64(float64, float64)', 'float32(float32, float32)'], cache=False, nopython=True)
def beta_numba(val_1, val_2):
    """ Return vectorized function for """
    return math.exp(math.lgamma(val_1) + math.lgamma(val_2) - math.lgamma(val_1 + val_2))


# @guvectorize(
#     ['void(float64[:], float64[:])', 'void(float32[:], float32[:])'],
#     '(n)->(n)', cache=True, nopython=True)
# def t_func_arr_helper_beta(arr, res):
#     """ Return beta function for vector"""
#     for idx, item in enumerate(arr):
#         res[idx] = math.exp(math.lgamma(item) - math.lgamma(0.5 + item))
#     res *= np.sqrt(np.pi)
#     # old
#     # gamma_1 = math.lgamma(0.5)
#     # for idx, item in enumerate(arr / 2):
#     #     res[idx] = np.exp(gamma_1 + math.lgamma(item) - math.lgamma(0.5 + item))
#
#
# @njit
# def beta_func(a, b):
#     return np.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def get_bocpd_v5_from_generator(time, data, mu, kappa, alpha, beta, lamb,
        shock_intervals=None, non_shock_intervals=None, with_progress=False):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    my_data = data  # np.abs(data)
    begin = 0
    if with_progress:
        bocpd_model_gen = tqdm(
            bayesian_online_changepoint_detection_v6_generator(
                my_data, mu, kappa, alpha, beta, lamb), total=len(data))
    else:
        bocpd_model_gen = bayesian_online_changepoint_detection_v6_generator(
            my_data, mu, kappa, alpha, beta, lamb)
    shocks, non_shocks = detection_to_intervals_for_generator_v1(
        time, begin, bocpd_model_gen)
    return shocks, non_shocks


def get_bocpd(
        time, data, mu, kappa, alpha, beta, lamb,
        shock_intervals=None, non_shock_intervals=None, with_progress=False):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    begin = 0
    shock = False
    cps, cps_2 = 0, 0
    run_length = 1  # Iterations since last changepoint
    maxes = deque((0,), maxlen=2)
    probabilities = np.array([1])
    my_data = np.abs(np.asarray(data))
    mu = np.mean(my_data[:100])
    items = tqdm(my_data, total=len(my_data)) if with_progress else my_data
    for idx, event in enumerate(items):
        is_attack = False
        probabilities, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr = calculate_probabilities_v3(event, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr, probabilities, lamb, trunc_threshold=1e-32)
        max_idx = find_max_cp(probabilities)
        maxes.append(run_length_arr[max_idx])
        if maxes[-1] < maxes[0]:
            # event is an attack
            # Update count
            cps += 1
            # reset run length and accumulator
            run_length, accumulator = update_attack(event)
            # alpha_arr, beta_arr, mu_arr, kappa_arr = np.concatenate([[alpha], alpha_arr]), np.concatenate([[beta], beta_arr]), np.concatenate([[mu], mu_arr]), np.concatenate([[kappa], kappa_arr])
            # reset params
            probabilities = np.asarray([probabilities.sum()])
            run_length_arr = np.asarray([0], dtype=np.uint32)
            # maxes = [0]
            alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
        else:
            # update
            cp = probabilities[0]
            run_length, accumulator, alpha_arr, beta_arr, mu_arr, kappa_arr = update_no_attack_arr(
                event, run_length, cp, accumulator, alpha_arr, beta_arr, mu_arr, kappa_arr, alpha, beta, mu, kappa)
        # # I think this is the probability distribution of the run length

        attack_probs = calculate_prior_arr(event, alpha_arr, beta_arr, mu_arr, kappa_arr)
        attack_probs *= probabilities
        val_prob = attack_probs.sum()
        is_attack = val_prob <= 0.05
        if is_attack and not shock:
            non_shocks.append((time[begin], time[idx + 1]))
            shock = True
            begin = idx
        elif not is_attack and shock:
            shocks.append((time[begin], time[idx + 1]))
            shock = False
            begin = idx
    print(f'Total changepoints: {cps} vs {cps_2}, max idx: {max(maxes)}')
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def get_bocpd_windowed(time, data, mu, kappa, alpha, beta, lamb,
                       window_size=1, with_progress=False):
    total_shocks, total_non_shocks = list(), list()
    if with_progress:
        itr = tqdm(range(0, len(data), window_size), total=len(data) // window_size)
    else:
        itr = range(0, len(data), window_size)
    begin = 0
    shock = False
    cps, cps_2 = 0, 0
    run_length = 1  # Iterations since last changepoint
    maxes = deque((0,), maxlen=2)
    accumulator = 0
    attack = False
    my_data = np.abs(np.asarray(data))
    mu = np.mean(my_data[:100])
    for idx in itr:
        # shocks, non_shocks = get_bocpd(
        #     time, data[idx: idx + window_size], mu, kappa, alpha, beta, lamb)
        run_length_arr = np.array([0], dtype=np.uint32)
        probabilities = np.array([1.0])
        maxes.clear()
        maxes.append(0)
        accumulator = 0.0
        alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
        for jdx, event in enumerate(my_data[idx:idx + window_size], start=idx):
            probabilities, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr = calculate_probabilities_v3(
                event, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr, probabilities, lamb, trunc_threshold=1e-32)
            max_idx = find_max_cp(probabilities)
            maxes.append(run_length_arr[max_idx])
            if maxes[-1] < maxes[0]:
                # a change definitely occurred in the past
                run_length, accumulator = update_attack(event)
                # reset params
                probabilities = np.asarray([probabilities.sum()])
                run_length_arr = np.asarray([0], dtype=np.uint32)
                # maxes = [0]
                alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
            else:
                # update
                cp = probabilities[0]
                run_length, accumulator, mu, kappa, alpha, beta = update_no_attack_v5(
                    event, run_length, cp, accumulator, mu, kappa, alpha, beta)
            attack_probs = calculate_prior_arr(event, alpha_arr, beta_arr, mu_arr, kappa_arr)
            attack_probs *= probabilities
            val_prob = attack_probs.sum()
            is_attack = val_prob <= 0.05
            if is_attack and not shock:
                total_non_shocks.append((time[begin], time[idx + 1]))
                shock = True
                begin = idx
            elif not is_attack and shock:
                total_shocks.append((time[begin], time[idx + 1]))
                shock = False
                begin = idx
    if shock:
        total_shocks.append((time[begin], time[-1]))
    else:
        total_non_shocks.append((time[begin], time[-1]))
    return total_shocks, total_non_shocks


def get_plot_bocpd(file_path, with_progress=False):
    """ """
    print('Starting bocpd')
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    mu, kappa, alpha, beta, lamb = np.mean(data[:100]), 0.1, 0.00001, 0.00001, 100
    # mu = 0
    lamb = 100
    alpha, beta = 1, 1
    # alpha, beta = 0.1, 0.01
    shock_intervals, non_shock_intervals = get_bocpd(
        time, data, mu, kappa, alpha, beta, lamb, with_progress=with_progress)
    fig = plot_shock(time, data, shock_intervals, non_shock_intervals)

    mu, kappa, alpha, beta, lamb = np.mean(data[:100]), 0.1, 0.00001, 0.00001, 100
    # mu = 0
    alpha, beta = 1, 1
    # alpha, beta = 0.1, 0.01
    lamb = 100
    shock_intervals_w, non_shock_intervals_w = get_bocpd_windowed(
        time, data, mu, kappa, alpha, beta, lamb, window_size=100, with_progress=with_progress)
    fig_w = plot_shock(time, data, shock_intervals_w, non_shock_intervals_w)
    # return shock_intervals, non_shock_intervals, fig
    # return shock_intervals_2, non_shock_intervals_2, fig_2
    return shock_intervals_w, non_shock_intervals_w, fig_w
