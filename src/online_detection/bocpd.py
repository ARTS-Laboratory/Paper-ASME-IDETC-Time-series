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
from utils.read_data import get_data


def bayesian_online_changepoint_detection(data, mu, kappa, alpha, beta, lamb):
    """ """
    my_data = np.asarray(data)
    ref_p_1, ref_p_2 = 0, 0
    run_length = 1  # Iterations since last changepoint
    maxes = list()
    prev_probabilities = list()
    probabilities = np.asarray([1])
    # x_bars = list()
    attacks = list()
    for idx, event in enumerate(my_data):
        prior = calculate_prior(event, alpha, beta, mu, kappa)
        hazard = hazard_function(lamb)
        non_cp = calculate_non_cp(probabilities[0:idx], prior, hazard)
        cp = calculate_cp(probabilities[0:idx], prior, hazard)
        new_probabilities = np.array([cp, *non_cp])
        prev_probabilities.append(probabilities)
        probabilities = new_probabilities
        probabilities = normalize_probs(probabilities)
        max_idx = find_max_cp(probabilities)
        maxes.append(max_idx)
        if maxes[idx] < maxes[idx - 1]:
            # event is an attack
            # num += 1
            # chngpnt.append(idx)
            ref_p_1 = idx
            # x_bars.append(event)
            run_length = 1
        else:
            # update
            run_length += 1  # rl.append(rl[idx - 1] + 1)
            ref_p_2 = idx + 1
            x_bar = (1 - cp) * np.mean(my_data[ref_p_1:ref_p_2]) + event * cp
            # sum_3 += (event - mu)**2
            # x_bars.append(x_bar)
            mu = (kappa * mu + x_bar) / (kappa + 1)
            kappa += 1
            alpha += 0.5
            beta += kappa * (event - x_bar) ** 2 / (2 * (kappa + 1))
        attacks.append(calculate_prior(event, alpha, beta, mu, kappa) < 0.1)
    return attacks


# def bayesian_online_changepoint_detection_v2(data, mu, kappa, alpha, beta, lamb):
#     """ """
#     my_data = np.asarray(data)
#     ref_p_1, ref_p_2 = 0, 0
#     run_length = 1  # Iterations since last changepoint
#     maxes = list()
#     attacks = list()
#     prev_probabilities = list()
#     probabilities = np.asarray([1])
#     accumulator = 0
#     for idx, event in enumerate(tqdm(my_data)):
#         new_probs = calculate_probabilities(
#             idx, event, alpha, beta, mu, kappa, probabilities, lamb)
#         prev_probabilities.append(probabilities)
#         probabilities = new_probs
#         max_idx = find_max_cp(probabilities)
#         maxes.append(max_idx)
#         if max_idx > 0:
#             print(max_idx)
#         if maxes[idx] < maxes[idx - 1]:
#             # event is an attack
#             # run_length, ref_p_1 = update_attack(idx)
#             run_length, accumulator = update_attack_v4(event)
#         else:
#             # update
#             # cp = probabilities[0]
#             # run_length, ref_p_2, mu_hat, kappa_hat, alpha_hat, beta_hat = update_no_attack(
#             #     my_data, idx, run_length, probabilities[0], ref_p_1, ref_p_2, mu_hat, kappa_hat, alpha_hat, beta_hat)
#             run_length, accumulator, mu, kappa, alpha, beta = update_no_attack_v4(
#                 my_data, idx, run_length, probabilities[0], accumulator, mu, kappa, alpha, beta)
#         attacks.append(calculate_prior(event, alpha, beta, mu, kappa) < 0.1)
#     return attacks
#
#
# def bayesian_online_changepoint_detection_v4(data, mu, kappa, alpha, beta, lamb):
#     """ """
#     my_data = np.asarray(data)
#     run_length = 1  # Iterations since last changepoint
#     # max_val, max_idx = None, None
#     max_val = 1
#     head, tail = 1, 0
#     attacks = list()
#     accumulator = 0
#     for idx, event in enumerate(tqdm(my_data)):
#         prior = calculate_prior(event, alpha, beta, mu, kappa)
#         hazard = hazard_function(lamb)
#         non_cp = [head * prior * (1 - hazard), tail * prior * (1 - hazard)]
#         cp = (head * prior * hazard) + (tail * prior * hazard)
#         head, tail = cp, sum(non_cp)
#         total = head + tail
#         if total != 0:
#             head, tail = head / total, tail / total
#         # new_probs = calculate_probabilities(
#         #     idx, event, alpha_hat, beta_hat, mu_hat, kappa_hat, probabilities, lamb)
#         # probabilities = new_probs
#         max_val = max_val * prior * (1 - hazard)
#         if cp > max_val:  # event is an attack
#             # max_idx = 0
#             max_val = cp
#             run_length, accumulator = update_attack_v4(event)
#         else:
#             # max_idx += 1
#             # update
#             run_length, accumulator, mu, kappa, alpha, beta = update_no_attack_v4(
#                 my_data, idx, run_length, cp, accumulator, mu, kappa, alpha, beta)
#
#         attacks.append(calculate_prior(event, alpha, beta, mu, kappa) < 0.1)
#     return attacks


# @jit
# def bayesian_online_changepoint_detection_generator(
#         data, mu, kappa, alpha, beta, lamb, filter_threshold=1e-16,
#         attack_threshold=0.1):
#     """ """
#     maxes = [0]
#     for idx, event in data:
#         probabilities = calculate_probabilities(
#             idx, event, alpha, beta, mu, kappa, probabilities, lamb)
#         max_idx = find_max_cp(probabilities)
#         maxes.append(max_idx)
#         if maxes[idx + 1] < maxes[idx]:  # if max_idx == 0:
#             # event is an attack
#             run_length, accumulator = update_attack_v4(event)
#             # cps += 1
#             attack = True
#         else:
#             # update
#             cp = probabilities[0]
#             run_length, accumulator, mu, kappa, alpha, beta = update_no_attack_v5(
#                 event, run_length, cp, accumulator, mu, kappa, alpha, beta)
#         attack_prob = calculate_prior(event, alpha, beta, mu, kappa) > 0.001  # < 0.1
#         yield attack_prob


def bayesian_online_changepoint_detection_v5(data, mu, kappa, alpha, beta, lamb):
    """ """
    print('New bocpd alg')
    my_data = np.asarray(data)
    run_length = 1  # Iterations since last changepoint
    # todo make a queue with only two elements allowed
    maxes = [0]
    probabilities = np.asarray([1.0])
    accumulator = 0.0
    cps, cps_2 = 0, 0
    alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
    for idx, event in enumerate(my_data):
        probabilities = calculate_probabilities_v2(event, alpha_arr, beta_arr, mu_arr, kappa_arr, probabilities, lamb)
        max_idx = find_max_cp(probabilities)
        maxes.append(max_idx)
        if maxes[-1] < maxes[-2]:
            # event is an attack
            # Update count
            cps += 1
            # reset run length and accumulator
            run_length, accumulator = update_attack_v4(event)
            # reset params
            probabilities = np.asarray([1.0])
            maxes = [0]
            alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
        else:
            # update
            cp = probabilities[0]
            run_length, accumulator, alpha_arr, beta_arr, mu_arr, kappa_arr = update_no_attack_arr(
                event, run_length, cp, accumulator, alpha_arr, beta_arr, mu_arr, kappa_arr, alpha, beta, mu, kappa)
        # I think this is the probability distribution of the run length
        attack_probs = calculate_prior_arr(event, alpha_arr, beta_arr, mu_arr, kappa_arr)
        attack_prob, no_attack_prob = attack_probs[0], attack_probs[1:].sum()
        if attack_prob > no_attack_prob:
            cps_2 += 1
    print(f'Number of changepoints vs predicted: {cps} vs {cps_2}')


# def bayesian_online_changepoint_detection_v5_generator(data, mu, kappa, alpha, beta, lamb):
#     """ """
#     print('New bocpd alg')
#     my_data = np.asarray(data)
#     run_length = 1  # Iterations since last changepoint
#     is_attack = False
#     maxes = deque((0,), maxlen=2)
#     probabilities = np.asarray([1.0])
#     accumulator = 0.0
#     cps, cps_2 = 0, 0
#     alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
#     for idx, event in enumerate(my_data):
#         probabilities, alpha_arr, beta_arr, mu_arr, kappa_arr = calculate_probabilities_v2(event, alpha_arr, beta_arr, mu_arr, kappa_arr, probabilities, lamb, trunc_threshold=1e-32)
#         max_idx = find_max_cp(probabilities)
#         maxes.append(max_idx)
#         if maxes[-1] < maxes[-2]:
#             is_attack = True
#             # event is an attack
#             # Update count
#             cps += 1
#             # reset run length and accumulator
#             run_length, accumulator = update_attack_v4(event)
#             # alpha_arr, beta_arr, mu_arr, kappa_arr = np.concatenate([[alpha], alpha_arr]), np.concatenate([[beta], beta_arr]), np.concatenate([[mu], mu_arr]), np.concatenate([[kappa], kappa_arr])
#             # reset params
#             probabilities = np.asarray([probabilities.sum()])
#             # maxes = [0]
#             alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
#         else:
#             # update
#             cp = probabilities[0]
#             run_length, accumulator, alpha_arr, beta_arr, mu_arr, kappa_arr = update_no_attack_arr(
#                 event, run_length, cp, accumulator, alpha_arr, beta_arr, mu_arr, kappa_arr, alpha, beta, mu, kappa)
#         # I think this is the probability distribution of the run length
#         attack_probs = calculate_prior_arr(event, alpha_arr, beta_arr, mu_arr, kappa_arr)
#         attack_likely = attack_probs.sum() < 0.001
#         yield is_attack
#         is_attack = False
#         # if attack_likely:
#         #     cps_2 += 1
#         #     yield True
#         # else:
#         #     yield False
#         # attack_prob, no_attack_prob = attack_probs[0], attack_probs[1:].sum()
#         # if attack_prob >= 0.1 or attack_prob > no_attack_prob:
#         #     cps_2 += 1
#         #     yield True
#         # else:
#         #     yield False
#     print(f'Number of changepoints vs predicted: {cps} vs {cps_2}')


def bayesian_online_changepoint_detection_v6_generator(data, mu, kappa, alpha, beta, lamb):
    """ """
    my_data = np.asarray(data)
    run_length = 1  # Iterations since last changepoint
    maxes = deque((0,), maxlen=2)
    run_length_arr = np.asarray([0], dtype=np.uint32)
    probabilities = np.asarray([1.0])
    accumulator = 0.0
    cps = 0
    alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
    for idx, event in enumerate(my_data):
        is_attack = False
        probabilities, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr = calculate_probabilities_v3(event, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr, probabilities, lamb, trunc_threshold=1e-32)
        max_idx = find_max_cp(probabilities)
        maxes.append(run_length_arr[max_idx])
        if maxes[-1] < maxes[0]:
            is_attack = True
            # event is an attack
            # Update count
            cps += 1
            # reset run length and accumulator
            run_length, accumulator = update_attack_v4(event)
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
        # attack_probs = calculate_prior_arr(event, alpha_arr, beta_arr, mu_arr, kappa_arr)
        # attack_likely = attack_probs.sum() < 0.001
        # is_attack = attack_probs[0] == np.max(attack_probs)
        yield is_attack
    print(f'Number of changepoints: {cps}')


def bayesian_online_changepoint_detection_deque_generator(data, mu, kappa, alpha, beta, lamb):
    """ """
    initial_params = NormalInverseGamma(
        alpha=alpha, beta=beta, mu=mu, kappa=kappa)
    my_data = np.asarray(data)
    run_length = 1  # Iterations since last changepoint
    maxes = deque((0,), maxlen=2)
    run_lengths = deque((0,))
    parameters = deque((initial_params,))
    probabilities = deque((1.0,))
    accumulator = 0.0
    cps = 0
    for event in my_data:
        is_attack = False
        calculate_probabilities_deque(event, lamb, parameters, run_lengths, probabilities)
        # find and append max probability
        max_idx, max_val = 0, probabilities[0]
        for idx, val in zip(run_lengths, probabilities):
            if val >= max_val:
                max_idx = idx
                max_val = val
        maxes.append(max_idx)
        if maxes[-1] < maxes[0]:
            # event is an attack
            is_attack = True
            run_length, accumulator = update_attack_v4(event)
            # reset params
            parameters.clear()
            parameters.append(NormalInverseGamma(
                alpha=alpha, beta=beta, mu=mu, kappa=kappa))
            probabilities.clear()
            probabilities.append(1.0)
            run_lengths.clear()
            run_lengths.append(0)
        else:
            run_length, accumulator = update_no_attack_deque(
                event, run_length, accumulator, parameters, alpha, beta, mu, kappa)
        yield is_attack


# @jit
# def calculate_probabilities_inplace(idx, event, alpha, beta, mu, kappa, probabilities, lamb):
#     """ """
#     prior = calculate_prior(event, alpha, beta, mu, kappa)
#     hazard = hazard_function(lamb)
#     calculate_probs_helper_3(probabilities, prior, hazard, idx)
#     # Add 1 because the size of the array grew
#     normalize_probs_inplace(probabilities, idx + 1)
#     return 0


@njit
def calculate_probabilities(idx, event, alpha, beta, mu, kappa, probabilities, lamb):
    """ """
    threshold = 1e-16
    prior = calculate_prior(event, alpha, beta, mu, kappa)
    new_probabilities = np.empty(len(probabilities) + 1)
    calculate_probs_helper(
        probabilities, prior, hazard_function(lamb),
        new_probabilities)
    threshold_filter = new_probabilities > threshold
    threshold_filter[0] = True
    new_probabilities = new_probabilities[threshold_filter]
    normalize_probs_2(new_probabilities)
    return new_probabilities


@njit
def calculate_probabilities_v2(event, alpha, beta, mu, kappa, probabilities, lamb, trunc_threshold=1e-16):
    """ """
    hazard = hazard_function(lamb)
    threshold = trunc_threshold
    priors = calculate_prior_arr(event, alpha, beta, mu, kappa)
    new_probabilities = np.empty(probabilities.size + 1)
    # Multiply probabilities by their priors
    priors *= probabilities
    new_probabilities[0] = priors.sum() * hazard
    new_probabilities[1:] = priors * (1 - hazard)
    # Normalize probabilities
    prob_sum = new_probabilities.sum()
    if prob_sum != 0.0:
        new_probabilities /= prob_sum
        # new_probabilities /= new_probabilities.sum()
    # Truncate near zero values
    trunc = new_probabilities < trunc_threshold
    new_probabilities[trunc] = 0.0
    # threshold_filter = new_probabilities > threshold
    # threshold_filter[0] = True
    # new_probabilities = new_probabilities[threshold_filter]
    # threshold_filter = threshold_filter[1:]
    # new_alpha, new_beta, new_mu, new_kappa = alpha[threshold_filter], beta[threshold_filter], mu[threshold_filter], kappa[threshold_filter]
    new_alpha, new_beta, new_mu, new_kappa = alpha, beta, mu, kappa
    return new_probabilities, new_alpha, new_beta, new_mu, new_kappa


@profile
# @njit
def calculate_probabilities_v3(event, alpha, beta, mu, kappa, run_lengths, probabilities, lamb, trunc_threshold=1e-16):
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


@profile
def calculate_probabilities_deque(event: float, lamb: float, params: deque, run_lengths: deque, probabilities: deque, trunc_threshold=1e-16):
    """ """
    size_switch = 1_000
    hazard = hazard_function(lamb)
    arr_size = len(params)
    if arr_size < size_switch:
        # old option
        # priors = calculate_prior_deque_list(event, params)
        # new option
        priors = calculate_prior_deque_list(event, {dataclasses.astuple(param) for param in params})
    else:
        priors = np.empty(arr_size)
        calculate_prior_deque_ndarray(event, params, priors)
    if isinstance(priors, list):
        head = sum(priors) * hazard
        neg_prob = 1 - hazard
        tail = [prior * neg_prob for prior in priors]
    elif isinstance(priors, np.ndarray):
        head = priors.sum() * hazard
        priors *= (1 - hazard)
        tail = priors
    # Update probabilities
    for idx in range(arr_size):
        probabilities[0] = tail[idx]
        probabilities.rotate(1)
        run_lengths[0] += 1
        run_lengths.rotate(1)
    probabilities.appendleft(head)
    run_lengths.appendleft(0)
    # Normalize vector
    prob_sum = sum(probabilities)
    if prob_sum != 0.0:
        for idx in range(arr_size):
            probabilities[0] /= prob_sum
            probabilities.rotate(1)
    # Truncate values near zero
    probabilities.rotate(1)
    for idx in range(arr_size):
        if probabilities[0] < trunc_threshold:
            probabilities.popleft()
            params.popleft()
            run_lengths.popleft()
        else:
            probabilities.rotate(1)
            params.rotate(1)
            run_lengths.rotate(1)



@jit
def calculate_probs_helper(probs, prior, hazard, arr):
    """ Calculate the new probabilities in-place"""
    arr[0] = np.sum(probs) * hazard
    arr[1:] = probs * (1 - hazard)
    arr *= prior


# @jit
# def calculate_probs_helper_2(probs, prior, hazard):
#     """ Calculate the new probabilities in-place"""
#     arr = np.empty(shape=(len(probs) + 1))
#     arr[1:] = probs * prior * (1 - hazard)
#     arr[0] = np.sum(probs * prior * hazard)
#     return arr


# @jit
# def calculate_normalize_probs_helper(probs, prior, hazard):
#     """ Calculate the new probabilities in-place"""
#     arr = np.zeros(shape=(len(probs) + 1))
#     arr[1:] = probs * prior * (1 - hazard)
#     arr[0] = np.sum(probs * prior * hazard)
#     total = np.sum(arr)
#     if total != 0:
#         return arr * (1 / total)
#     return arr


# @jit
# def calculate_probs_helper_3(probs, prior, hazard, idx):
#     """ Calculate the new probabilities in-place."""
#     total = np.sum(probs[0:idx])
#     probs[1:idx + 1] = probs[0:idx] * (1 - hazard)
#     probs[0] = total * hazard
#     probs[0:idx + 1] *= prior


# @jit
# def update_attack(idx):
#     """ """
#     # event is an attack
#     ref_p_1_p = idx
#     run_length_p = 1
#     return run_length_p, ref_p_1_p


def update_attack_v4(event):
    """ """
    # event is an attack
    accum = event
    run_length_p = 1
    return run_length_p, accum


# @jit
# def update_no_attack(
#         data, idx, run_length, cp, ref_p_1, ref_p_2,
#         mu, kappa, alpha, beta):
#     """ """
#     # update
#     event = data[idx]
#     run_length_p = run_length + 1
#     ref_p_2_p = idx + 1
#     kappa_p = kappa + 1
#     alpha_p = alpha + 0.5
#     x_bar = (1 - cp) * np.mean(data[ref_p_1:ref_p_2_p]) + event * cp
#     mu_p = (kappa * mu + x_bar) / kappa_p  # (kappa_hat * mu_hat + x_bar) / (kappa_hat + 1)
#     beta_p = beta + kappa * np.square(event - x_bar) / (
#                 2 * kappa_p)  # beta_hat + kappa_hat * (event - x_bar) ** 2 / (2 * (kappa_hat + 1))
#     return run_length_p, ref_p_2_p, mu_p, kappa_p, alpha_p, beta_p


# @jit
# def update_no_attack_v4(
#         data, idx, run_length, cp, accumulator,
#         mu, kappa, alpha, beta):
#     """ """
#     # update
#     run_length_p = run_length + 1
#     kappa_p = kappa + 1
#     alpha_p = alpha + 0.5
#     event = data[idx]
#     new_accumulator = event + accumulator
#     x_bar = (1 - cp) * (new_accumulator / run_length_p) + event * cp
#     mu_p = (kappa * mu + x_bar) / kappa_p
#     # mu_p = cp * event + ((x_bar + run_length * mu_hat) * (1 - cp))/(run_length_p)# (kappa_hat * mu_hat + x_bar) / kappa_p  # (kappa_hat * mu_hat + x_bar) / (kappa_hat + 1)
#     beta_p = beta + kappa * np.square(event - x_bar) / (
#                 2 * kappa_p)  # beta_hat + kappa_hat * (event - x_bar) ** 2 / (2 * (kappa_hat + 1))
#     return run_length_p, new_accumulator, mu_p, kappa_p, alpha_p, beta_p


@njit
def update_no_attack_v5(
        event, run_length, cp, accumulator,
        mu, kappa, alpha, beta):
    """ """
    # update
    kappa_plus_one = kappa + 1
    run_length_p = run_length + 1
    kappa_p = kappa_plus_one
    alpha_p = alpha + 0.5
    new_accumulator = event + accumulator
    mu_p = (kappa * mu + event) / kappa_plus_one
    beta_p = beta + kappa * np.square(event - mu) / (2 * kappa_plus_one)

    # run_length_p = run_length + 1
    # kappa_p = kappa + 1
    # alpha_p = alpha + 0.5
    # new_accumulator = event + accumulator
    # x_bar = (1 - cp) * (new_accumulator / run_length_p) + event * cp
    # mu_p = (kappa * mu + x_bar) / kappa_p
    # # mu_p = cp * event + ((x_bar + run_length * mu_hat) * (1 - cp))/(run_length_p)# (kappa_hat * mu_hat + x_bar) / kappa_p  # (kappa_hat * mu_hat + x_bar) / (kappa_hat + 1)
    # beta_p = beta + kappa * np.square(event - mu) / (2 * kappa_p)
    # beta_hat + kappa_hat * (event - x_bar) ** 2 / (2 * (kappa_hat + 1))
    # beta_p = beta_hat + kappa_hat * np.square(event - x_bar) / ( 2 * kappa_p)
    return run_length_p, new_accumulator, mu_p, kappa_p, alpha_p, beta_p


@njit
def update_no_attack_arr(event, run_length, cp, accumulator, alpha_arr, beta_arr, mu_arr, kappa_arr, alpha, beta, mu, kappa):
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
    # run_length_p = run_length + 1
    # kappa_p = kappa_arr + 1
    # alpha_p = alpha_arr + 0.5
    # new_accumulator = event + accumulator
    # x_bar = (1 - cp) * (new_accumulator / run_length_p) + event * cp
    # # mu_p = (kappa_arr * mu_arr + x_bar) / kappa_p
    # mu_p = cp * event + ((x_bar + run_length * mu_arr) * (1 - cp))/(run_length_p)# (kappa_hat * mu_hat + x_bar) / kappa_p  # (kappa_hat * mu_hat + x_bar) / (kappa_hat + 1)
    # beta_p = beta_arr + kappa_arr * np.square(event - mu_arr) / (2 * kappa_p)
    # # beta_hat + kappa_hat * (event - x_bar) ** 2 / (2 * (kappa_hat + 1))
    # # beta_p = beta_hat + kappa_hat * np.square(event - x_bar) / ( 2 * kappa_p)
    mu_p[0] = mu
    kappa_p[0] = kappa
    alpha_p[0] = alpha
    beta_p[0] = beta
    # mu_p = np.concatenate(((mu,), mu_p))
    # kappa_p = np.concatenate(((kappa,), kappa_p))
    # alpha_p = np.concatenate(((alpha,), alpha_p))
    # beta_p = np.concatenate(((beta,), beta_p))
    return run_length_p, new_accumulator, alpha_p, beta_p, mu_p, kappa_p


@profile
def update_no_attack_deque(event: float, run_length: int, accumulator: float,
                           params: deque, alpha: float, beta: float, mu: float, kappa: float):
    """ Update if no attack was detected.


    """
    new_accumulator = event + accumulator
    new_run_length = run_length + 1
    for param in params:
        kappa_plus = param.kappa + 1
        param.beta += param.kappa * np.square(event - param.mu) / (2 * kappa_plus)
        param.mu = (param.kappa * param.mu + event) / kappa_plus
        param.alpha += 0.5
        param.kappa += 1.0
    params.appendleft(NormalInverseGamma(
        alpha=alpha, beta=beta, mu=mu, kappa=kappa))
    return new_run_length, new_accumulator




@njit
def calculate_prior_arr(point, alphas, betas, mus, kappas):
    """ Return student's T distribution PDF given parameters of inverse gamma distribution."""
    return t_func_arr(point, mus, ((betas * (kappas + 1.0)) / (alphas * kappas)), 2 * alphas)


@profile
# @njit
def calculate_prior_arr_v1(point, alphas, betas, mus, kappas):
    """ Return student's T distribution PDF for given parameters of inverse gamma distribution."""
    t_values = calculate_prior_helper(point, alphas, betas, mus, kappas)
    t_values /= scipy.special.beta(0.5, alphas)
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
    # # old code
    denom = 2 * betas * (kappas + 1.0) / kappas
    t_values = (point - mus)**2 / denom
    t_values += 1.0
    # t_values **= -(alphas + 0.5)
    exponent = -(alphas + 0.5)
    t_values **= exponent
    t_values /= np.sqrt(denom)
    return t_values


# def calculate_prior_deque(point, params, out):
#     if isinstance(out, list):
#         calculate_prior_deque_list(point, tuple(params), out)
#     elif isinstance(out, np.ndarray):
#         calculate_prior_deque_ndarray(point, params, out)
#     else:
#         raise NotImplementedError()


# @njit
def calculate_prior_deque_list(point, params):
        # alpha, beta, mu, kappa is the order of items in param
    # return [calculate_prior_helper(point, param.alpha, param.beta, param.mu,
    #                                       param.kappa) / scipy.special.beta(0.5, param.alpha) for param in params]
    return [calculate_prior_helper(point, param[0], param[1], param[2],
                                          param[3]) / scipy.special.beta(0.5, param[1]) for param in params]


# @njit
def calculate_prior_deque_ndarray(point, params, out):
    for idx, param in enumerate(params):
        out[idx] = calculate_prior_helper(point, param.alpha, param.beta, param.mu, param.kappa) / scipy.special.beta(0.5, param.alpha)


@overload(scipy.special.beta)
def scipy_beta(a, b):
    def scipy_beta_fn(a, b):
        return scipy.special.beta(a, b)
    return scipy_beta_fn

@jit
def calculate_prior(point, alpha, beta, mu, kappa):
    """ """
    return t_func(point, mu, ((beta * (kappa + 1)) / (alpha * kappa)), 2 * alpha)


@jit
def calculate_non_cp(probs, prior, h):
    """ """
    # H is a function or a constant
    return np.asarray(probs) * prior * (1 - h)


@jit
def calculate_cp(probs, prior, h):
    return np.sum(np.asarray(probs) * prior * h)


@jit
def normalize_probs(probs):
    total = np.sum(probs)
    if total == 0:
        return probs
    return np.asarray(probs) / total  # np.asarray(probs) * (1 / total)


@jit
def normalize_probs_2(probs):
    total = np.sum(probs)
    if total == 0:
        return
    probs /= total  # np.asarray(probs) * (1 / total)


@jit
def normalize_probs_inplace(probs, idx):
    total = np.sum(probs[0:idx])
    if total != 0:
        probs[0:idx] /= total


@jit
def find_max_cp(probs):
    return np.argmax(probs)


@jit
def hazard_function(lam):
    return 1 / lam


@jit
def t_func(x_bar, mu, s, n):
    """

    s(a in function) normalizing value: beta * (kappa + 1) / (alpha * kappa)
    n(degrees of freedom): 2 * alpha
    """
    # n is 2 * alpha_hat is df
    # s = beta_hat * (kappa_hat + 1) / (alpha_hat * kappa_hat) is a

    # return (1 / s) * t_pdf((x_bar - mu) / s, n)
    s_root = math.sqrt(s)
    return t_pdf((x_bar - mu) / s_root, n) / s_root
    # return (x_bar - mu_hat) / (s / np.sqrt(n))


@njit
def t_func_arr(x_bar, mu_arr, s_arr, n_arr):
    """ """
    # t_values = np.zeros_like(mu_arr)
    s_n_arr = s_arr * n_arr
    n_half = n_arr * 0.5
    t_values = ((x_bar - mu_arr)**2 / s_n_arr + 1.0) ** (-(n_half + 0.5))
    # t_values = np.square(x_bar - mu_arr)
    # t_values /= s_n_arr
    # t_values += 1.0
    # t_values **= (-(n_arr + 1) / 2)

    # t_values = ((x_bar - mu_arr)**2 / s_n_arr + 1.0) ** (-(n_arr + 1) / 2)

    # t_values += (np.square(x_bar - mu_arr) / (s_arr * n_arr) + 1.0) ** (-(n_arr + 1) / 2)

    # t_values /= (np.sqrt(s_n_arr))
    # # out = np.empty_like(t_values)
    # out = t_func_arr_helper_beta(n_half)
    # t_values *= out

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


@njit
def t_pdf(t_value, df):
    """ """
    exponential = (1.0 + t_value ** 2 / df) ** (-(df + 1) / 2)
    exponential /= (np.sqrt(df) * beta_func(0.5, df / 2))
    # coeff = 1.0 / (np.sqrt(df) * beta_func(0.5, df / 2))
    return exponential
    # return coeff * exponential


@vectorize(['float64(float64, float64)', 'float32(float32, float32)'], cache=False, nopython=True)
def beta_numba(val_1, val_2):
    """ Return vectorized function for """
    return math.exp(math.lgamma(val_1) + math.lgamma(val_2) - math.lgamma(val_1 + val_2))


@guvectorize(
    ['void(float64[:], float64[:])', 'void(float32[:], float32[:])'],
    '(n)->(n)', cache=True, nopython=True)
def t_func_arr_helper_beta(arr, res):
    """ Return beta function for vector"""
    for idx, item in enumerate(arr):
        res[idx] = math.exp(math.lgamma(item) - math.lgamma(0.5 + item))
    res *= np.sqrt(np.pi)
    # old
    # gamma_1 = math.lgamma(0.5)
    # for idx, item in enumerate(arr / 2):
    #     res[idx] = np.exp(gamma_1 + math.lgamma(item) - math.lgamma(0.5 + item))
#
#
# @guvectorize(
#     ['(float32, float32[:], float32[:])', '(float32, float64[:], float64[:])', '(float64, float64[:], float64[:])'],
#     '(), (n) -> (n)', cache=True)
# def beta_numba_arr(arr_1, arr_2, res):
#     """ Return beta function for vector"""
#     gamma_1 = math.lgamma(arr_1)
#     for idx, item in enumerate(arr_2):
#         res[idx] = math.exp(gamma_1 + math.lgamma(item) - math.lgamma(arr_1 + item))


@njit
def beta_func(a, b):
    return np.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def get_bocpd_v5_from_generator(time, data, mu, kappa, alpha, beta, lamb,
        shock_intervals=None, non_shock_intervals=None, with_progress=False):
    """ """
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    my_data = data  # np.abs(data)
    begin = 0
    shock = False
    attacks, not_attacks = 0, 0
    # This is a generator
    bocpd_model_gen = bayesian_online_changepoint_detection_v6_generator(my_data, mu, kappa, alpha, beta, lamb)
    # bocpd_model_gen = bayesian_online_changepoint_detection_deque_generator(my_data, mu, kappa, alpha, beta, lamb)
    items = tqdm(bocpd_model_gen, total=len(my_data)) if with_progress else bocpd_model_gen
    for idx, is_attack in enumerate(items):
        if is_attack:  # if there was a change, we flip state and mark last segment
            attacks += 1
            if shock:  # if shock, mark end of shock state
                shocks.append((time[begin], time[idx]))
            else:
                non_shocks.append((time[begin], time[idx]))
            begin = idx
            shock = not shock
        else:
            not_attacks += 1
        # if is_attack and not shock:
        #     non_shocks.append((time[begin], time[idx]))
        #     shock = True
        #     begin = idx
        #     attacks += 1
        # elif not is_attack and shock:
        #     shocks.append((time[begin], time[idx]))
        #     shock = False
        #     begin = idx
        #     not_attacks += 1
    print(f'Safe points: {not_attacks}, change points: {attacks}')
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
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
    maxes = [0]
    probabilities = np.asarray([1])
    my_data = np.abs(np.asarray(data))
    mu = np.mean(my_data[:100])
    items = tqdm(enumerate(my_data), total=len(my_data)) if with_progress else enumerate(my_data)
    accumulator = 0.0
    attack = False
    for idx, event in items:
        probabilities = calculate_probabilities(
            idx, event, alpha, beta, mu, kappa, probabilities, lamb)
        max_idx = find_max_cp(probabilities)
        maxes.append(max_idx)
        if maxes[idx + 1] < maxes[idx]:  # if max_idx == 0:
            # event is an attack
            run_length, accumulator = update_attack_v4(event)
            cps += 1
            attack = True
        else:
            # update
            cp = probabilities[0]
            run_length, accumulator, mu, kappa, alpha, beta = update_no_attack_v5(
                event, run_length, cp, accumulator, mu, kappa, alpha, beta)
        attack_prob = calculate_prior(event, alpha, beta, mu, kappa) > 0.001  # < 0.1
        if attack_prob:
            cps_2 += 1
        if attack_prob:
            if shock:
                shocks.append((time[begin], time[idx]))
            else:
                non_shocks.append((time[begin], time[idx]))
            begin = idx
            shock = not shock
    print(f'Total changepoints: {cps} vs {cps_2}, max idx: {max(maxes)}')
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def get_bocpd_windowed(time, data, mu, kappa, alpha, beta, lamb,
                       window_size=1, with_progress=False):
    total_shocks, total_non_shocks = list(), list()
    itr = tqdm(range(0, len(data), window_size), total=len(data) // window_size) if with_progress else range(0,
                                                                                                             len(data),
                                                                                                             window_size)
    begin = 0
    shock = False
    cps, cps_2 = 0, 0
    run_length = 1  # Iterations since last changepoint
    maxes = [0]
    accumulator = 0
    attack = False
    probabilities = np.asarray([1.0])
    my_data = np.abs(np.asarray(data))
    mu = np.mean(my_data[:100])
    for idx in itr:
        # shocks, non_shocks = get_bocpd(
        #     time, data[idx: idx + window_size], mu, kappa, alpha, beta, lamb)
        probabilities = np.asarray([1.0])
        maxes = [0]
        accumulator = 0.0
        for jdx, event in enumerate(my_data[idx:idx + window_size], start=idx):
            # print(f'Accumulator: {accumulator}')
            # prior = calculate_prior(event, alpha, beta, mu, kappa)
            probabilities = calculate_probabilities(
                jdx, event, alpha, beta, mu, kappa, probabilities, lamb)
            max_idx = find_max_cp(probabilities)
            maxes.append(max_idx)
            # if max_idx < maxes[idx] and max_idx == 0:
            #     print('discrepancy')
            if maxes[-1] < maxes[-2]:  # if max_idx == 0:
                # event is an attack
                run_length, accumulator = update_attack_v4(event)
                cps += 1
                attack = True
            else:
                # update
                cp = probabilities[0]
                # cp = probabilities_2[0]
                run_length, accumulator, mu, kappa, alpha, beta = update_no_attack_v5(
                    event, run_length, cp, accumulator, mu, kappa, alpha, beta)
            attack_prob = calculate_prior(event, alpha, beta, mu, kappa) > 0.001  # < 0.1
            if attack_prob:
                cps_2 += 1
            if attack_prob:
                if shock:
                    total_shocks.append((time[begin], time[jdx]))
                else:
                    total_non_shocks.append((time[begin], time[jdx]))
                begin = jdx
                shock = not shock
    print(f'Total changepoints: {cps} vs {cps_2}, max idx: {max(maxes)}')
    if shock:
        total_shocks.append((time[begin], time[-1]))
    else:
        total_non_shocks.append((time[begin], time[-1]))
    # print(f'BOCPD: copy vs inplace close enough: {all(np.isclose(probabilities, probabilities_2))}')
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
