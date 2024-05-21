import math

import numpy as np

from numba import jit
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


def bayesian_online_changepoint_detection_v2(data, mu, kappa, alpha, beta, lamb):
    """ """
    my_data = np.asarray(data)
    ref_p_1, ref_p_2 = 0, 0
    run_length = 1  # Iterations since last changepoint
    maxes = list()
    attacks = list()
    prev_probabilities = list()
    probabilities = np.asarray([1])
    accumulator = 0
    for idx, event in enumerate(tqdm(my_data)):
        new_probs = calculate_probabilities(
            idx, event, alpha, beta, mu, kappa, probabilities, lamb)
        prev_probabilities.append(probabilities)
        probabilities = new_probs
        max_idx = find_max_cp(probabilities)
        maxes.append(max_idx)
        if max_idx > 0:
            print(max_idx)
        if maxes[idx] < maxes[idx - 1]:
            # event is an attack
            # run_length, ref_p_1 = update_attack(idx)
            run_length, accumulator = update_attack_v4(event)
        else:
            # update
            # cp = probabilities[0]
            # run_length, ref_p_2, mu, kappa, alpha, beta = update_no_attack(
            #     my_data, idx, run_length, probabilities[0], ref_p_1, ref_p_2, mu, kappa, alpha, beta)
            run_length, accumulator, mu, kappa, alpha, beta = update_no_attack_v4(
                my_data, idx, run_length, probabilities[0], accumulator, mu, kappa, alpha, beta)
        attacks.append(calculate_prior(event, alpha, beta, mu, kappa) < 0.1)
    return attacks


def bayesian_online_changepoint_detection_v4(data, mu, kappa, alpha, beta, lamb):
    """ """
    my_data = np.asarray(data)
    run_length = 1  # Iterations since last changepoint
    # max_val, max_idx = None, None
    max_val = 1
    head, tail = 1, 0
    attacks = list()
    accumulator = 0
    for idx, event in enumerate(tqdm(my_data)):
        prior = calculate_prior(event, alpha, beta, mu, kappa)
        hazard = hazard_function(lamb)
        non_cp = [head * prior * (1 - hazard), tail * prior * (1 - hazard)]
        cp = (head * prior * hazard) + (tail * prior * hazard)
        head, tail = cp, sum(non_cp)
        total = head + tail
        if total != 0:
            head, tail = head / total, tail / total
        # new_probs = calculate_probabilities(
        #     idx, event, alpha, beta, mu, kappa, probabilities, lamb)
        # probabilities = new_probs
        max_val = max_val * prior * (1 - hazard)
        if cp > max_val:  # event is an attack
            # max_idx = 0
            max_val = cp
            run_length, accumulator = update_attack_v4(event)
        else:
            # max_idx += 1
            # update
            run_length, accumulator, mu, kappa, alpha, beta = update_no_attack_v4(
                my_data, idx, run_length, cp, accumulator, mu, kappa, alpha, beta)

        attacks.append(calculate_prior(event, alpha, beta, mu, kappa) < 0.1)
    return attacks


@jit
def calculate_probabilities_inplace(idx, event, alpha, beta, mu, kappa, probabilities, lamb):
    """ """
    prior = calculate_prior(event, alpha, beta, mu, kappa)
    hazard = hazard_function(lamb)
    calculate_probs_helper_3(probabilities, prior, hazard, idx)
    # Add 1 because the size of the array grew
    normalize_probs_inplace(probabilities, idx + 1)
    return 0


@jit
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


@jit
def calculate_probs_helper(probs, prior, hazard, arr):
    """ Calculate the new probabilities in-place"""
    arr[0] = np.sum(probs) * hazard
    arr[1:] = probs * (1 - hazard)
    arr *= prior


@jit
def calculate_probs_helper_2(probs, prior, hazard):
    """ Calculate the new probabilities in-place"""
    arr = np.empty(shape=(len(probs) + 1))
    arr[1:] = probs * prior * (1 - hazard)
    arr[0] = np.sum(probs * prior * hazard)
    return arr


@jit
def calculate_normalize_probs_helper(probs, prior, hazard):
    """ Calculate the new probabilities in-place"""
    arr = np.zeros(shape=(len(probs) + 1))
    arr[1:] = probs * prior * (1 - hazard)
    arr[0] = np.sum(probs * prior * hazard)
    total = np.sum(arr)
    if total != 0:
        return arr * (1 / total)
    return arr


@jit
def calculate_probs_helper_3(probs, prior, hazard, idx):
    """ Calculate the new probabilities in-place."""
    total = np.sum(probs[0:idx])
    probs[1:idx + 1] = probs[0:idx] * (1 - hazard)
    probs[0] = total * hazard
    probs[0:idx + 1] *= prior


@jit
def update_attack(idx):
    """ """
    # event is an attack
    ref_p_1_p = idx
    run_length_p = 1
    return run_length_p, ref_p_1_p


def update_attack_v4(event):
    """ """
    # event is an attack
    accum = event
    run_length_p = 1
    return run_length_p, accum


@jit
def update_no_attack(
        data, idx, run_length, cp, ref_p_1, ref_p_2,
        mu, kappa, alpha, beta):
    """ """
    # update
    event = data[idx]
    run_length_p = run_length + 1
    ref_p_2_p = idx + 1
    kappa_p = kappa + 1
    alpha_p = alpha + 0.5
    x_bar = (1 - cp) * np.mean(data[ref_p_1:ref_p_2_p]) + event * cp
    mu_p = (kappa * mu + x_bar) / kappa_p  # (kappa * mu + x_bar) / (kappa + 1)
    beta_p = beta + kappa * np.square(event - x_bar) / (
                2 * kappa_p)  # beta + kappa * (event - x_bar) ** 2 / (2 * (kappa + 1))
    return run_length_p, ref_p_2_p, mu_p, kappa_p, alpha_p, beta_p


@jit
def update_no_attack_v4(
        data, idx, run_length, cp, accumulator,
        mu, kappa, alpha, beta):
    """ """
    # update
    run_length_p = run_length + 1
    kappa_p = kappa + 1
    alpha_p = alpha + 0.5
    event = data[idx]
    new_accumulator = event + accumulator
    x_bar = (1 - cp) * (new_accumulator / run_length_p) + event * cp
    mu_p = (kappa * mu + x_bar) / kappa_p
    # mu_p = cp * event + ((x_bar + run_length * mu) * (1 - cp))/(run_length_p)# (kappa * mu + x_bar) / kappa_p  # (kappa * mu + x_bar) / (kappa + 1)
    beta_p = beta + kappa * np.square(event - x_bar) / (
                2 * kappa_p)  # beta + kappa * (event - x_bar) ** 2 / (2 * (kappa + 1))
    return run_length_p, new_accumulator, mu_p, kappa_p, alpha_p, beta_p


@jit
def update_no_attack_v5(
        event, run_length, cp, accumulator,
        mu, kappa, alpha, beta):
    """ """
    # update
    run_length_p = run_length + 1
    kappa_p = kappa + 1
    alpha_p = alpha + 0.5
    new_accumulator = event + accumulator
    x_bar = (1 - cp) * (new_accumulator / run_length_p) + event * cp
    mu_p = (kappa * mu + x_bar) / kappa_p
    # mu_p = cp * event + ((x_bar + run_length * mu) * (1 - cp))/(run_length_p)# (kappa * mu + x_bar) / kappa_p  # (kappa * mu + x_bar) / (kappa + 1)
    beta_p = beta + kappa * np.square(event - x_bar) / (
                2 * kappa_p)  # beta + kappa * (event - x_bar) ** 2 / (2 * (kappa + 1))
    return run_length_p, new_accumulator, mu_p, kappa_p, alpha_p, beta_p


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
    # n is 2 * alpha is df
    # s = beta * (kappa + 1) / (alpha * kappa) is a

    return (1 / s) * t_pdf((x_bar - mu) / s, n)
    # s_root = math.sqrt(s)
    # return t_pdf((x_bar - mu) / s_root, n) / s_root
    # return (x_bar - mu) / (s / np.sqrt(n))


@jit
def t_pdf(t_value, df):
    """ """
    coeff = 1.0 / (np.sqrt(df) * beta_func(0.5, df / 2))
    exponential = (1.0 + t_value ** 2 / df) ** (-(df + 1) / 2)
    return coeff * exponential


@jit
def beta_func(a, b):
    return np.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


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
            attack = False
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
    probabilities = np.abs(np.asarray([1.0]))
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
                attack = False
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

    # mu, kappa, alpha, beta, lamb = np.mean(data[:100]), 0.1, 0.00001, 0.00001, 100
    # mu = 0
    # print(mu)
    # alpha, beta = 0.1, 0.01  # alpha, beta = 1, 1

    # shock_intervals_2, non_shock_intervals_2 = get_bocpd_2(
    #     time, data, mu, kappa, alpha, beta, lamb, with_progress=with_progress)
    # fig_2 = plot_shock(time, data, shock_intervals_2, non_shock_intervals_2)

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
