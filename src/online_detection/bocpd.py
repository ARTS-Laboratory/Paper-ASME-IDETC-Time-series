import numpy as np


def bayesian_online_changepoint_detection(data, mu, kappa, alpha, beta, lamb):
    """ """
    ref_p_1, ref_p_2 = 0, 0
    run_length = 1  # Iterations since last changepoint
    maxes = list()
    prev_probabilities = list()
    probabilities = [1]
    x_bars = list()
    for idx, event in enumerate(data):
        prior = calculate_prior(event, alpha, beta, mu, kappa)
        hazard = hazard_function(lamb)
        non_cp = calculate_non_cp(probabilities[1:idx], prior, hazard)
        cp = calculate_cp(probabilities[1:idx], prior, hazard)
        new_probabilities = [*cp, *non_cp]
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
            x_bars.append(event)
            run_length = 1
        else:
            # update
            run_length += 1  # rl.append(rl[idx - 1] + 1)
            ref_p_2 = idx
            x_bar = (1-cp) * np.mean(data[ref_p_1:ref_p_2]) + event * cp
            # sum_3 += (event - mu)**2
            x_bars.append(x_bar)
            mu = (kappa * mu + x_bar) / (kappa + 1)
            kappa += 1
            alpha += 0.5
            beta += kappa*(event - x_bar)**2 / (2 * (kappa + 1))
    return calculate_prior(data[-1], alpha, beta, mu, kappa) < 0.1


# def probability_of_run_length(run_length, ):
#     pass


def calculate_prior(point, alpha, mu, beta, kappa):
    """ """
    return t_func(point, mu, ((beta * (kappa + 1))/(alpha * kappa)), 2 * alpha)
    # return t_func(point, 2 * alpha, mu, (beta * (kappa + 1))/(alpha * kappa))


def calculate_non_cp(probs, prior, h):
    """ """
    # H is a function or a constant
    return probs * prior * (1 - h)


def calculate_cp(probs, prior, h):
    return sum(probs * prior * h)


def normalize_probs(probs):
    total = sum(probs)
    return probs / total


def find_max_cp(probs):
    return np.argmax(probs)


def hazard_function(lam):
    return 1/lam


def t_func(x_bar, mu, s, n):
    """ """
    return (x_bar - mu) / (s / np.sqrt(n))
