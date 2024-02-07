import operator

import matplotlib.pyplot as plt
import numpy as np

from src.fig_funcs.detection_plots import plot_shock
from src.utils.read_data import get_data


def expectation_maximization(
        safe, not_safe, unknown, mean_1, mean_2, var_1,
        var_2, pi, epochs=1, converge_threshold=1e-3):
    """ Perform expectation maximization on one unknown.

        :returns: Tuple of (attack classification, updated mean 1,
        updated mean 2, updated variance 1, updated variance 2,
         updated attack probability."""
    # This assumes these are python lists and not numpy arrays
    if isinstance(safe, np.ndarray):
        data = np.concatenate((safe, not_safe, unknown))
    else:
        data = safe + not_safe + unknown
    # todo this may need to loop till convergence
    # Variable initialization
    mu1_hat, mu2_hat = mean_1, mean_2
    sig1_hat, sig2_hat = var_1, var_2
    pi_hat = pi
    # For some number of epochs, iterate over given data until convergence
    # last_prob = np.inf
    for idx in range(epochs):
        # Expectation
        attack_prob = list(map(lambda x: posterior_prob(
                x, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat), data))
        # Maximization
        mu1_hat, mu2_hat = update_means(attack_prob, data)
        sig1_hat, sig2_hat = update_variances(attack_prob, data, mu1_hat, mu2_hat)
        pi_hat = update_attack_prob(attack_prob, data)
    is_attack = posterior_prob(data[-1], pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat) > 0.01
    return is_attack, mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat


def phi(value, mean, variance):
    """ Z-score for value."""
    return (value - mean)/np.sqrt(variance)


def posterior_prob(point, attack_prob, attack_mean, attack_var, normal_mean, normal_var):
    """ Calculate probability of each latent variable for each data point."""
    # Probability of attack * Probability of point occurring if it was an attack
    # Divided by probability of point occurring
    num = attack_prob * phi(point, attack_mean, attack_var)
    denom = num + (1 - attack_prob) * phi(point, normal_mean, normal_var)
    post = num/denom
    return post


def mean_var_1_denom(probs):
    """ Return sum of anti-probs."""
    return sum(map(lambda x: 1-x, probs))


def mean_var_2_denom(probs):
    """ Return sum of probs."""
    return sum(probs)


def update_means(probs, events):
    """ Return updated values for means.

        :param probs: List of probabilities of attack.
        :type probs: List[float]
        :param events: List of events corresponding to probs.
        :type probs: List[float]
        :rtype: (float, float)
        :returns: tuple of updated means (mean 1, mean 2)"""
    # todo fix these iterables
    mean_1_num = sum(map(lambda x, y: 1-x*y), probs, events)
    # mean_1_denom = sum(map(lambda x: 1-x, probs))
    mean_1_denom = mean_var_1_denom(probs)
    mean_2_num = sum(map(lambda x, y: x*y), probs, events)
    # mean_2_denom = sum(probs)
    mean_2_denom = mean_var_2_denom(probs)
    mean_1 = mean_1_num/mean_1_denom
    mean_2 = mean_2_num/mean_2_denom
    return mean_1, mean_2


def variance_1_helper(probs, events, mean):
    """ """
    return sum(map(lambda x, y: 1-x*(y-mean)**2, probs, events))


def variance_2_helper(probs, events, mean):
    """ """
    return sum(map(lambda x, y: x*(y-mean)**2, probs, events))


def update_variances(probs, events, mean_1, mean_2):
    """ """
    var_1_num = variance_1_helper(probs, events, mean_1)
    var_1_denom = mean_var_1_denom(probs)
    var_2_num = variance_2_helper(probs, events, mean_2)
    var_2_denom = mean_var_2_denom(probs)
    var_1 = var_1_num/var_1_denom
    var_2 = var_2_num/var_2_denom
    return var_1, var_2


def update_attack_prob(probs, data):
    """ """
    return np.mean(list(map(operator.mul, probs, data)))


def get_expectation_max(
        normal_obs, abnormal_obs, unknowns, mean_1=None, mean_2=None, var_1=None, var_2=None, pi=None,
        shock_intervals=None, non_shock_intervals=None, epochs=1):
    """ Get and return probable shock events computed by expectation maximization.

        :rtype: (List[], List[])
        :returns: (list of shocks, list of non-shocks)"""
    # Instantiating variables
    shocks = [] if shock_intervals is None else shock_intervals
    non_shocks = [] if non_shock_intervals is None else non_shock_intervals
    # get params theta = mu , mu2, sig, sig2, pi
    mean_1_p = mean_1 if mean_1 is not None else np.mean(normal_obs)
    mean_2_p = mean_2 if mean_2 is not None else np.mean(abnormal_obs)
    var_1_p = var_1 if var_1 is not None else np.var(normal_obs)
    var_2_p = var_2 if var_2 is not None else np.var(abnormal_obs)
    if pi is not None:
        pi_p = pi
    else:
        normal_size, ab_size = len(normal_obs), len(abnormal_obs)
        pi_p = ab_size/(normal_size + ab_size)
    # Begin algorithm loop
    for idx, unknown in enumerate(unknowns):
        attack, mean_1_p, mean_2_p, var_1_p, var_2_p, pi_p = expectation_maximization(
            normal_obs, abnormal_obs, unknown, mean_1_p, mean_2_p, var_1_p,
            var_2_p, pi_p, epochs=epochs)
        if attack:
            shocks.append(unknown)
        else:
            non_shocks.append(unknown)
    return shocks, non_shocks


def get_plot_expectation_maximization(file_path):
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    shock_intervals, non_shock_intervals = get_expectation_max()
    plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.show()
