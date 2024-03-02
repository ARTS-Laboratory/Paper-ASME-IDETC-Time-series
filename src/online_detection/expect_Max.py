import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional, Iterable
from scipy.stats import norm
from time import perf_counter

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
    # This assumes these are python lists and not numpy arrays
    if isinstance(safe, np.ndarray):
        data = np.append(np.concatenate((safe, not_safe)), unknown)
    else:
        data = np.asarray(safe + not_safe + unknown)
    # todo this may need to loop till convergence
    # Variable initialization
    size = len(data)
    mu1_hat, mu2_hat = mean_1, mean_2
    sig1_hat, sig2_hat = var_1, var_2
    pi_hat = pi
    # For some number of epochs, iterate over given data until convergence
    # last_prob = np.inf
    for idx in range(epochs):
        # Expectation
        # TODO confirm that the following 3 lines are equivalent
        # attack_prob = posterior_probs(data, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat)
        # inverse = posterior_probs(data, 1 - pi_hat, mu1_hat, sig1_hat, mu2_hat, sig2_hat)
        attack_prob, inverse = posterior_probs_v2(data, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat)
        # Maximization
        density, inverse_density = sum(attack_prob), sum(inverse)
        # If all probabilities are zero for attack or not attack, no need to update
        if not (density == 0 or inverse_density == 0):
            new_mu1_hat, new_mu2_hat = update_means(attack_prob, inverse, density, inverse_density, data)
            new_sig1_hat, new_sig2_hat = update_variances(attack_prob, inverse, density, inverse_density, data, mu1_hat,
                                                          mu2_hat)
            new_pi_hat = update_attack_prob(density, size)
            # Update variables
            mu1_hat, mu2_hat = new_mu1_hat, new_mu2_hat
            sig1_hat, sig2_hat = new_sig1_hat, new_sig2_hat
            pi_hat = new_pi_hat
    is_attack = posterior_prob(data[-1], pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat) > 0.01
    return is_attack, mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat


def phi(value, mean, variance):
    """ Return the probability density function for value.

        :param float value: Value to get function for.
        :param float mean: Population mean.
        :param float variance: Population variance.
        :returns: PDF of value given.
        :rtype: float
    """
    return norm.pdf(value, loc=mean, scale=np.sqrt(variance))


def parameter_denom(point, attack_prob, attack_mean, attack_var, normal_mean, normal_var):
    """ """
    return ((attack_prob * phi(point, attack_mean, attack_var)) +
            ((1 - attack_prob) * phi(point, normal_mean, normal_var)))


def posterior_prob(point, attack_prob, attack_mean, attack_var, normal_mean, normal_var):
    """ Calculate probability of latent variable for given data point."""
    # Probability of attack * Probability of point occurring if it was an attack
    # Divided by probability of point occurring
    num = attack_prob * phi(point, attack_mean, attack_var)
    denom = num + (1 - attack_prob) * phi(point, normal_mean, normal_var)
    post = num / denom
    return post


def posterior_probs(points, attack_prob, attack_mean, attack_var, normal_mean, normal_var):
    """ Calculate probability of each latent variable for each data point."""
    # Probability of attack * Probability of point occurring if it was an attack
    # Divided by probability of point occurring
    return posterior_prob(
        np.asarray(points), attack_prob, attack_mean, attack_var, normal_mean, normal_var)
    # return np.asarray(
    #     [posterior_prob(point, attack_prob, attack_mean, attack_var, normal_mean, normal_var) for point in points])


def posterior_probs_v2(points, attack_prob, attack_mean, attack_var, normal_mean, normal_var):
    """ Calculate probabilities of each latent variable for each data point."""
    num_1 = attack_prob * phi(points, attack_mean, attack_var)
    num_2 = (1 - attack_prob) * phi(points, normal_mean, normal_var)
    denom = num_1 + num_2
    return num_1/denom, num_2/denom


def mean_var_1_denom(probs):
    """ Return sum of anti-probs."""
    return sum(map(lambda x: 1 - x, probs))


def mean_var_2_denom(probs):
    """ Return sum of probs."""
    return sum(probs)


def dot_prod(x, y):
    """ Return the dot product of the given vectors.

        :param Iterable x:
        :param Iterable y:
        """
    return np.dot(x, y)  # sum(map(operator.mul, x, y))


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
    # todo fix these iterables
    mean_1_denom = inverse_density
    mean_2_denom = density

    mean_1_num = dot_prod(inverse, events)
    mean_2_num = dot_prod(probs, events)
    mean_1 = mean_1_num / mean_1_denom
    mean_2 = mean_2_num / mean_2_denom
    return mean_1, mean_2


def variance_helper(probs, events, mean):
    """ """
    return dot_prod(probs, np.asarray((events - mean) ** 2))


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
    var_1_num = variance_helper(inverse, events, mean_1)
    var_2_num = variance_helper(probs, events, mean_2)
    var_1_denom = inverse_density  # Reassignment for clarity
    var_2_denom = density  # Reassignment for clarity
    var_1 = var_1_num / var_1_denom
    var_2 = var_2_num / var_2_denom
    return var_1, var_2


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
        shock_intervals=None, non_shock_intervals=None, epochs=1):
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
    for idx, unknown in enumerate(unknowns):
        start = perf_counter()
        attack, mean_1_p, mean_2_p, var_1_p, var_2_p, pi_p = expectation_maximization(
            normal_obs, abnormal_obs, np.asarray(unknown), mean_1_p, mean_2_p, var_1_p,
            var_2_p, pi_p, epochs=epochs)
        stop = perf_counter()
        elapsed += stop - start
        if idx % 10_000 == 9_999:
            print(f'Function took {elapsed / 10_000} sec to run on average over last 10,000 loops, {elapsed} sec total.')
            elapsed = 0
        if attack and not shock:  # If detected attack and not in shock state, change state
            non_shocks.append((time[begin], time[idx - 1]))
            shock = True
            begin = idx
        elif not attack and shock:
            shocks.append((time[begin], time[idx - 1]))
            shock = False
            begin = idx
    # Check if remaining segment is shock or not
    if shock:
        shocks.append((time[begin], time[-1]))
    else:
        non_shocks.append((time[begin], time[-1]))
    return shocks, non_shocks


def get_plot_expectation_maximization(file_path):
    my_data = get_data(file_path)
    time, data = my_data[:, 0], my_data[:, 1]
    print(len(my_data[:, 0]))
    shock_intervals, non_shock_intervals = get_expectation_max(time, data[:500], data[250_000:250_030],
                                                               data, epochs=5)
    plot_shock(time, data, shock_intervals, non_shock_intervals)
    plt.show()