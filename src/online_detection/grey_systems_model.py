import numpy as np


def x_0(obs, n):
    return obs[0:n]


def x_1(obs, n):
    values = [sum(obs[0:idx]) for idx in range(n)]
    # for idx in range(n):
    #     values[idx] = sum(obs[0:idx])
    return values


def z_1(obs, n):
    x_1s = x_1(obs, n)
    values = [(x_1s[idx - 1] + x_1s[idx]) / 2 for idx in range(1, n)]
    return values


def x_i(obs, n):
    """ """
    # vals = x_0(obs, n)
    # return vals[n - 1] - vals[0]

    # The two lines above are equivalent to the line below
    return obs[n - 1] - obs[0]


def s_i(obs, n):
    """ """
    summation = sum(lambda k: x_i(obs, k), range(0, n - 1))
    val = 0.5 * x_i(obs, n)
    return summation + val


def grey_incidence(i_obs, j_obs, n, c=3):
    """ """
    numerator = 1 + np.abs(s_i(i_obs, n)) + np.abs(s_i(j_obs, n))
    # denominator = 1 + np.abs(s_i(i_obs, n)) + np.abs(s_i(j_obs, n))
    denominator = numerator + c * np.abs(s_i(i_obs, n) - s_i(j_obs, n))
    return numerator/denominator


def is_attack(grey_incidence_degree):
    """ """
    return grey_incidence_degree <= 0.5
