import math
import numpy as np


def natural_frequency(k, m):
    """ """
    return math.sqrt(k / m)

def damped_natural_frequency(natural_freq, crit_damping_ratio):
    """ Return damped frequency of vibration."""
    return natural_freq * math.sqrt(1 - crit_damping_ratio ** 2)


# def critical_damping_ratio(c, cr):
#     """ """
#     return c / cr

# def critical_damping_ratio(c, k, m):
#     """ """
#     return c / (2*math.sqrt(k * m))


def critical_damping_ratio(c, mass, natural_freq):
    """ """
    return c / (2 * mass * natural_freq)



def free_vibration(time_vec, x_0, v_0, nat_freq):
    """ """
    res = np.empty_like(time_vec)
    res[:] = np.sqrt(np.square(nat_freq) * x_0 ** 2 + v_0 ** 2) / nat_freq
    freq = nat_freq * time_vec
    res *= np.sin(freq + np.arctan((x_0 / v_0) * nat_freq))
    return res



def force_acceleration(force, mass):
    """ Return force acceleration for force applied to mass."""
    return force / mass


def forced_vibration(time_vec, x_0, v_0, nat_freq, f_0, force_freq):
    """ """
    res = np.empty_like(time_vec)
    freq = time_vec * nat_freq
    res[:] = (v_0 / force_freq) * np.sin(freq)
    coeff = f_0 / (nat_freq ** 2 - force_freq ** 2)
    res += (x_0 - coeff) * np.cos(freq)
    f_freq = time_vec * force_freq
    res += coeff * np.cos(f_freq)
    return res


def arbitrary_input_response_arr(time_vec, tau_vec, force_vec, damped_nat_freq, nat_freq, mass, critical_damping_ratio):
    """ """
    xx, yy = np.meshgrid(time_vec, tau_vec)
    t_diff = np.zeros_like(xx)
    np.subtract(xx, yy, where=(xx >= yy), out=t_diff)
    res = np.exp(-critical_damping_ratio * nat_freq * t_diff, where=t_diff != 0.0)
    # This should multiply the force vector by tau for each time t
    np.multiply(res, np.sin(damped_nat_freq * t_diff), out=res)
    res_vector = force_vec @ res
    return res_vector * 1 / (mass * damped_nat_freq)


def arbitrary_input_response(time_vec, tau_vec, force_vec, damped_nat_freq, nat_freq, mass, critical_damping_ratio):
    """ """
    res = np.empty_like(time_vec)
    for idx, t in enumerate(time_vec):
        acc = 0.0
        tau_slice = tau_vec < t
        for tau, force in zip(tau_vec[tau_slice], force_vec[tau_slice]):
            t_diff = t - tau
            pos_t = force * math.exp(-critical_damping_ratio * nat_freq * t_diff)
            pos_t *= math.sin(damped_nat_freq * t_diff)
            acc += pos_t
        res[idx] = acc
    res *= 1 / (mass * damped_nat_freq)
    return res


