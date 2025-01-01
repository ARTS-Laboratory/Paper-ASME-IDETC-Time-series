# distutils: language = c++
# cython: language_level=3
# cython: profile=True
# cython: linetrace=True
import numpy as np
cimport numpy as np
cimport scipy.special.cython_special as csc
import cython
from libcpp.deque cimport deque

# from online_detection.normal_inverse_gamma import NormalInverseGamma


from libc.math cimport lgamma, exp, sqrt

from online_detection.normal_inverse_gamma import NormalInverseGamma

ctypedef fused real_type:
    float
    double


cdef struct NormInverseGammaCy:
    double alpha
    double beta
    double mu
    double kappa


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void calculate_prior_arr_helper_float_cy(
        float point, const float[:] alphas, const float[:] betas,
        const float[:] mus, const float[:] kappas, float[:] out):
    # cdef double prior
    cdef Py_ssize_t idx, length
    cdef float value
    length = out.shape[0]
    # cdef float[:] out_view = out
    for idx in range(length):
        value = calculate_prior_float_cy(
            point, alphas[idx], betas[idx], mus[idx], kappas[idx])
        out[idx] = value
        # out_view[idx] = value
    return

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void calculate_prior_arr_helper_double_cy(
        double point, const double[:] alphas, const double[:] betas,
        const double[:] mus, const double[:] kappas, double[:] out):
    # cdef double prior
    cdef Py_ssize_t idx, length
    cdef double value
    length = out.shape[0]
    for idx in range(length):
        value = calculate_prior_double_cy(
            point, alphas[idx], betas[idx], mus[idx], kappas[idx])
        out[idx] = value
    return


cpdef np.ndarray calculate_prior_arr_cy(
        cython.floating point, cython.floating[:] alphas, cython.floating[:] betas,
        cython.floating[:] mus, cython.floating[:] kappas):
    """ Return student's T distribution PDF for given parameters of inverse gamma distribution."""
    out = np.empty_like(alphas)
    cdef cython.floating[:] out_view = out
    assert (alphas.shape[0] == betas.shape[0] == mus.shape[0] == kappas.shape[0])
    if cython.floating is double:
        calculate_prior_arr_helper_double_cy(point, alphas, betas, mus, kappas, out_view)
    else:
        calculate_prior_arr_helper_float_cy(point, alphas, betas, mus, kappas, out_view)
    # calculate_prior_arr_helper_cy(point, alphas, betas, mus, kappas, out)
    return out


cpdef calculate_prior_arr_inplace_cy(
        cython.floating point, cython.floating[:] alphas, cython.floating[:] betas,
        cython.floating[:] mus, cython.floating[:] kappas, cython.floating[:] out):
    """ """
    cdef cython.floating[:] out_view = out
    assert (alphas.shape[0] == betas.shape[0] == mus.shape[0] == kappas.shape[0])
    if cython.floating is double:
        calculate_prior_arr_helper_double_cy(point, alphas, betas, mus, kappas, out_view)
    else:
        calculate_prior_arr_helper_float_cy(point, alphas, betas, mus, kappas, out_view)

@cython.cdivision(True)
cdef float calculate_prior_float_cy(
        float point, float alpha, float beta, float  mu, float kappa):
    cdef double denom, t_value, exponent, out
    # if kappa == 0.0:
    #     return 1.0 if point == mu else 0.0
    denom = 2 * beta * (kappa + 1)
    denom /= kappa
    t_value = (point - mu)**2 / denom + 1
    exponent = -(alpha + 0.5)
    t_value **= exponent
    denom = sqrt(denom)
    t_value /= denom
    t_value /= exp(lgamma(0.5) + lgamma(alpha) - lgamma(0.5 + alpha))
    return t_value


@cython.cdivision(True)
cpdef inline double calculate_prior_double_cy(
        double point, double alpha, double beta, double  mu, double kappa):
    cdef double denom, t_value, exponent, temp
    # if kappa == 0.0:
    #     return 1.0 if point == mu else 0.0
    # denom = 2 * beta * (kappa + 1)
    # denom /= kappa
    denom = 2 * beta * (kappa + 1) / kappa
    t_value = (point - mu)**2 / denom + 1
    exponent = -(alpha + 0.5)
    t_value **= exponent
    denom = sqrt(denom)
    t_value /= denom
    # t_value /= exp(lgamma(0.5) + lgamma(alpha) - lgamma(0.5 + alpha))
    temp = csc.beta(0.5, alpha)
    t_value /= temp
    # t_value *= exp(lgamma(0.5 + alpha) - (lgamma(0.5) + lgamma(alpha)))
    return t_value


cdef real_type beta_scalar_scalar(real_type val_1, real_type val_2):
    # """ return beta function for given values."""
    return exp(lgamma(val_1) + lgamma(val_2) - lgamma(val_1 + val_2))

# cdef double[:] beta(double val_1, double[:] val_2, int length):
#     """ return beta function for given values."""
#     idx: int = 0
#
#     for idx in range(length):
#         out[idx] = exp(lgamma(val_1) + lgamma(val_2[idx]) - lgamma(val_1 + val_2[idx]))

cdef void beta_scalar_arr(double val_1, double[:] val_2, double[:] out):
    """ return beta function for given values."""
    cdef Py_ssize_t idx
    idx = 0
    for idx in range(out.shape[0]):
        out[idx] = exp(lgamma(val_1) + lgamma(val_2[idx]) - lgamma(val_1 + val_2[idx]))
    return


# cdef beta(double val_1, double val_2):
#     """ return beta function for given values."""
#     return exp(lgamma(val_1) + lgamma(val_2) - lgamma(val_1 + val_2))


cpdef bayesian_online_changepoint_detection_deque_cy(
double[:] data, double mu, double kappa, double alpha, double beta, double lamb):
    """ """
    cdef Py_ssize_t out_size, arr_size, idx, jdx
    cdef double[:] out_view, data_view
    cdef double accumulator, max_val, val_prob, event
    out_size = data.shape[0]
    out = np.ndarray(shape=(out_size,), dtype=np.float64)
    out_view = out
    data_view = data
    cdef int run_length, prev_max, curr_max, max_idx, cps
    cdef deque[int] run_lengths
    cdef deque[double] probabilities
    cdef deque[NormInverseGammaCy] parameters
    initial_params = NormInverseGammaCy(
        alpha=alpha, beta=beta, mu=mu, kappa=kappa)
    run_length = 1  # Iterations since last changepoint
    run_lengths.push_back(0)
    parameters.push_back(initial_params)
    probabilities.push_back(1.0)
    accumulator = 0.0
    cps = 0
    curr_max = 0
    # print('this worked: 2')
    for idx in range(out_size):
        if idx % 1_000 == 0:
            print(idx)
        # print(f'outer loop level: {idx}')
        event = data_view[idx]
        run_lengths, probabilities = calculate_probabilities_deque(event, lamb, parameters, run_lengths, probabilities)
        # find and append max probability
        max_idx, max_val = 0, -1e99
        arr_size = probabilities.size()
        # print(f'Size of probability vector: {arr_size}')
        for jdx in range(arr_size):
            run_length = run_lengths[jdx]
            probability = probabilities[jdx]
            if probability > max_val:
                max_idx = run_length
                max_val = probability
        prev_max, curr_max = curr_max, max_idx
        if curr_max < prev_max:
            # print('a change has occurred in the past')
            # event is an attack
            run_length, accumulator = update_attack_cy(event)
            # reset params
            parameters.clear()
            parameters.push_back(NormInverseGammaCy(
                alpha=alpha, beta=beta, mu=mu, kappa=kappa))
            probabilities.clear()
            probabilities.push_back(1.0)
            run_lengths.clear()
            run_lengths.push_back(0)
            # print('all container reset')
        else:
            run_length, accumulator, parameters = update_no_attack_deque(
                event, run_length, accumulator, parameters, alpha, beta, mu, kappa)
            # Calculate probability of change point
        arr_size = probabilities.size()
        attack_probs = calculate_prior_deque_list(event, parameters)
        val_prob = 0.0
        for idx in range(arr_size):
            val_prob += attack_probs[idx] * probabilities[idx]
        # attack_probs *= probabilities
        # val_prob = attack_probs.sum()
        # is_attack = val_prob <= 0.05
        # out_view[idx] = is_attack
        out_view[idx] = val_prob
    # print('everything worked: 3')
    return out


cdef (int, double) update_attack_cy(double event):
    cdef double accum
    cdef int run_length
    accum = event
    run_length = 1
    return run_length, accum


cdef (deque[int], deque[double]) calculate_probabilities_deque(
        double event,  double lamb, deque[NormInverseGammaCy] params, deque[int] run_lengths,
        deque[double] probabilities):
    """ """
    # note: deques are pass-by-value
    # print('made it to prob calc')
    cdef double hazard, head, neg_prob
    cdef Py_ssize_t arr_size, priors_size
    cdef deque[double] priors
    arr_size = params.size()
    # print(arr_size)
    hazard = 1 / lamb
    priors = calculate_prior_deque_list(event, params)
    head = 0.0
    for idx in range(arr_size):
        head += priors[idx]
    head *= hazard
    neg_prob = 1 - hazard
    # tail = [prior * neg_prob for prior in priors]
    # Update probabilities
    # for idx in range(arr_size):
    #     probabilities[idx] = tail[idx]
    #     run_lengths[idx] += 1
    for idx in range(arr_size):
        probabilities[idx] = priors[idx] * neg_prob
        run_lengths[idx] += 1
    probabilities.push_front(head)
    run_lengths.push_front(0)
    arr_size = probabilities.size()
    # Normalize vector
    cdef float prob_sum = 0.0
    for idx in range(arr_size):
        prob_sum += probabilities[idx]
    # prob_sum = sum(probabilities)
    if prob_sum != 0.0:
        for idx in range(arr_size):
            probabilities[idx] /= prob_sum
            # probabilities[0] /= prob_sum
            # probabilities.rotate(1)
    # print('made it to end')
    # print(arr_size)
    # # Truncate values near zero
    # probabilities.rotate(1)
    # for idx in range(arr_size):
    #     if probabilities[0] < trunc_threshold:
    #         probabilities.popleft()
    #         params.popleft()
    #         run_lengths.popleft()
    #     else:
    #         probabilities.rotate(1)
    #         params.rotate(1)
    #         run_lengths.rotate(1)
    return run_lengths, probabilities


cdef deque[double] calculate_prior_deque_list(double point, deque[NormInverseGammaCy] params):
    """ """
    cdef deque[double] out
    cdef double val
    cdef NormInverseGammaCy param
    cdef Py_ssize_t idx, length
    length = params.size()
    for idx in range(length):
        param = params[idx]
        val = calculate_prior_double_cy(
        point, param.alpha, param.beta, param.mu, param.kappa)
        out.push_back(val)
    return out


cdef (int, double, deque[NormInverseGammaCy]) update_no_attack_deque(
        double event, int run_length, double accumulator,
        deque[NormInverseGammaCy] params, double alpha, double beta,
        double mu, double kappa):
    """ Update if no attack was detected.


    """
    cdef double new_accumulator, kappa_plus
    cdef int new_run_length
    cdef NormInverseGammaCy param
    cdef deque[NormInverseGammaCy] new_params
    cdef Py_ssize_t idx, length
    new_accumulator = event + accumulator
    new_run_length = run_length + 1
    length = params.size()
    for idx in range(length):
        param = params[idx]
        kappa_plus = param.kappa + 1
        param.beta += param.kappa * (event - param.mu)**2 / (2 * kappa_plus)
        param.mu = (param.kappa * param.mu + event) / kappa_plus
        param.alpha += 0.5
        param.kappa += 1.0
        new_params.push_back(param)
    # params.push_front(NormInverseGammaCy(
    #     alpha=alpha, beta=beta, mu=mu, kappa=kappa))
    new_params.push_front(NormInverseGammaCy(
        alpha=alpha, beta=beta, mu=mu, kappa=kappa))
    return new_run_length, new_accumulator, new_params
