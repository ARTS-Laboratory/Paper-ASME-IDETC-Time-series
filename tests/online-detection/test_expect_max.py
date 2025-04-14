import math
import tracemalloc

import pytest
import warnings

import numpy as np

from online_detection.expect_Max import expectation_maximization_generator


def get_parameters():
    mean_1, var_1 = 0.0, 1.0
    mean_2, var_2 = 10.0, 2.0
    return mean_1, var_1, mean_2, var_2


def test_expectation_maximization_generator_all_normal():
    mean_1, var_1, mean_2, var_2 = get_parameters()
    safe_size, unsafe_size = 70, 30
    pi = 0.3
    epochs = 100
    num_unknowns = 1_000
    rng = np.random.default_rng()
    safe = rng.normal(mean_1, math.sqrt(var_1), safe_size)
    unsafe = rng.normal(mean_2, math.sqrt(var_2), unsafe_size)
    my_unknowns = rng.normal(mean_1, math.sqrt(var_1), num_unknowns)
    em_model_gen = expectation_maximization_generator(
        safe, unsafe, my_unknowns, mean_1, mean_2,
        var_1, var_2, pi, epochs)
    assert all([item == False for item in em_model_gen])

def test_expectation_maximization_generator_all_abnormal():
    mean_1, var_1, mean_2, var_2 = get_parameters()
    safe_size, unsafe_size = 70, 30
    pi = 0.3
    epochs = 100
    num_unknowns = 1_000
    rng = np.random.default_rng()
    safe = rng.normal(mean_1, math.sqrt(var_1), safe_size)
    unsafe = rng.normal(mean_2, math.sqrt(var_2), unsafe_size)
    my_unknowns = rng.normal(mean_2, math.sqrt(var_2), num_unknowns)
    em_model_gen = expectation_maximization_generator(
        safe, unsafe, my_unknowns, mean_1, mean_2,
        var_1, var_2, pi, epochs)
    assert all([item == True for item in em_model_gen])

def test_posterior_probs_v2_inplace():
    pytest.skip()
    in_vals = np.array([0.0, 0.5, -0.5])
    out = np.full((2, len(in_vals)), -np.inf)


def test_maximization():
    pytest.skip()

def get_benchmark_vars():
    mean_1, var_1, mean_2, var_2 = get_parameters()
    safe_mean, unsafe_mean = 0.0, 50.0
    safe_var, unsafe_var = 1.0, 4.0
    safe_size, unsafe_size = 70, 30
    pi = 0.3
    epochs = 100
    num_unknowns = 10_000
    rng = np.random.default_rng()
    safe = rng.normal(mean_1, math.sqrt(var_1), safe_size)
    unsafe = rng.normal(mean_2, math.sqrt(var_2), unsafe_size)
    my_unknowns = rng.normal(mean_2, math.sqrt(var_2), num_unknowns)
    param_dict = {
        'safe_mean': safe_mean, 'unsafe_mean': unsafe_mean,
        'safe_var': safe_var, 'unsafe_var': unsafe_var, 'pi': pi}
    return param_dict, safe, unsafe, my_unknowns, epochs


def test_benchmark_em_model_from_python(benchmark):
    params, safe, unsafe, my_unknowns, epochs = get_benchmark_vars()
    model_gen = expectation_maximization_generator(
        safe, unsafe, my_unknowns, params['safe_mean'], params['unsafe_mean'],
        params['safe_var'], params['unsafe_var'], params['pi'], epochs)
    benchmark(lambda: [item for item in model_gen])

def test_profile_run_em_from_python():
    """ Test memory utilization of expectation maximization with tracemalloc."""
    params, safe, unsafe, my_unknowns, epochs = get_benchmark_vars()
    kib_scalar = 1_024
    kib_unit = 'KiB'
    print('\n')
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    model_gen = expectation_maximization_generator(
        safe, unsafe, my_unknowns, params['safe_mean'], params['unsafe_mean'],
        params['safe_var'], params['unsafe_var'], params['pi'], epochs)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After generator current memory usage: {curr / kib_scalar} {kib_unit}, peak of {peak / kib_scalar} {kib_unit}')
    tracemalloc.reset_peak()
    # Take 1 element
    next(model_gen)
    print(tracemalloc.get_traced_memory())
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After calling next once current memory usage: {curr / kib_scalar:.4} {kib_unit}, peak of {peak / kib_scalar:.4} {kib_unit}')
    out = [item for item in model_gen]
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After data collection current memory usage: {curr / kib_scalar:.4} {kib_unit}, peak of {peak / kib_scalar:.4} {kib_unit}')
