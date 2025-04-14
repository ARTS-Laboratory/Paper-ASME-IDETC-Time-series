import tracemalloc

import pytest
import warnings

import numpy as np

from online_detection import bocpd
from online_detection.bocpd import bayesian_online_changepoint_detection_v6_generator

try:
    from tda_project_rusty import run_bocpd, run_bocpd_inplace, copy_vector
except ModuleNotFoundError:
    warnings.warn('Rust module not included in environment.')
except ImportError:
    warnings.warn('Expected rust extensions to be available.')

def get_standard_cauchy():
    # point alpha beta mu kappa
    points = np.array([0.0, 1.0])
    alphas = np.array([0.5, 0.5])
    kappas = np.array([1.0, 1.0])
    mus = np.array([0.0, 0.0])
    # var 1 = (beta * (kappa + 1)) / (alpha * kappa) => beta = 0.25
    betas = np.array([0.25, 0.25])
    return points, alphas, betas, mus, kappas

def test_calculate_prior_arr_standard_cauchy():
    points, alphas, betas, mus, kappas = get_standard_cauchy()
    cauchy_priors = bocpd.calculate_prior_arr(points, alphas, betas, mus, kappas)
    cauchy_out = np.array([1/np.pi, 1/(2 * np.pi)])
    # cauchy standard distribution
    # cauchy standard pdf [f(x, 0, 1)] => 1 / (pi * (1 + x^2))
    assert np.all(cauchy_priors == cauchy_out)



def get_benchmark_vars():
    vec_size = 10_000
    rng = np.random.default_rng()
    alpha, beta, mu, kappa, lamb = 1.0, 1.0, 0.0, 2.0, 2.0
    std_dev = 1.0
    points = rng.normal(mu, std_dev, size=vec_size)
    return points, alpha, beta, mu, kappa, lamb


def test_benchmark_copy_from_rust(benchmark):
    points, _, _, _, _, _ = get_benchmark_vars()
    out = np.empty_like(points)
    try:
        benchmark(copy_vector, points, out)
    except NameError:
        pytest.skip("Rust version not available.")

def test_benchmark_bocpd_from_rust(benchmark):
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    try:
        rust_priors = run_bocpd(points, mu, kappa, alpha, beta, lamb)
        benchmark(run_bocpd, points, mu, kappa, alpha, beta, lamb)
    except NameError:
        pytest.skip("Rust version not available.")
    # rust_priors = run_bocpd(data_list, mu, kappa, alpha, beta, lamb)


def test_benchmark_bocpd_inplace_from_rust(benchmark):
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    out = np.empty_like(points)
    try:
        benchmark(run_bocpd_inplace, points, mu, kappa, alpha, beta, lamb, out)
    except NameError:
        pytest.skip("Rust version not available.")

def test_benchmark_bocpd_rust_hybrid(benchmark):
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    try:
        bocpd_model_gen = bocpd.bocpd_rust_hybrid(
            points, mu, kappa, alpha, beta, lamb)
        benchmark(lambda: [item for item in bocpd_model_gen])
    except NameError:
        pytest.skip("Rust version not available.")

def test_benchmark_bocpd_from_python(benchmark):
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    bocpd_model_gen = bayesian_online_changepoint_detection_v6_generator(
        points, mu, kappa, alpha, beta, lamb)
    benchmark(lambda: [item for item in bocpd_model_gen])

def test_profile_run_bocpd_from_rust():
    """ Test memory utilization of bocpd from rust with tracemalloc."""
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    kib_scalar = 1_024
    kib_unit = 'KiB'
    print('\n')
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    bocpd_model_gen = bocpd.bocpd_rust_hybrid(
        points, mu, kappa, alpha, beta, lamb)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After generator current memory usage: {curr / kib_scalar} {kib_unit}, peak of {peak / kib_scalar} {kib_unit}')
    tracemalloc.reset_peak()
    # Take 1 element
    next(bocpd_model_gen)
    print(tracemalloc.get_traced_memory())
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After calling next once current memory usage: {curr / kib_scalar} {kib_unit}, peak of {peak / kib_scalar} {kib_unit}')
    for item in bocpd_model_gen:
        continue
    # out = [item for item in bocpd_model_gen]
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After data collection current memory usage: {curr / kib_scalar} {kib_unit}, peak of {peak / kib_scalar} {kib_unit}')
    tracemalloc.reset_peak()

def test_profile_run_bocpd_from_python():
    """ Test memory utilization of bocpd with tracemalloc."""
    points, alpha, beta, mu, kappa, lamb = get_benchmark_vars()
    kib_scalar = 1_024
    kib_unit = 'KiB'
    print('\n')
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    bocpd_model_gen = bayesian_online_changepoint_detection_v6_generator(
        points, mu, kappa, alpha, beta, lamb)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After generator current memory usage: {curr / kib_scalar} {kib_unit}, peak of {peak / kib_scalar} {kib_unit}')
    tracemalloc.reset_peak()
    # Take 1 element
    next(bocpd_model_gen)
    print(tracemalloc.get_traced_memory())
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After calling next once current memory usage: {curr / kib_scalar:.4} {kib_unit}, peak of {peak / kib_scalar:.4} {kib_unit}')
    for item in bocpd_model_gen:
        continue
    # out = [item for item in bocpd_model_gen]
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After data collection current memory usage: {curr / kib_scalar:.4} {kib_unit}, peak of {peak / kib_scalar:.4} {kib_unit}')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    snapshot_end = tracemalloc.take_snapshot()
    top_n = 30
    scalar = 1024
    unit = 'KiB'
    top_stats = snapshot_end.statistics('traceback')
    print(f'Snapshot end contains {len(top_stats)} lines totaling {sum(stat.size for stat in top_stats) / scalar} {unit}')
    for stat in top_stats[:top_n]:
        print(stat)



def test_update_no_attack_arr():
    pass
