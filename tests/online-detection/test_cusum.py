import tracemalloc

import numpy as np

from online_detection.cusum import cusum_alg_generator

def get_benchmark_vars():
    mean = 20.0
    std_dev = 5.0
    h = 5
    alpha = 0.95
    data_size = 10_000
    rng = np.random.default_rng()
    data = rng.normal(mean, std_dev, size=data_size)
    return data, mean, std_dev, h, alpha

def test_benchmark_cusum_alg(benchmark):
    data, mean, std_dev, h, alpha = get_benchmark_vars()
    model_gen = cusum_alg_generator(data, mean, std_dev, h, alpha)
    benchmark(lambda: [item for item in model_gen])

def test_profile_cusum_alg():
    """ """
    data, mean, std_dev, h, alpha = get_benchmark_vars()
    kib_scalar = 1_024
    kib_unit = 'KiB'
    print('\n')
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    model_gen = cusum_alg_generator(data, mean, std_dev, h, alpha)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After generator current memory usage: {curr / kib_scalar} {kib_unit}, peak of {peak / kib_scalar} {kib_unit}')
    tracemalloc.reset_peak()
    # Take 1 element
    next(model_gen)
    print(tracemalloc.get_traced_memory())
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After calling next once current memory usage: {curr / kib_scalar:.4} {kib_unit}, peak of {peak / kib_scalar:.4} {kib_unit}')
    for item in model_gen:
        continue
    # out = [item for item in model_gen]
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After data collection current memory usage: {curr / kib_scalar:.4} {kib_unit}, peak of {peak / kib_scalar:.4} {kib_unit}')
