import tracemalloc

import numpy as np

from online_detection.grey_systems_model import grey_model_generator


def get_benchmark_vars():
    mean, std_dev = 0.0, 1.0
    window_size = 10
    c = 1.5
    c_ratio = 1.5
    data_size = 10_000
    rng = np.random.default_rng()
    data = rng.normal(mean, std_dev, size=data_size)
    return data, window_size, c, c_ratio

def test_profile_run_grey_model():
    data, window_size, c, c_ratio = get_benchmark_vars()
    kib_scalar = 1_024
    kib_unit = 'KiB'
    print('\n')
    tracemalloc.start(20)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage: {curr} B, peak of {peak} B')
    tracemalloc.reset_peak()
    print(tracemalloc.get_traced_memory())
    model_gen = grey_model_generator(
        data, window_size, c=c, c_ratio=c_ratio)
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
