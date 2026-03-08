"""
Hot-path kernels for network link-load accumulation.

Uses Numba JIT when available (installed); falls back to plain NumPy otherwise.
Both implementations assume **no duplicate indices** in idx (one entry per edge).

Public API
----------
accumulate_max(arr, idx, vals, scale, throughput) -> float
    Add vals*scale into arr[idx] and return max(vals*scale)/throughput.

accumulate(arr, idx, vals, scale)
    Add vals*scale into arr[idx].

NUMBA_AVAILABLE : bool
    True when Numba was imported successfully.
"""

import numpy as np

try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _accum_max_jit(arr, idx, vals, scale, throughput):
        max_v = 0.0
        for i in range(len(idx)):
            v = vals[i] * scale
            arr[idx[i]] += v
            if v > max_v:
                max_v = v
        return max_v / throughput

    @_njit(cache=True)
    def _accum_jit(arr, idx, vals, scale):
        for i in range(len(idx)):
            arr[idx[i]] += vals[i] * scale

    def accumulate_max(arr, idx, vals, scale, throughput):
        """Accumulate vals*scale into arr[idx]; return max load / throughput."""
        if len(idx) == 0:
            return 0.0
        return float(_accum_max_jit(arr, idx, vals, scale, throughput))

    def accumulate(arr, idx, vals, scale):
        """Accumulate vals*scale into arr[idx]."""
        if len(idx) > 0:
            _accum_jit(arr, idx, vals, scale)

    NUMBA_AVAILABLE = True

    # Warm up JIT with tiny arrays so first real call has no latency spike.
    _dummy_arr = np.zeros(4, dtype=np.float64)
    _dummy_idx = np.zeros(2, dtype=np.int32)
    _dummy_val = np.ones(2, dtype=np.float64)
    accumulate_max(_dummy_arr, _dummy_idx, _dummy_val, 1.0, 1.0)
    accumulate(_dummy_arr, _dummy_idx, _dummy_val, 1.0)
    del _dummy_arr, _dummy_idx, _dummy_val

except ImportError:
    NUMBA_AVAILABLE = False

    def accumulate_max(arr, idx, vals, scale, throughput):
        """NumPy fallback: accumulate vals*scale into arr[idx]; return max/throughput."""
        if len(idx) == 0:
            return 0.0
        v = vals * scale
        arr[idx] += v          # safe: indices are unique per-edge coefficient dict
        return float(v.max()) / throughput

    def accumulate(arr, idx, vals, scale):
        """NumPy fallback: accumulate vals*scale into arr[idx]."""
        if len(idx) > 0:
            arr[idx] += vals * scale   # safe: no duplicate indices
