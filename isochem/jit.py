import os

USE_JIT = os.environ.get("ISOCHEM_USE_JIT", "1") == "1"
USE_CACHE = os.environ.get("ISOCHEM_USE_CACHE", "1") == "1"

try:
    if USE_JIT:
        from numba import jit as _jit
    else:
        _jit = None
except ImportError:
    _jit = None
    USE_JIT = False


def jit(*args, **kwargs):
    """
    Drop-in replacement for numba.jit.
    Acts as a no-op decorator if JIT is disabled.
    Provides sensible defaults: nopython=True, parallel=True, cache=USE_CACHE.
    """
    if _jit is None:
        return lambda f: f

    # Inject defaults only if not explicitly provided
    kwargs.setdefault("nopython", True)
    kwargs.setdefault("cache", USE_CACHE)

    return _jit(*args, **kwargs)