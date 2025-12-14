import os

USE_JIT = os.environ.get("ISOCHEM_USE_JIT", "1") == "1"

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
    """
    if _jit is None:
        return lambda f: f
    return _jit(*args, **kwargs)