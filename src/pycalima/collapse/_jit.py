"""``njit`` when numba is installed (the ``accel`` extra), a no-op otherwise."""

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f
