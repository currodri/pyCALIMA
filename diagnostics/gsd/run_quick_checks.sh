#!/usr/bin/env bash
# The fast GSD check, `pytest -m gsd_quick` (about a minute, one thread), on the laptop or on megatron.
# Run it from the repository root before every commit of the GSD work:
#
#   diagnostics/gsd/run_quick_checks.sh                 # laptop
#   CORES=74 diagnostics/gsd/run_quick_checks.sh        # megatron: pinned to the given core(s), nice 19, idle I/O
#
# PY selects the interpreter (default: python). Tests that need data not present here (DustEM, Jones 2012, PAHdb)
# skip. Extra arguments go to pytest (e.g. -x, -k collide).
set -euo pipefail
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
cmd=("${PY:-python}" -m pytest -m gsd_quick -q -p no:cacheprovider "$@")
if [[ -n "${CORES:-}" ]] && command -v taskset >/dev/null; then
    cmd=(nice -n 19 ionice -c 3 taskset -c "$CORES" "${cmd[@]}")
fi
exec "${cmd[@]}"
