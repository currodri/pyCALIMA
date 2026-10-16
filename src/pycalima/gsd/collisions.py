"""Grain-grain collisions of one population: coagulation, cratering and
shattering as two-moment pair transfers, MPRK22 in time.

Each bin's power-law profile (``pl2m``) is represented by two nodes, the
lower and upper halves of its number split at the median, each carrying
N/2 at the mean mass of its half; the two nodes reproduce the bin's N and M
exactly (``bin_nodes``). A collision model gives, for every node pair
(p, q) and every outcome channel c (a ``Channel``), a rate coefficient
K_c(p, q) [cm^3 s^-1] and the products of one collision, for each of the
two grains:

* a remnant of mass m_rem (the merged grain for coagulation, the cratered
  target for shattering; 0 if none), deposited as one grain spread
  uniformly in mass over the width of its parent nodes' intervals (mean
  m_rem exactly). A point deposit would switch bins discontinuously as a
  node's mass crosses an edge: small cratering losses then pile grains at
  the lower edges and the step control chatters;
* fragments of total mass m_frag with dN/da ~ a^-alpha_f on
  [a_lo, a_hi], integrated exactly over the bins (number and mass).

Channels must be symmetric in the pair (the target is whichever grain is
larger), so only one ordering of each pair is evaluated.

Pairs collide at r = K w_p w_q / 2 (ordered pairs, so each collision is
counted once), removing one grain from each node. Products are attributed
to the two donors in proportion to their masses. Fragment spectra that
cover whole bins are deposited with difference arrays (their in-bin
integrals are the same for every pair up to a factor), so a pair costs O(1)
plus its two partial end bins (the self-similar O(N^2) idea of Rafikov,
Silsbee & Booth 2020).

Fragments below the grid go to the ``collector``: ``"smallest"`` keeps
them as grains at the smallest edge (Hirashita & Yan 2009), ``"remove"``
takes their mass out of the population (Hirashita & Kobayashi 2013; it is
returned as ``sink_M`` for the caller to put back in the gas, or not).
Products above the grid go to the overflow.

Mass moves between bins (plus overflow and sink) as a conservative
production-destruction system integrated with MPRK22 (Kopecz & Meister
2018): flows out of a bin are weighted by its new-to-old ratio, so mass is
conserved to round-off and stays positive for any step. Number losses are
Patankar-implicit in each bin (positive), number gains follow the delivered
mass. The first/second-order stage difference sizes the sub-steps.

The pair sums run in compiled code without the GIL on Python threads
(``run_rows``): ``default_threads()`` = the performance cores (or
``PYCALIMA_GSD_THREADS``), each marked user-interactive so that macOS keeps
them off the efficiency cores. numba's own threading layers are not used.

Collisions are within one population: different materials do not collide
with each other here.
"""

from __future__ import annotations

import math

import numpy as np

from pycalima.collapse._jit import njit

from .pl2m import BinProfile, int_pow
from .remap import realisable

COLLECTORS = ("smallest", "remove")


def bin_nodes(grid, N, M, widths=False):
    """Two nodes per bin: masses m[nb, 2] and weights w[nb, 2] (N/2 each)
    reproducing N and M of every bin. Empty bins get zero weight. With
    ``widths``, also the nodes' mass intervals (``node_widths``) from the
    same profile."""
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    prof = BinProfile(grid, N, M)
    nb = grid.n_bins
    mL = grid.m_edges[:-1]
    m = np.tile(grid.m_centres[:, None], (1, 2)).astype(float)
    w = np.zeros((nb, 2))
    filled = N > 0
    w[filled] = 0.5 * N[filled, None]
    uR = prof.uR
    p = prof.lam + 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u_med = np.where(np.abs(p) < 1e-12, np.sqrt(uR), (0.5 * (1.0 + uR ** p)) ** (1.0 / p))
    pl = filled & ~prof.delta
    lo = int_pow(prof.lam[pl] + 3.0, 1.0, u_med[pl]) / int_pow(prof.lam[pl], 1.0, u_med[pl])
    m[pl, 0] = mL[pl] * lo
    # the upper node takes the rest of the mass exactly
    m[pl, 1] = 2.0 * M[pl] / N[pl] - m[pl, 0]
    sp = filled & prof.delta
    m[sp, :] = (mL[sp] * prof.u_delta[sp] ** 3)[:, None]
    if widths:
        m_med = mL * u_med ** 3
        wid = np.stack([m_med - mL, grid.m_edges[1:] - m_med], axis=1)
        wid[~filled | prof.delta] = 0.0
        return m, w, wid
    return m, w


def node_widths(grid, N, M):
    """Mass widths of the two node intervals of every bin, [m_L, m_med] and
    [m_med, m_R] (0 for spikes and empty bins)."""
    prof = BinProfile(grid, np.asarray(N, float), np.asarray(M, float))
    mL, mR = grid.m_edges[:-1], grid.m_edges[1:]
    p = prof.lam + 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u_med = np.where(np.abs(p) < 1e-12, np.sqrt(prof.uR),
                         (0.5 * (1.0 + prof.uR ** p)) ** (1.0 / p))
    m_med = mL * u_med ** 3
    wid = np.stack([m_med - mL, mR - m_med], axis=1)
    wid[(np.asarray(N) <= 0) | prof.delta] = 0.0
    return wid


@njit(cache=True, nogil=True)
def _bin_log(edges, lx, geo, x=-1.0):
    """_bin for ln x on a geometric grid: geo = (ln edges[0], 1 / ln ratio);
    O(1), corrected against the edges for round-off (pass x if known)."""
    nb = edges.size - 1
    k = int(math.floor((lx - geo[0]) * geo[1]))
    if k < 0:
        k = -1 if lx < geo[0] else 0
    if k > nb:
        k = nb
    if x < 0.0:
        x = math.exp(lx)
    while k > 0 and k <= nb and x < edges[k]:
        k -= 1
    while 0 <= k < nb and x >= edges[k + 1]:
        k += 1
    if k == 0 and x < edges[0]:
        k = -1
    return k


@njit(cache=True)
def _bin(edges, x):
    nb = edges.size - 1
    if x < edges[0]:
        return -1
    if x >= edges[nb]:
        return nb
    lo, hi = 0, nb
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if edges[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True, nogil=True)
def _deposit_segment(F, H, i, j, fp, fq, r, mr, hw, m_edges, keep_small, geo_m):
    """One grain per collision (rate r) with mass spread uniformly over
    [mr - hw, mr + hw]: number and mass split across the bins it overlaps
    (mean exactly mr), so the deposit is continuous in the node masses."""
    nb = m_edges.size - 1
    lo = mr - hw
    hi = mr + hw
    k = _bin_log(m_edges, math.log(mr), geo_m, mr)
    if hw <= 0.0 or (0 <= k < nb and lo >= m_edges[k] and hi < m_edges[k + 1]):
        if k >= 0:
            F[k, i] += r * mr * fp
            F[k, j] += r * mr * fq
            H[k, i] += r * fp
            H[k, j] += r * fq
            return
        lo = mr                          # all below the grid
        hi = mr
    if lo < m_edges[0]:
        # the part below the grid goes to the collector, like fragments
        x2 = min(hi, m_edges[0])
        frac = 1.0 if hi <= lo else (x2 - lo) / (2.0 * hw)
        dm = r * frac * (0.5 * (lo + x2) if hi > lo else mr)
        k = 0 if keep_small else nb + 1
        F[k, i] += dm * fp
        F[k, j] += dm * fq
        if keep_small:
            # as grains at the smallest edge: mass kept, number follows
            H[0, i] += dm / m_edges[0] * fp
            H[0, j] += dm / m_edges[0] * fq
        if hi <= m_edges[0]:
            return
        lo = m_edges[0]
    k0 = _bin_log(m_edges, math.log(lo), geo_m, lo)
    k1 = _bin_log(m_edges, math.log(hi), geo_m, hi)
    for k in range(k0, k1 + 1):
        x1 = max(lo, m_edges[k])
        x2 = min(hi, m_edges[k + 1]) if k < nb else hi
        if x2 <= x1:
            continue
        dn = r * (x2 - x1) / (2.0 * hw)
        dm = dn * 0.5 * (x1 + x2)
        F[k, i] += dm * fp
        F[k, j] += dm * fq
        H[k, i] += dn * fp
        H[k, j] += dn * fq


@njit(cache=True, nogil=True)
def _fragments(F, H, D, i, j, fp, fq, r, mf, lo, hi, m_edges, a_edges, GM, GN, eM, eN,
               mfac, keep_small, geo_a, lo_pre=None):
    """Deposit r * mf of fragments with dN/da ~ a^-alpha on [lo, hi].
    ``lo_pre`` = (ln lo, lo^eM, lo^eN) if precomputed (a fixed lower edge)."""
    nb = m_edges.size - 1
    a0, aT = a_edges[0], a_edges[nb]
    if hi <= lo or hi <= a0:
        # no fragment reaches the grid: all of it to the collector
        dm = r * mf
        if keep_small:
            F[0, i] += dm * fp
            F[0, j] += dm * fq
            H[0, i] += dm / m_edges[0] * fp
            H[0, j] += dm / m_edges[0] * fq
        else:
            F[nb + 1, i] += dm * fp
            F[nb + 1, j] += dm * fq
        return
    if lo_pre is None:
        Llo = math.log(lo)
        PloM, PloN = math.exp(eM * Llo), math.exp(eN * Llo)
    else:
        Llo, PloM, PloN = lo_pre
    Lhi = math.log(hi)
    PhiM, PhiN = math.exp(eM * Lhi), math.exp(eN * Lhi)
    cM = r * mf * eM / (PhiM - PloM)      # mass = cM (y2^eM - y1^eM) / eM
    cN = cM / mfac
    # below the grid: collector
    if lo < a0:
        dm = cM * (GM[0] - PloM) / eM
        if keep_small:
            F[0, i] += dm * fp
            F[0, j] += dm * fq
            H[0, i] += dm / m_edges[0] * fp
            H[0, j] += dm / m_edges[0] * fq
        else:
            F[nb + 1, i] += dm * fp
            F[nb + 1, j] += dm * fq
    # above the grid: overflow
    if hi > aT:
        up = lo > aT
        dm = cM * (PhiM - (PloM if up else GM[nb])) / eM
        dn = cN * (PhiN - (PloN if up else GN[nb])) / eN
        F[nb, i] += dm * fp
        F[nb, j] += dm * fq
        H[nb, i] += dn * fp
        H[nb, j] += dn * fq
    if lo >= aT:
        return
    inside_lo = lo > a0
    inside_hi = hi < aT
    j0 = _bin_log(a_edges, Llo, geo_a, lo) if inside_lo else 0
    j1 = _bin_log(a_edges, Lhi, geo_a, hi) if inside_hi else nb - 1
    if j1 > nb - 1:
        j1 = nb - 1
    # partial end bins exactly, the bins between by difference array
    for jj in (j0, j1):
        l_in = inside_lo and lo > a_edges[jj]
        h_in = inside_hi and hi < a_edges[jj + 1]
        y1M = PloM if l_in else GM[jj]
        y2M = PhiM if h_in else GM[jj + 1]
        y1N = PloN if l_in else GN[jj]
        y2N = PhiN if h_in else GN[jj + 1]
        dm = cM * (y2M - y1M) / eM
        dn = cN * (y2N - y1N) / eN
        F[jj, i] += dm * fp
        F[jj, j] += dm * fq
        H[jj, i] += dn * fp
        H[jj, j] += dn * fq
        if j1 == j0:
            break
    if j1 - j0 >= 2:
        D[j0 + 1, i] += cM * fp
        D[j0 + 1, j] += cM * fq
        D[j1, i] -= cM * fp
        D[j1, j] -= cM * fq


@njit(cache=True, nogil=True)
def _finish(F, H, D, GM, GN, eM, eN, mfac):
    """Full-bin deposits from the difference array; zero the diagonal."""
    nb = D.shape[1]
    for i in range(nb):
        acc = 0.0
        for jj in range(nb):
            acc += D[jj, i]
            if acc != 0.0:
                F[jj, i] += acc * (GM[jj + 1] - GM[jj]) / eM
                H[jj, i] += acc / mfac * (GN[jj + 1] - GN[jj]) / eN
    for k in range(F.shape[0]):
        F[k, k] = 0.0                    # mass staying in its bin does not move


@njit(cache=True, nogil=True)
def _finish_blocks(F, H, D, a_edges, rho, alpha):
    mfac = 4.0 / 3.0 * math.pi * rho
    eM = 4.0 - alpha
    eN = 1.0 - alpha
    _finish(F, H, D, a_edges ** eM, a_edges ** eN, eM, eN, mfac)


def run_rows(rows, args, nb, a_edges, rho, alpha, n_threads, n_chunks=None):
    """Sum a nogil rows kernel ``rows(t, T, *args, F, L, H, D)`` (rows t,
    t + T, ... of the pair sum) over T = n_chunks interleaved chunks, on
    ``n_threads`` Python threads that each pull chunks from a shared counter
    into their own accumulators (fast cores take more; Apple chips mix
    performance and efficiency cores). Returns (F, L, H) after ``_finish``."""
    import threading

    nz = nb + 2
    T = n_chunks or 4 * n_threads
    lock = threading.Lock()
    nxt = [0]

    def work():
        _prefer_performance_cores()
        F = np.zeros((nz, nz))
        L = np.zeros(nb)
        H = np.zeros((nb + 1, nb))
        D = np.zeros((nb + 1, nb))
        while True:
            with lock:
                t = nxt[0]
                nxt[0] += 1
            if t >= T:
                return F, L, H, D
            rows(t, T, *args, F, L, H, D)

    _LAST_THREADS[0] = max(1, n_threads)
    if n_threads <= 1:
        F, L, H, D = work()
    else:
        parts = [f.result() for f in [_pool(n_threads).submit(work) for _ in range(n_threads)]]
        F, L, H, D = (sum(p_[k] for p_ in parts) for k in range(4))
    _finish_blocks(F, H, D, a_edges, rho, alpha)
    return F, L, H


def performance_cores():
    """Number of performance cores (Apple silicon: hw.perflevel0; else all
    CPUs)."""
    import os
    import sys

    if sys.platform == "darwin":
        try:
            import ctypes

            n = ctypes.c_int(0)
            size = ctypes.c_size_t(ctypes.sizeof(n))
            lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            if lib.sysctlbyname(b"hw.perflevel0.physicalcpu", ctypes.byref(n),
                                ctypes.byref(size), None, 0) == 0 and n.value > 0:
                return int(n.value)
        except Exception:  # pragma: no cover
            pass
    return os.cpu_count() or 1


_QOS_SET = set()


def _prefer_performance_cores():
    """Mark the calling thread user-interactive (macOS QoS), which the
    scheduler runs on performance cores; a no-op elsewhere."""
    import sys
    import threading

    tid = threading.get_ident()
    if sys.platform != "darwin" or tid in _QOS_SET:
        return
    try:
        import ctypes

        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.pthread_set_qos_class_self_np(ctypes.c_uint(0x21), ctypes.c_int(0))
    except Exception:  # pragma: no cover
        pass
    _QOS_SET.add(tid)


_LAST_THREADS = [1]


def default_threads():
    """Threads for the pair sums: ``PYCALIMA_GSD_THREADS`` if set, else the
    performance cores (efficiency cores, ~3x slower, would set the pace)
    that are free: their number less the 1-minute load average not due to
    our own last pair sum. Oversubscribing a busy machine makes a pair sum
    slower than serial."""
    import os

    env = os.environ.get("PYCALIMA_GSD_THREADS")
    if env:
        return max(1, int(env))
    n_p = performance_cores()
    try:
        load = os.getloadavg()[0] - _LAST_THREADS[0]
    except OSError:  # pragma: no cover
        load = 0.0
    free = (os.cpu_count() or n_p) - max(load, 0.0)
    return int(max(1, min(n_p, math.floor(free))))


_POOL = None


def _pool(n):
    global _POOL
    if _POOL is None or _POOL._max_workers < n:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=n)
    return _POOL


def run_rows(rows, args, nb, a_edges, rho, alpha, n_threads, n_chunks=None):
    """Sum a nogil rows kernel ``rows(t, T, *args, F, L, H, D)`` (rows t,
    t + T, ... of the pair sum) over T = n_chunks interleaved chunks, on
    ``n_threads`` Python threads that each pull chunks from a shared counter
    into their own accumulators (fast cores take more; Apple chips mix
    performance and efficiency cores). Returns (F, L, H) after ``_finish``."""
    import threading

    nz = nb + 2
    T = n_chunks or 4 * n_threads
    lock = threading.Lock()
    nxt = [0]

    def work():
        _prefer_performance_cores()
        F = np.zeros((nz, nz))
        L = np.zeros(nb)
        H = np.zeros((nb + 1, nb))
        D = np.zeros((nb + 1, nb))
        while True:
            with lock:
                t = nxt[0]
                nxt[0] += 1
            if t >= T:
                return F, L, H, D
            rows(t, T, *args, F, L, H, D)

    _LAST_THREADS[0] = max(1, n_threads)
    if n_threads <= 1:
        F, L, H, D = work()
    else:
        parts = [f.result() for f in [_pool(n_threads).submit(work) for _ in range(n_threads)]]
        F, L, H, D = (sum(p_[k] for p_ in parts) for k in range(4))
    _finish_blocks(F, H, D, a_edges, rho, alpha)
    return F, L, H


def performance_cores():
    """Number of performance cores (Apple silicon: hw.perflevel0; else all
    CPUs)."""
    import os
    import sys

    if sys.platform == "darwin":
        try:
            import ctypes

            n = ctypes.c_int(0)
            size = ctypes.c_size_t(ctypes.sizeof(n))
            lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            if lib.sysctlbyname(b"hw.perflevel0.physicalcpu", ctypes.byref(n),
                                ctypes.byref(size), None, 0) == 0 and n.value > 0:
                return int(n.value)
        except Exception:  # pragma: no cover
            pass
    return os.cpu_count() or 1


_QOS_SET = set()


def _prefer_performance_cores():
    """Mark the calling thread user-interactive (macOS QoS), which the
    scheduler runs on performance cores; a no-op elsewhere."""
    import sys
    import threading

    tid = threading.get_ident()
    if sys.platform != "darwin" or tid in _QOS_SET:
        return
    try:
        import ctypes

        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.pthread_set_qos_class_self_np(ctypes.c_uint(0x21), ctypes.c_int(0))
    except Exception:  # pragma: no cover
        pass
    _QOS_SET.add(tid)


_LAST_THREADS = [1]


def default_threads():
    """Threads for the pair sums: ``PYCALIMA_GSD_THREADS`` if set, else the
    performance cores (efficiency cores, ~3x slower, would set the pace)
    that are free: their number less the 1-minute load average not due to
    our own last pair sum. Oversubscribing a busy machine makes a pair sum
    slower than serial."""
    import os

    env = os.environ.get("PYCALIMA_GSD_THREADS")
    if env:
        return max(1, int(env))
    n_p = performance_cores()
    try:
        load = os.getloadavg()[0] - _LAST_THREADS[0]
    except OSError:  # pragma: no cover
        load = 0.0
    free = (os.cpu_count() or n_p) - max(load, 0.0)
    return int(max(1, min(n_p, math.floor(free))))


_POOL = None


def _pool(n):
    global _POOL
    if _POOL is None or _POOL._max_workers < n:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=n)
    return _POOL


def _pair_terms_threads(m, w, wid, K, rem, frag, a_hi, a_lo, m_edges, a_edges, rho, alpha,
                        keep_small, n_threads):
    T = n_threads
    futs = [_pool(T).submit(_pair_block, t, T, m, w, wid, K, rem, frag, a_hi, a_lo, m_edges,
                            a_edges, rho, alpha, keep_small) for t in range(T)]
    parts = [f.result() for f in futs]
    F, L, H, D = (sum(p[k] for p in parts) for k in range(4))
    _finish_blocks(F, H, D, a_edges, rho, alpha)
    return F, L, H


PARALLEL_MIN_PAIRS = 100_000   # channel x pair evaluations above which threads pay


def _pair_terms(m, w, wid, K, rem, frag, a_hi, a_lo, m_edges, a_edges, rho, alpha,
                keep_small, threads=None):
    """Mass flows F[k, i] (donor bin i to node k; k = nb overflow, nb + 1
    sink), number losses L[i] and number gains H[k, i] per unit donor
    weight (k <= nb).

    K[c, p, q] is the rate coefficient of channel c; rem, frag, a_hi and
    a_lo have an extra leading axis of two product sets (s = 0: products of
    grain p, s = 1: of grain q; a merger puts everything in set 0). Channels
    must be symmetric under p <-> q with the sets swapped, so only q >= p is
    visited. Large problems run on ``default_threads()`` threads."""
    P = K.shape[1]
    work = K.shape[0] * P * (P + 1) // 2
    if threads is None:
        threads = default_threads() if work >= PARALLEL_MIN_PAIRS else 1
    nb = m.shape[0]
    mfac = 4.0 / 3.0 * math.pi * rho
    eM, eN = 4.0 - alpha, 1.0 - alpha
    GM, GN = a_edges ** eM, a_edges ** eN
    args = (P, m, w, wid, K, rem, frag, a_hi, a_lo, m_edges, a_edges, GM, GN, eM, eN, mfac,
            keep_small)
    return run_rows(_pair_rows, args, nb, a_edges, rho, alpha, threads)


@njit(cache=True, nogil=True)
def _pair_rows(t, T, P, m, w, wid, K, rem, frag, a_hi, a_lo, m_edges, a_edges, GM, GN,
               eM, eN, mfac, keep_small, F, L, H, D):
    """Rows p = t, t + T, ... of the pair sum into one thread's arrays."""
    geo_m = np.array([math.log(m_edges[0]), 1.0 / math.log(m_edges[1] / m_edges[0])])
    geo_a = np.array([math.log(a_edges[0]), 1.0 / math.log(a_edges[1] / a_edges[0])])
    for c in range(K.shape[0]):
        for p in range(t, P, T):
            i, ia = p // 2, p % 2
            wp = w[i, ia]
            if wp <= 0.0:
                continue
            mp = m[i, ia]
            for q in range(p, P):
                j, jb = q // 2, q % 2
                wq = w[j, jb]
                if wq <= 0.0:
                    continue
                # all collisions between nodes p and q (half that for p = q)
                r = (0.5 if q == p else 1.0) * K[c, p, q] * wp * wq
                if r <= 0.0:
                    continue
                mq = m[j, jb]
                fp = mp / (mp + mq)
                fq = 1.0 - fp
                L[i] += r
                L[j] += r
                for st in range(2):
                    mr = rem[st, c, p, q]
                    if mr > 0.0:
                        # spread like the parent node(s): a cratered remnant of
                        # p keeps p's width scaled by mr/mp, a merger both widths
                        m_own, w_own, m_oth, w_oth = ((mp, wid[i, ia], mq, wid[j, jb]) if st == 0
                                                      else (mq, wid[j, jb], mp, wid[i, ia]))
                        hw = 0.5 * (w_own * min(mr / m_own, 1.0)
                                    + w_oth * max(0.0, min((mr - m_own) / m_oth, 1.0)))
                        _deposit_segment(F, H, i, j, fp, fq, r, mr, hw, m_edges, keep_small,
                                         geo_m)
                    mf = frag[st, c, p, q]
                    if mf > 0.0:
                        _fragments(F, H, D, i, j, fp, fq, r, mf, a_lo[st, c, p, q],
                                   a_hi[st, c, p, q], m_edges, a_edges, GM, GN, eM, eN,
                                   mfac, keep_small, geo_a)


class CoagulationModel:
    """Perfect sticking: every collision merges the two grains. ``kernel``
    is K(m1, m2) [cm^3 s^-1], vectorised in grain masses."""

    alpha_f = 3.3
    collector = "smallest"

    def __init__(self, kernel):
        self.kernel = kernel

    def channels(self, a, m):
        K = np.asarray(self.kernel(m[:, None], m[None, :]), float)
        K = np.broadcast_to(K, (m.size, m.size))
        return [Channel(K, m[:, None] + m[None, :])]

    def stacked_channels(self, a, m):
        """Channel arrays for the kernel directly (one merger channel)."""
        P = m.size
        K = np.ascontiguousarray(np.broadcast_to(
            np.asarray(self.kernel(m[:, None], m[None, :]), float), (P, P)))[None]
        rem = np.zeros((2, 1, P, P))
        rem[0, 0] = m[:, None] + m[None, :]
        z = np.zeros((2, 1, P, P))
        o = np.ones((2, 1, P, P))
        return K, rem, z, o, o


class Channel:
    """One collision outcome for all node pairs (arrays [P, P]): rate
    coefficient K and the products of grain 1 (rows) and grain 2 (columns):
    remnant masses and fragment masses with their radius ranges. Unset
    products are empty; a merger puts the merged mass in ``rem1``."""

    def __init__(self, K, rem1=None, frag1=None, a_hi1=None, a_lo1=None,
                 rem2=None, frag2=None, a_hi2=None, a_lo2=None):
        z = np.zeros(np.shape(K))
        o = z + 1.0
        self.K = K
        self.sets = ((z if rem1 is None else rem1, z if frag1 is None else frag1,
                      o if a_hi1 is None else a_hi1, o if a_lo1 is None else a_lo1),
                     (z if rem2 is None else rem2, z if frag2 is None else frag2,
                      o if a_hi2 is None else a_hi2, o if a_lo2 is None else a_lo2))


def _terms(grid, N, M, model):
    m, w, wid = bin_nodes(grid, N, M, widths=True)
    mf = m.reshape(-1)
    a = grid.radius(mf)
    fused = getattr(model, "pair_terms", None)
    if fused is not None:
        out = fused(grid, m, w, wid, a, mf)
        if out is not None:
            return out
    fast = getattr(model, "stacked_channels", None)
    stack = fast(a, mf) if fast is not None else None
    if stack is None:
        chans = model.channels(a, mf)
        K = np.ascontiguousarray(np.array([ch.K for ch in chans], float))
        stack = [np.ascontiguousarray(np.array([[ch.sets[st][k] for ch in chans] for st in (0, 1)],
                                               float)) for k in range(4)]
        stack = [K] + stack
    if model.collector not in COLLECTORS:
        raise ValueError(f"collector must be one of {COLLECTORS}")
    if abs(model.alpha_f - 4.0) < 1e-9 or abs(model.alpha_f - 1.0) < 1e-9:
        raise ValueError("alpha_f = 1 or 4 is not supported (log integrals)")
    return _pair_terms(m, w, wid, *stack, grid.m_edges, grid.a_edges, float(grid.rho),
                       float(model.alpha_f), model.collector == "smallest")


def _patankar(z_old, z_den, F, h):
    """Solve z = z_old + h [F z/z_den - diag(colsum F) z/z_den] (columns of
    the system matrix sum to one: conservative and positive)."""
    n = z_old.size
    # nodes below 1e-30 of the total do not donate (and 1/z would overflow)
    tiny = 1e-30 * max(z_den.sum(), 1e-300)
    inv = np.zeros(n)
    big = z_den > tiny
    inv[big] = 1.0 / z_den[big]
    A = -h * F * inv[None, :]
    A[np.diag_indices(n)] = 1.0 + h * F.sum(axis=0) * inv
    return np.maximum(np.linalg.solve(A, z_old), 0.0)


def _number(N_old, N_den, theta, L, H, h, nb):
    """Number after a Patankar step: losses implicit in N, gains carried by
    the delivered mass (donor ratios theta). Returns (N[nb], N_overflow)."""
    gain = H @ theta[:nb]
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.where(N_den > 0, L / N_den, 0.0)
    N = (N_old + h * gain[:nb]) / (1.0 + h * loss)
    return N, h * gain[nb]


def collide(grid, N, M, model, dt, errmax=1e-3, atol=1e-8, h0=None, max_steps=100000,
            source=None, norm="max"):
    """Advance (N, M) over dt under the collision ``model``. The local error
    of every bin, relative to its value or to ``atol`` times the total
    (number or mass), is held below ``errmax``. ``source`` = (S_N, S_M),
    constant production rates per bin [cm^-3 s^-1, g cm^-3 s^-1], is added
    inside the step (an external supply, e.g. grains fed into a cascade).
    Returns a dict with N, M, overflow_N, overflow_M, sink_M (fragments
    removed below the grid) and n_steps."""
    nb = grid.n_bins
    S_N = np.zeros(nb) if source is None else np.asarray(source[0], float)
    S_M = np.zeros(nb + 2)
    if source is not None:
        S_M[:nb] = source[1]
    N = np.asarray(N, float).copy()
    z = np.concatenate([np.asarray(M, float), [0.0, 0.0]])   # bins, overflow, sink
    oN = 0.0
    t = 0.0
    h = dt if h0 is None else min(h0, dt)
    steps = 0
    while t < dt * (1.0 - 1e-12) and steps < max_steps:
        h = min(h, dt - t)
        F0, L0, H0 = _terms(grid, N, z[:nb], model)
        if not F0.any() and not L0.any() and source is None:
            break
        # stage 1 (first order): rates at the old state
        z1 = _patankar(z + h * S_M, z, F0, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            th1 = np.where(z > 0, z1 / z, 0.0)
        N1, _ = _number(N + h * S_N, N, th1, L0, H0, h, nb)
        N1, M1 = realisable(grid, N1, z1[:nb])
        # stage 2: mean of the rates, denominators at stage 1
        F1, L1, H1 = _terms(grid, N1, M1, model)
        z1[:nb] = M1
        Fm, Lm, Hm = 0.5 * (F0 + F1), 0.5 * (L0 + L1), 0.5 * (H0 + H1)
        z2 = _patankar(z + h * S_M, z1, Fm, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            th2 = np.where(z1 > 0, z2 / z1, 0.0)
        N2, oN2 = _number(N + h * S_N, N1, th2, Lm, Hm, h, nb)
        scale_z = np.maximum(np.maximum(z, z2), atol * max(z.sum(), 1e-300))
        scale_N = np.maximum(np.maximum(N, N2), atol * max(N.sum(), 1e-300))
        ez = np.abs(z2 - z1) / scale_z
        eN = np.abs(N2 - N1) / scale_N
        if norm == "rms":
            err = max(math.sqrt(np.mean(ez ** 2)), math.sqrt(np.mean(eN ** 2)))
        else:
            err = max(ez.max(), eN.max())
        if err <= errmax or h <= 1e-12 * dt:
            t += h
            steps += 1
            N, Mb = realisable(grid, N2, z2[:nb])
            z = np.concatenate([Mb, z2[nb:]])
            oN += oN2
        fac = 0.9 * math.sqrt(errmax / max(err, 1e-300))
        h *= min(5.0, max(0.2, fac))
    return {"N": N, "M": z[:nb], "overflow_N": oN, "overflow_M": z[nb],
            "sink_M": z[nb + 1], "n_steps": steps}
