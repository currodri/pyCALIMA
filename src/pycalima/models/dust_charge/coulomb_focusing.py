"""Coulomb focusing factors of a grain over its charge distribution, as RAMSES computes them
(``compute_Coulomb_focusing_ions`` in calima/dust_charging.f90).

D(j) is the mean, over the grain's charge distribution P(Z), of the Weingartner & Draine (1999,
eqs. 6-7) factor for an impactor of charge j, with c = |j| e^2/(a k T):

    Z j < 0 (attractive)    1 + |Z| c
    Z j > 0 (repulsive)     exp(-|Z| c)
    Z = 0, j != 0           1 + |j| sqrt(pi e^2/(2 a k T))

RAMSES knows only the charging model's <Z> and sigma_Z, so P(Z) is a Gaussian of that mean and
width on the integer charges:

* sigma < SIG_LO: the sum over every charge within K_WIN max(sigma, 1/2) of the mean, the last
  unit of that width tapered to zero weight (cos^2), so that charges enter and leave the sum
  with zero weight;
* sigma > SIG_HI: the integral over the continuous Gaussian, with the neutral charge as the bin
  [-1/2, 1/2] and the repulsive exponential taken at bin centres, i.e. the integral times
  (c/2)/sinh(c/2);
* in between, a C1 blend of the two.

D is therefore continuous in <Z>, sigma_Z and T: the factors of the cooling solver's T and
T(1 + 1e-5) calls differ only to first order. It is floored at D_FLOOR.

The Gaussian is the approximation. Where the true P(Z) falls off much faster than a Gaussian on
the attractive side (small positive grains in cold gas, with c of tens), a strongly repulsive
impactor's D, which that tail sets, is overestimated.

RAMSES builds the weights by recurrence and stops once every higher charge is at the floor;
both change the result only by rounding (the two agree to ~1e-13).
"""
import math

import numpy as np
from scipy.special import erfc, erfcx, ndtr

KB = 1.3806490e-16                  # RAMSES amr/constants.f90 [erg K^-1]
E2 = 2.3070775e-19                  # e^2 [statC^2]
PI_RAMSES = 6.2831853 / 2.0         # RAMSES pi (twopi/2), in the neutral-grain factor

SIG_LO, SIG_HI = 3.5, 4.5           # discrete sum below SIG_LO, Gaussian integrals above SIG_HI
K_WIN = 6.0                         # half-width of the summed window [max(sigma, 1/2)]
D_FLOOR = 1e-10


def focusing_factor(Z, j, T, a_cm):
    """WD99 factor of grains of charge Z (array) for an impactor of charge j."""
    Z = np.asarray(Z, float)
    if j == 0:
        return np.ones_like(Z)
    C1 = E2 / (KB * T * a_cm)
    C = j * C1
    out = np.where(Z * j > 0, np.exp(np.minimum(-Z * C, 700.0)), 1.0 - Z * C)
    return np.where(Z == 0, 1.0 + abs(j) * math.sqrt(PI_RAMSES * C1 / 2.0), out)


def coulomb_factors_exact(Z, P, T, a_cm, jmax):
    """D(-1..jmax) over the charge distribution P on the charges Z."""
    P = np.asarray(P, float)
    return np.array([P @ focusing_factor(Z, j, T, a_cm) for j in range(-1, jmax + 1)])


def _smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _discrete(mu, sig, C1, js):
    sw = max(sig, 0.5)
    Z = np.arange(math.floor(mu - K_WIN * sw), math.ceil(mu + K_WIN * sw) + 1, dtype=float)
    x = np.abs(Z - mu) / sw
    taper = np.where(x <= K_WIN - 1.0, 1.0,
                     np.where(x >= K_WIN, 0.0, np.cos(0.5 * np.pi * (x - K_WIN + 1.0)) ** 2))
    q = (Z - mu) ** 2
    w = taper * np.exp(-0.5 * (q - q.min()) / (sig * sig))
    p = w / w.sum()
    neg, pos = Z < 0, Z > 0
    p0 = p[Z == 0].sum()
    out = []
    for j in js:
        if j == 0:
            out.append(1.0)
            continue
        c = abs(j) * C1
        B0 = 1.0 + abs(j) * math.sqrt(PI_RAMSES * C1 / 2.0)
        att_side, rep_side = (neg, pos) if j > 0 else (pos, neg)
        att = np.sum(p[att_side] * (1.0 + np.abs(Z[att_side]) * c))
        rep = np.sum(p[rep_side] * np.exp(np.maximum(-np.abs(Z[rep_side]) * c, -745.0)))
        out.append(att + p0 * B0 + rep)
    return np.array(out)


def _gaussian(mu, sig, C1, js):
    out = []
    for j in js:
        if j == 0:
            out.append(1.0)
            continue
        m = mu if j > 0 else -mu        # a negative impactor on Z is a positive one on -Z
        c = abs(j) * C1
        B0 = 1.0 + abs(j) * math.sqrt(PI_RAMSES * C1 / 2.0)
        tA, tR = (-0.5 - m) / sig, (0.5 - m) / sig
        PA = ndtr(tA)
        att = PA - c * (m * PA - sig * math.exp(-0.5 * tA * tA) / math.sqrt(2.0 * math.pi))
        neu = (ndtr(tR) - PA) * B0
        x = tR + c * sig
        if x > 0.0:                     # exp(-c m + c^2 sig^2/2) Q(x), Q(x) = erfc(x/sqrt 2)/2
            rep = 0.5 * math.exp(-0.5 * c - 0.5 * tR * tR) * erfcx(x / math.sqrt(2.0))
        else:
            rep = math.exp(-c * m + 0.5 * c * c * sig * sig) * 0.5 * erfc(x / math.sqrt(2.0))
        kappa = 1.0 if c < 1e-4 else (c * math.exp(-0.5 * c) if c > 1e2 else (0.5 * c) / math.sinh(0.5 * c))
        out.append(att + neu + kappa * rep)
    return np.array(out)


def coulomb_factors(mu, sigma, T, a_cm, jmax, floor=D_FLOOR):
    """D(-1..jmax) of a grain of radius a_cm [cm] at gas temperature T [K] whose charge
    distribution is a Gaussian of mean mu and width sigma on the integer charges."""
    C1 = E2 / (KB * T * a_cm)
    js = list(range(-1, jmax + 1))
    sig = max(sigma, 1e-3)
    w = _smoothstep((sig - SIG_LO) / (SIG_HI - SIG_LO))
    D = np.zeros(len(js))
    if w < 1.0:
        D += (1.0 - w) * _discrete(mu, sig, C1, js)
    if w > 0.0:
        D += w * _gaussian(mu, sig, C1, js)
    return np.maximum(D, floor)
