"""Shifts of the EA and IPs of dehydrogenated and super-hydrogenated PAHs from Delta-SCF DFT (2026-11-10).

``LADDER`` holds, for BaselineModel's three parents (C24H12, C54H18, C96H24), the shift of each threshold of the state with
N_H0 + dN_H hydrogens over the parent's, for dN_H = -4 ... +2: (EA, IP0, IP1) in eV.

How they were computed (``diagnostics/pahchem/dft_ladders``; results/pahchem/dft_ladders_2026-11-08):

- adiabatic Delta-SCF, with each charge at its own optimised geometry, and the lowest spin kept;
- B3LYP/6-31G* for the IPs; the EAs at 6-31+G*, as single points at the 6-31G* geometries;
- the H removed from, or added to, the end of a duo edge.

What they show:

- **Dehydrogenation.** The IPs rise by 0.03-0.08 eV per missing H, less than Andrews+16's 0.1. The EA jumps instead of
  rising smoothly: it goes up to max(parent EA, sigma-anion EA ~1.7-2.1 eV), by +1.2 / +0.5 / +0.06 eV at -1 H. It
  alternates with the parity of N_H.
- **Super-hydrogenation.** An odd extra H (a CH2 group, a pi radical) lowers IP0 by 1.29 / 0.68 / 0.27 eV and raises the
  EA by 0.95 / 0.48 / 0.13 eV. An even count is close to the parent.

Not computed, and filled in (all labelled assumptions):

- **C96's EA at -2 ... -4 (estimated).** The anion SCFs did not converge in time. The value is max(EA_parent,
  EA_sigma) - EA_parent, with EA_sigma = 2.56 - 4.38/sqrt(N_C), the -1 H anions' law (C24/C54/C96).
- **C96's IP1 at -3 (interpolated).** Midway between its -2 and -4 values.
- **dN_H < -4 (extension).** The IPs grow at the -4 point's per-H slope, shift(-4) x m / 4 for m missing H. The EA
  holds its -3 value for odd m and its -4 value for even m, since the sigma-anion EA saturates. DFT fails for the bare
  skeletons: they are polyradicals, with <S^2> 4-9.
- **dN_H > +2 (extension).** Odd counts take +1's shifts and even counts +2's, since an odd count is again a pi radical.
- **Z >= 2 thresholds** (C54's and C96's higher IPs) take IP1's shift.

``law_shift`` carries the ladder to any molecule (the population's "joint" chain), in two forms (``LAWS``):

- **Lower configuration.** The state's threshold is A + B/sqrt(N_C), the conducting-disc form, unless the parent's own
  is beyond it: EA = max(EA_parent, ...) for the dehydrogenated and odd super-H anions, IP0 = min(IP_parent, ...) for
  the odd super-H neutral.
- **Shift.** The others are shifts a + b/sqrt(N_C) over the parent's.

The fits are least squares to C24/C54/C96, or exact through C24/C54 where C96 is missing (-2 ... -4 EA, -3 IP1).

A transfer test on four PAHdb v4.00 molecules of other shapes and edges (UIDs 1969, 2226, 4 and 105; one site per edge
class; results/pahchem/dft_transfer_2026-11-10) tested the -1 H EA and the +1 H IP0 and EA forms, fixed beforehand. They
reproduced the DFT shifts to 0.15 eV rms, within 0.34 eV; a shift depending on N_C alone gave 0.53 / 1.14. The edge sites
of one molecule scatter by about +-0.2 eV. The other forms are the ladder's only.

Outside the ladder's sizes (assumptions):

- **N_C < 24.** C24's shifts. Against the literature's small radicals, this gives 1-naphthyl's EA step +1.19 eV (measured
  +1.6) and 1-hydronaphthyl's IP -1.29 eV (measured -1.57). The lower-configuration forms extrapolated there would put
  the latter at -2.3 eV.
- **N_C > 96.** The shift forms hold C96's values. The lower-configuration forms go on, and the parent's threshold
  takes over as N_C grows.
"""

from __future__ import annotations

from math import sqrt

#: the 6-31+G* EA of a parent over PAHdb's 6-31G* one [eV], which the lower-configuration EAs are compared at: the
#: ladder parents' 0.28-0.37; the transfer test's four parents within 0.09 of PAHdb + 0.33
DIFFUSE_EA_EV = 0.33

#: the ladder parents: (N_C, IP0, IP1, EA at 6-31+G*) [eV]
LADDER_PARENTS = {"C24": (24, 6.78, 10.58, 0.48), "C54": (54, 5.91, 8.63, 1.45), "C96": (96, 5.41, 7.52, 2.05)}

#: {(dN_H, z): (form, A, B)}, z = -1 (EA), 0 (IP0), 1 (IP1 and above): "max" / "min", the state's threshold
#: max / min(the parent's, A + B/sqrt(N_C)); "shift", the shift A + B/sqrt(N_C). The -1 H EA and +1 H forms are the
#: transfer test's.
LAWS = {
    (-1, -1): ("max", 2.56, -4.38), (-1, 0): ("shift", 0.025, -0.021), (-1, 1): ("shift", 0.0, 0.385),
    (-2, -1): ("max", 2.057, -3.417), (-2, 0): ("shift", 0.009, 0.748), (-2, 1): ("shift", -0.053, 1.362),
    (-3, -1): ("max", 2.487, -3.276), (-3, 0): ("shift", -0.010, 0.999), (-3, 1): ("shift", -0.070, 1.779),
    (-4, -1): ("max", 2.060, -2.551), (-4, 0): ("shift", -0.052, 1.736), (-4, 1): ("shift", -0.033, 2.534),
    (1, -1): ("max", 2.94, -7.40), (1, 0): ("min", 4.79, 3.42), (1, 1): ("shift", -0.056, -0.291),
    (2, -1): ("shift", -0.204, 1.507), (2, 0): ("shift", 0.327, -3.731), (2, 1): ("shift", 0.409, -3.530),
}
#: the sizes the laws were fitted over
LAW_NC = (24, 96)

#: {parent: {dN_H: (dEA, dIP0, dIP1)}} [eV]
LADDER = {
    "C24": {-4: (1.06, 0.30, 0.48), -3: (1.34, 0.19, 0.29), -2: (0.88, 0.15, 0.24), -1: (1.19, 0.02, 0.08),
            0: (0.0, 0.0, 0.0), 1: (0.95, -1.29, -0.11), 2: (0.11, -0.44, -0.32)},
    "C54": {-4: (0.27, 0.18, 0.31), -3: (0.60, 0.13, 0.17), -2: (0.15, 0.14, 0.10), -1: (0.51, 0.02, 0.03),
            0: (0.0, 0.0, 0.0), 1: (0.48, -0.68, -0.10), 2: (-0.01, -0.16, -0.05)},
    # EA at -2 ... -4 estimated; IP1 at -3 interpolated
    "C96": {-4: (0.06, 0.13, 0.22), -3: (0.06, 0.09, 0.17), -2: (0.06, 0.06, 0.11), -1: (0.06, 0.02, 0.05),
            0: (0.0, 0.0, 0.0), 1: (0.13, -0.27, -0.08), 2: (-0.04, -0.07, 0.03)},
}


def _extend(dnh, z):
    """(dN_H, z, factor): the computed entry that state dN_H's threshold z takes, and the factor on its shift (the
    module's extensions outside dN_H = -4 ... +2; z >= 2 as IP1)."""
    z = min(z, 1)
    if dnh > 2:                                           # odd: as +1, even: as +2
        return 2 - dnh % 2, z, 1.0
    if dnh < -4:
        if z == -1:                                       # the sigma-anion EA saturates, by parity
            return (-3 if dnh % 2 else -4), z, 1.0
        return -4, z, -dnh / 4.0                          # the -4 point's per-H slope
    return dnh, z, 1.0


def ladder_shift(parent, dnh, z):
    """The shift [eV] of threshold z (Z -> Z+1; z = -1 is the EA) of ``parent`` with N_H0 + ``dnh`` hydrogens, over
    the parent's, with the module's extensions outside dN_H = -4 ... +2."""
    dnh, z, factor = _extend(dnh, z)
    return factor * LADDER[parent][dnh][z + 1]


def law_shift(nc, dnh, z, ip0, ea):
    """The shift [eV] of threshold z of a molecule of ``nc`` carbons with N_H0 + ``dnh`` hydrogens, over its own: ``LAWS``,
    with the module's extensions. ``ip0``, ``ea``: the intact molecule's IP0 and EA (PAHdb's, 6-31G*; ``ea`` is needed
    only for z = -1)."""
    if dnh == 0:
        return 0.0
    dnh, z, factor = _extend(dnh, z)
    if nc < LAW_NC[0]:
        return factor * LADDER["C24"][dnh][z + 1]
    form, a, b = LAWS[(dnh, z)]
    if form == "shift":
        return factor * (a + b / sqrt(min(nc, LAW_NC[1])))
    x = a + b / sqrt(nc)
    if form == "max":
        parent = ea + DIFFUSE_EA_EV
        return max(parent, x) - parent
    return min(ip0, x) - ip0
