"""Grain charging and photoelectric heating from radiation-transport photon groups.

RAMSES-RT carries, per cell, the photon density N_g of each group g and, per
run, one mean photon energy egy_g per group. Photoemission depends on the grain
potential U and the photon energy only, so everything that involves photons
can be tabulated once per dust bin and group as a function of U:

    K_g(U) = < C_abs Y_tot + sigma_pdt >_g               [cm^2]
    H_g(U) = < C_abs sum_k Y_k <E>_k + sigma_pdt E_pd >_g [cm^2 eV]

(WDB06 yields, ``wdb06_yields``), averaged over the group with a blackbody
photon spectrum n_E ~ E^2 / (exp(E/kT_s) - 1). Each group is tabulated for a
family of T_s, i.e. of group mean energies, so a run whose group energies
change (SED updates) just reads off its current egy_g. The photoemission rate
per grain is then  sum_g c_red N_g K_g(U), with no dependence on T or n_e.

The collisional currents (electron and ion capture, DS87 focusing, WD01
sticking) are analytic. The hot-gas terms (gas-electron secondaries and
electron pass-through, Draine & Salpeter 1979) are tabulated in (T, U).

``build_bin_table`` computes a bin's table, ``write_bin_table`` /
``read_bin_table`` store it in the text format RAMSES reads, and
``GroupChargingTable.solve`` is the per-cell algorithm RAMSES implements:
the same file, formulas and iteration, so the two agree to round-off. It
mirrors calima/dust_charging.f90 at RAMSES staging + PR #86 + calima/charge-lg5
7f842bb4 (rtgroups_solve_bin, rtgroups_solve_lg), except RAMSES's predictions
across cooling substeps (rtgroups_predict), a cost shortcut pyCALIMA does not
replay. On the 22-bin bench (858,000 cell-bin states, studies
pycalima_port_20261118) every output agrees with the Fortran's to 1e-10 or
better and every branch flag state by state.

Per-cell algorithm (``GroupChargingTable.solve``)
-------------------------------------------------
With Up(Z) = photoemission + ion capture + gas-electron secondaries and
Down(Z) = electron capture, detailed balance gives f(Z+1)/f(Z) = Up(Z)/Down(Z+1);
g(Z) = ln Up(Z) - ln Down(Z+1) decreases with Z. Kernels in U by PCHIP (default).

1. The root's bracket on the integer lattice, g(k*) > 0 >= g(k*+1), walked at most
   ``LAT_NWALK`` charges from the guess (the warm result's Z*, else ``Z_guess``),
   else from the continuous root (Illinois on a bracket grown from the guess in
   steps 1, 2, 4, ...; ``_root``).
2. One rule on the warm and the cold path (``_distribution``): P(Z) is discrete iff
   the width of the integer slope at the root (or else of the integer chord,
   ``_chord``) is below ``SIGMA_DISCRETE`` and the window it starts, grown or
   trimmed at each end to one charge past where P < ``DISCRETE_TAIL`` of its
   maximum (``_window``) and not capped at ``MAX_DISCRETE_STATES``, has a width
   below ``SIGMA_DISCRETE``; Z* is then the root of g interpolated between the
   integers. A warm discrete window is re-formed as a cold one, so the result does
   not depend on the start (except the chord of a wide P(Z) rooted at an end of the
   charge range). Otherwise P(Z) is wide: Z* by Illinois within the lattice bracket,
   its width from the pair Z* -+ 1/2 or the chord.
3. A wide P(Z) (``wide_moments``, default): <Z> and sigma from a model of g over its
   width (``_wide_moments``: g at m-1, m, m -+ 1.5 sigma, PCHIP, the lattice sum
   integrated, 8-point Gauss-Hermite moments), else the Gaussian of mean Z* + 1/2;
   Gamma and Lambda taken at that <Z>.
4. Heating Gamma = sum_g c N_g H_g(U); recombination cooling from DS87
   Lambda-tilde, scaled for electron pass-through, less the energy the
   gas-excited secondaries return; autoionisation cooling at Zmin.
5. Optionally (``dlnT``) the d/d ln T of <Z>, sigma_Z, Gamma and Lambda: analytic
   on a discrete window; for a wide P(Z) from one more set of rates at T (1 + dlnT)
   on the same pair (and chord), to serve the cooling solver's T (1 +- dlnT) calls.
6. Grain-assisted recombination alpha_i = sum_Z P(Z) pi a^2 v_i J~ Theta_i
   (``recomb``): exact for a discrete P(Z), over its window and the charges below it
   while their terms can matter (``_recombination_discrete``); for a wide one the
   Gaussian estimate (``_recombination_gauss``, recomb="gauss"), or the segmented
   Chebyshev reconstruction of ln P (``_recombination_fit``, recomb="fit"), reused by
   warm solves while the cell drifts little; ``solve_cell`` fits only the wide bins
   that carry the cell's recombination, as RAMSES's importance gate.
7. Optionally (``coulomb``) the Coulomb factors D (impactor charges -1, 1, 2, 3):
   the Gaussian factors of <Z>, sigma (``coulomb_focusing``) times, for a discrete
   P(Z), their exact-P(Z) correction from the window (``_coulomb_ratio``).
8. A wide positive P(Z) whose tail toward Z = 0 carries alpha or D (where the
   kink of the Coulomb factors bends g): both from that tail (``_tail_sums``: g
   modelled at the kink charges, summed from Z = -6 up), reported by
   ``alpha_tail`` and ``D_abs``.

``lattice=True`` (RAMSES rtg_lattice) replaces 1-3 and 7 by the lattice closure LG5
(``_solve_lattice``): the rates at the 5 integer charges around the mode (7 where
the central width is 0.6-3), P(Z) exact on that window with geometric tails, or the
central pair's Gaussian (with ``_wide_moments``) for a wide P(Z); D over that P(Z);
the tail as in 8. 2-4x cheaper than the default solve, within the documented
charging uncertainties of the exact P(Z).

The UV background (RAMSES rtz_UV_background_G0, carried outside the RT
groups) enters through two more per-bin kernels, the photoemission and heating
rates per grain in the Mathis et al. (1983) ISRF (below 13.6 eV) normalised to
G0 = 1 Habing flux: photo += G0_bg * K_bg(U), heating += G0_bg * H_bg(U).

Uniform-G0 table: where a cell has no local RT photons (tests, runs with only
a uniform background), the charge depends on (T, n_e) alone. ``uniform_table``
solves once per bin on a (log T, log n_e) grid at the run's G0 (ions: H+ with
n(H+) = n_e) and ``uniform_lookup`` interpolates bilinearly; RAMSES builds the
same table at start-up (and again only if G0 changes).

Not included: ion pass-through (it matters only above ~3e7 K).
"""

from pathlib import Path

import numpy as np

from pycalima.models.dust_charge import wdb06_yields as wy
from pycalima.models.dust_charge.shared_physics import (
    E_STATC,
    EV2ERG,
    KB_CGS,
    ME_CGS,
    DS87_J_function_vec,
    electron_affinity_graphite_vec,
    electron_affinity_silicate_vec,
    ionisation_potential_valence_vec,
    most_negative_allowed_charge_graphite,
    most_negative_allowed_charge_silicate,
    most_positive_allowed_charge,
    photodetachment_cross_section_vec,
)

FORMAT_VERSION = 3
FORMAT_VERSION_STICKING = 4           # cm20 tables: format 3 plus the sticking coefficient s_e(U)
C_CGS = 2.99792458e10
HABING_FLUX = 1.6e-3   # erg cm^-2 s^-1 in 6-13.6 eV: G0 = 1 (RAMSES advected_G0)
E2_EV_CM = E_STATC ** 2 / EV2ERG      # e^2 in eV cm: U [V] = Z * E2_EV_CM / a
MATERIAL_CODE = {"graphite": 1, "silicate": 2, "cm20": 3, "ac": 4, "ach": 5}
SIGMA_DISCRETE = 3.0                  # below this, P(Z) is built exactly
MAX_DISCRETE_STATES = 121
DISCRETE_TAIL = 1e-5                  # a discrete P(Z) ends one charge past where P < this x max
SIGMA_CHORD_ITER = 3                  # at most this many chords for the width of a wide P(Z)
LAT_NWALK = 4                         # the integer bracket of the root is walked at most this far from the guess
NBUF = 3 * MAX_DISCRETE_STATES + 4    # RAMSES's window buffers: the charges b_base + 1 .. b_base + NBUF
# The exact-P(Z) correction of the Gaussian Coulomb factors of a discrete P(Z) (_window_ratio): P continued outward
# from the window to RATIO_LN_CUT below its maximum, or until the geometric bound on the rest is below RATIO_EPS
RATIO_LN_CUT = -690.7755278982137     # ln 1e-300
RATIO_EPS, RATIO_CLAMP = 1e-6, 0.5e-10
COULOMB_DCHARGE = (-1, 1, 2, 3)       # impactor charges j of the Coulomb factors D (RAMSES ISRF_DCHARGE)
COULOMB_JMAX = 3
HUGE = np.finfo(float).max
# The tail of a positive P(Z) toward Z = 0 (_tail_sums): used where its alpha or D(+1) integrand at Z = 1, 0, -2 comes
# within e^-TAIL_CUT of its value on the window; its sums stop past their peak at e^-TAIL_STOP of their totals
TAIL_CUT, TAIL_STOP = 25.0, 20.0
# LG5 (solve(lattice=True), RAMSES rtg_lattice): its window is 5 charges, 7 where the central width is in
# [LG_SIGMA7, SIGMA_DISCRETE); the tails of a narrow P(Z) run down to LG_EPS of the peak, at most LG_TAIL charges
LG_SIGMA7, LG_EPS, LG_TAIL = 0.6, 1e-11, 96
# _wide_moments: 8-point Gauss-Hermite nodes and weights (as RAMSES)
GH_X = np.array([-2.930637420257244, -1.981656756695843, -1.157193712446780, -0.381186990207322,
                 0.381186990207322, 1.157193712446780, 1.981656756695843, 2.930637420257244])
GH_W = np.array([1.996040722113676e-4, 1.707798300741348e-2, 2.078023258148919e-1, 6.611470125582413e-1,
                 6.611470125582413e-1, 2.078023258148919e-1, 1.707798300741348e-2, 1.996040722113676e-4])
# Grain-assisted recombination of a wide P(Z) (sigma >= SIGMA_DISCRETE), recomb="fit":
# see GroupChargingTable._recombination_fit. Settings validated per bin against
# the exact sum over the integer charges (PCHIP kernels): <= 5% typically, < 10% worst,
# for rates above 1e-6 of the geometric rate pi a^2 v; below that they are not resolved
# (for Milky Way dust, < ~1e-21 cm^3 s^-1 per H: negligible against gas-phase recombination).
RECOMB_FIT_N = 5          # max Chebyshev nodes per fitted segment of g(Z)
RECOMB_FIT_NA = 5         # max Chebyshev nodes per segment of ln A(Z)
RECOMB_PER_NODE = 5.0     # charges per node (fewer nodes on short segments, min 3)
RECOMB_KINK = 2           # g evaluated exactly on the integers of [-K, K]
RECOMB_DOWN, RECOMB_UP = 8.0, 5.0     # P domain Z* - 8 sigma .. Z* + 5 sigma, extended below as needed
RECOMB_REACH = 8.0        # extend down to ion thresholds within this many sigma of Z*
RECOMB_CUT = 25.0         # stop extending once P and each ion's P A Theta are e^-CUT below their peaks
RECOMB_FLOOR = np.log(1e-8)           # ... or alpha is below 1e-8 of the geometric rate
RECOMB_EPS = 0.08         # summation step: |(ln P A)''| h^2 / 8 < EPS, h <= sigma
RECOMB_RESOLVED = 1e-6   # alpha / geometric rate above which the fit and its reuse are accurate
# Warm solves reuse a wide fit, carried by the change of its Gaussian estimate (control variate),
# while ln T, ln ne, ln N_g, ln n_ion, ln G0 stay within RECOMB_REUSE_TOL of the fitted state and the
# Gaussian estimate of every resolved alpha has moved by less than RECOMB_REUSE_EPS (in ln).
# Validated on RAMSES Stromgren call chains: reuse error 99% < 0.3%, worst 6%.
RECOMB_REUSE_TOL = 1e-2
RECOMB_REUSE_EPS = 0.02
# solve_cell fits the recombination of a wide P(Z) only for bins that carry more than this
# fraction of the cell's grain-assisted recombination of some ion (from the Gaussian estimates);
# the others keep their Gaussian estimate.
RECOMB_IMPORTANT = 0.1
MATHIS_BREAKS = (5.04, 9.26, 11.2, 13.6)  # eV: Mathis et al. (1983) ISRF piecewise breaks and cutoff

# default grids
N_U, U_MIN, U_MAX, U_SCALE = 401, -1000.0, 1000.0, 0.5   # volts, asinh spacing
TS_GRID = np.geomspace(3e3, 3e6, 31)                     # group-shape temperatures [K]
THOT_GRID = np.geomspace(1e3, 1e9, 61)                   # hot-gas table temperatures [K]
N_E_PER_GROUP = 64

# uniform-G0 table grid (RAMSES uses the same nodes)
UNIF_LT = np.linspace(1.0, 9.0, 81)       # log10 T [K]
UNIF_LNE = np.linspace(-8.0, 8.0, 97)     # log10 n_e [cm^-3]
M_HION = 1.67262192e-24
M_U = 1.66053906660e-24

# Grain-assisted recombination X+ + grain(Z) -> X + grain(Z+1) of the singly
# charged ions of the RTZ elements: (element, mass [amu], IP of the neutral [eV]).
RECOMB_IONS = (("H", 1.00794, 13.598), ("He", 4.002602, 24.587), ("C", 12.0107, 11.260),
               ("N", 14.0067, 14.534), ("O", 15.9994, 13.618), ("Ne", 20.1797, 21.565),
               ("Mg", 24.305, 7.646), ("Si", 28.0855, 8.152), ("S", 32.065, 10.360),
               ("Fe", 55.845, 7.902))

# the 8 groups of the CALIMA RTZ runs (eV)
DEFAULT_GROUP_L0 = (0.1, 1.0, 5.6, 11.2, 13.6, 15.2, 24.59, 54.42)
DEFAULT_GROUP_L1 = (1.0, 5.6, 11.2, 13.6, 15.2, 24.59, 54.42, 500.0)


def u_grid(n=N_U, u_min=U_MIN, u_max=U_MAX, u_scale=U_SCALE):
    """Potential grid [V], uniform in asinh(U / u_scale)."""
    x = np.linspace(np.arcsinh(u_min / u_scale), np.arcsinh(u_max / u_scale), n)
    return u_scale * np.sinh(x)


def _photon_weights(E, Ts):
    """Blackbody photon-number spectrum n_E ~ E^2/(exp(E/kT)-1) on nodes E, scaled to max 1."""
    x = E / (KB_CGS * Ts / EV2ERG)
    ln_n = 2.0 * np.log(E) - x - np.log(-np.expm1(-x))
    return np.exp(ln_n - ln_n.max())


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def _group_kernels(E, Zs, a, material):
    """K(E, Z) [cm^2] and H(E, Z) [cm^2 eV] on a group's energy nodes."""
    la = wy.dielectric_attenuation_length(E, material)
    C_abs = np.pi * a ** 2 * wy.absorption_efficiency(E, a, material)
    Y = wy.total_yield(E, Zs, a, material, la_wd01=la)
    YE = wy.yield_energy_product(E, Zs, a, material, la_wd01=la)
    K = C_abs[:, None] * Y
    H = C_abs[:, None] * YE
    # photodetachment of excess electrons (WD01 eqs. 18-20, van Hoof E_min)
    for j, Z in enumerate(Zs):
        if Z >= 0:
            continue
        E_pdt = float(wy.photodetachment_energy(Z, a, material))
        sig = photodetachment_cross_section_vec(E, E_pdt, Z)
        E_min = float(wy.min_energy_vH04(Z, a))
        K[:, j] += sig
        H[:, j] += sig * np.maximum(E - E_pdt + E_min, 0.0)
    return K, H


def _charge_bounds(material, a, U, L1):
    """(W, Zmin, Zmax) of a bin: the material's charge limits within the U grid."""
    Zmin = float({"graphite": most_negative_allowed_charge_graphite, "cm20": wy.most_negative_charge_cm20,
                  "ac": most_negative_allowed_charge_graphite, "ach": most_negative_allowed_charge_graphite,
                  "silicate": most_negative_allowed_charge_silicate}.get(
        material, most_negative_allowed_charge_graphite if wy.is_acx(material) else None)(a))
    W = wy.MATERIALS[material]["W"]
    Zmax = float(most_positive_allowed_charge(a, W, float(np.max(L1))))
    Zmax = min(Zmax, np.floor(U[-1] * a / E2_EV_CM))
    Zmin = max(Zmin, np.ceil(U[0] * a / E2_EV_CM))
    return W, Zmin, Zmax


#: the tabulated (photon and hot-gas) arrays of a bin table, as opposed to its analytic limits
TABULATED_KEYS = ("egy", "K", "H", "K_bg", "H_bg", "delta", "P_tr", "f_cool", "E_bar")


def interpolate_bin_tables(t0, t1, w, material):
    """A bin table for ``material`` whose tabulated arrays are linear in ``w`` between
    those of ``t0`` (w = 0) and ``t1`` (w = 1), tables of one radius on the same groups
    and grids; the analytic parts (W, Zmin, Zmax, recombination thresholds, and the IP and
    EA the solver evaluates) are ``material``'s own. This is option 1 of the plan for
    tracked hydrogen: two end-member tables per bin, against direct tables of the
    intermediate material (option 2)."""
    for k in ("a", "U", "L0", "L1", "log10_Ts", "log10_Thot"):
        if not np.allclose(t0[k], t1[k]):
            raise ValueError(f"tables differ in {k!r}")
    out = {k: v for k, v in t0.items() if k not in TABULATED_KEYS}
    for k in TABULATED_KEYS:
        out[k] = (1.0 - w) * np.asarray(t0[k]) + w * np.asarray(t1[k])
    W, Zmin, Zmax = _charge_bounds(material, t0["a"], np.asarray(t0["U"]), np.asarray(t0["L1"]))
    out.update(material=material, W=float(W), Zmin=Zmin, Zmax=Zmax,
               recomb_Zth=recombination_thresholds(material, t0["a"], W, Zmin, Zmax))
    return out


def build_bin_table(material, a, group_L0=DEFAULT_GROUP_L0, group_L1=DEFAULT_GROUP_L1,
                    U=None, Ts=TS_GRID, T_hot=THOT_GRID, n_E=N_E_PER_GROUP, label=""):
    """Table for one dust bin (dict of arrays; see ``write_bin_table``)."""
    material = wy._label(material)
    U = u_grid() if U is None else np.asarray(U, dtype=float)
    L0 = np.asarray(group_L0, dtype=float)
    L1 = np.asarray(group_L1, dtype=float)
    if np.any(L1 <= L0):
        raise ValueError("every group needs a finite upper edge above its lower one")
    Zs = U * a / E2_EV_CM
    W, Zmin, Zmax = _charge_bounds(material, a, U, L1)

    nG, nU, nTs = L0.size, U.size, np.size(Ts)
    egy = np.zeros((nG, nTs))
    K = np.zeros((nG, nTs, nU))
    H = np.zeros((nG, nTs, nU))
    for g in range(nG):
        E = np.geomspace(L0[g], L1[g], n_E)
        Kg, Hg = _group_kernels(E, Zs, a, material)
        for it, T in enumerate(Ts):
            w = _photon_weights(E, T)
            norm = np.trapezoid(w, E)
            egy[g, it] = np.trapezoid(w * E, E) / norm
            K[g, it] = np.trapezoid(w[:, None] * Kg, E, axis=0) / norm
            H[g, it] = np.trapezoid(w[:, None] * Hg, E, axis=0) / norm

    # UV background: Mathis ISRF at G0 = 1 (Habing flux in 6-13.6 eV), from 1 eV; for the
    # amorphous carbons from 0.1 eV (the bottom of the RT groups): their electron affinity
    # falls below 1 eV (below 0 for sub-nm grains), so the optical/IR ISRF photodetaches them
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field
    Eb = np.geomspace(0.1 if material in wy.KIMURA_MATERIALS else 1.0, 13.6, 400)
    uE = np.array([Mathis83_radiation_field(e) for e in Eb])
    band = (Eb >= 6.0)
    uE *= HABING_FLUX / (C_CGS * np.trapezoid(uE[band], Eb[band]))
    Fb = C_CGS * uE / (Eb * EV2ERG)                 # photons cm^-2 s^-1 eV^-1
    Kb, Hb = _group_kernels(Eb, Zs, a, material)
    K_bg = np.trapezoid(Fb[:, None] * Kb, Eb, axis=0)   # s^-1 per grain
    H_bg = np.trapezoid(Fb[:, None] * Hb, Eb, axis=0)   # eV s^-1 per grain

    nTh = np.size(T_hot)
    delta = np.zeros((nTh, nU))
    P_tr = np.zeros((nTh, nU))
    f_cool = np.zeros((nTh, nU))
    E_bar = np.zeros((nTh, nU))
    for it, T in enumerate(T_hot):
        delta[it] = wy.gas_secondary_yield(Zs, a, T, material)
        P_tr[it], f_cool[it] = wy.electron_transmission(Zs, a, T, material)
        E_bar[it] = wy.gas_secondary_mean_energy(Zs, a, T, material)

    tab = dict(version=FORMAT_VERSION, label=label, material=material, a=float(a), W=float(W),
               Zmin=Zmin, Zmax=Zmax, L0=L0, L1=L1, U=U, log10_Ts=np.log10(Ts), egy=egy, K=K, H=H,
               log10_Thot=np.log10(T_hot), delta=delta, P_tr=P_tr, f_cool=f_cool, E_bar=E_bar,
               K_bg=K_bg, H_bg=H_bg, recomb_Zth=recombination_thresholds(material, a, W, Zmin, Zmax))
    if material == "cm20":
        tab.update(version=FORMAT_VERSION_STICKING, s_e=wy.cm20_sticking(Zs, a))
    return tab


def grain_ionisation_potential(material, a, W, Z):
    """IP(a, Z) [eV]: energy to remove an electron from a grain of charge Z
    (valence IP for Z >= 0, electron affinity of charge Z + 1 below)."""
    Z = np.asarray(Z, dtype=float)
    if material in wy.KIMURA_MATERIALS:
        return np.where(Z >= 0, wy.ionisation_potential(W, Z, a, material), wy.electron_affinity(Z + 1.0, a, material))
    ea = electron_affinity_graphite_vec if material == "graphite" else electron_affinity_silicate_vec
    return np.where(Z >= 0, ionisation_potential_valence_vec(W, Z, a), ea(Z + 1.0, a))


def last_charge_with_ip_below(material, a, W, Zmin, Zmax, E):
    """Largest integer Z in [Zmin, Zmax] with IP(a, Z) <= E [eV], or None if there is none.
    IP(a, Z) increases with Z, so this is a bisection: an array over every allowed charge
    would hold ~4e8 values for a 1 mm grain."""
    ip = lambda z: float(grain_ionisation_potential(material, a, W, float(z)))   # noqa: E731
    lo, hi = int(round(Zmin)), int(round(Zmax))
    if ip(lo) > E:
        return None
    if ip(hi) <= E:
        return float(hi)
    while hi - lo > 1:                          # ip(lo) <= E < ip(hi)
        mid = (lo + hi) // 2
        if ip(mid) <= E:
            lo = mid
        else:
            hi = mid
    return float(lo)


def recombination_thresholds(material, a, W, Zmin, Zmax):
    """Largest grain charge Z at which X+ can take an electron from the grain,
    IP(a, Z) <= IP(X) (Weingartner & Draine 2001b case A), per RECOMB_IONS;
    Zmin - 1 where no charge allows it."""
    out = []
    for _, _, ip in RECOMB_IONS:
        z = last_charge_with_ip_below(material, a, W, Zmin, Zmax, ip)
        out.append(Zmin - 1.0 if z is None else z)
    return np.array(out)


# ---------------------------------------------------------------------------
# File format (read by RAMSES, calima/dust_charging_rtgroups.f90)
# ---------------------------------------------------------------------------
#   3 comment lines starting with '#'
#   version material_code a_cm W_eV Zmin Zmax
#   nGroups nU nTs nThot
#   L0(1:nG) / L1(1:nG) / U(1:nU) [V] / log10 Ts(1:nTs)
#   egy(1:nTs) for each group
#   K(1:nU) for each group, then each Ts     [cm^2]
#   H(1:nU) for each group, then each Ts     [cm^2 eV]
#   log10 T_hot(1:nThot)
#   delta(1:nU), P_tr(1:nU), f_cool(1:nU), E_bar(1:nU) [eV], each for every T_hot
#   K_bg(1:nU) [s^-1], H_bg(1:nU) [eV s^-1]: Mathis ISRF at G0 = 1 (format >= 2)
#   recomb_Zth(1:nIons): case-A charge thresholds for RECOMB_IONS (format >= 3)
#   s_e(1:nU): electron sticking coefficient at the U nodes (format 4, material 3 = cm20 only)
# One record per line, values separated by spaces, 12 significant digits.

def _line(v):
    return " ".join(f"{x:.12e}" for x in np.atleast_1d(v)) + "\n"


def write_bin_table(tab, path):
    path = Path(path)
    with open(path, "w") as fh:
        fh.write(f"# pyCALIMA RT-group grain charging table (WDB06 yields), format {tab['version']}\n")
        fh.write(f"# bin {tab['label']}: {tab['material']}, a = {tab['a']:.6e} cm\n")
        fh.write("# see pycalima.models.dust_charge.rt_group_charging for the layout\n")
        fh.write(f"{tab['version']} {MATERIAL_CODE[tab['material']]} {tab['a']:.12e} {tab['W']:.12e} "
                 f"{tab['Zmin']:.1f} {tab['Zmax']:.1f}\n")
        nG, nTs, nU = tab["K"].shape
        fh.write(f"{nG} {nU} {nTs} {tab['log10_Thot'].size}\n")
        for key in ("L0", "L1", "U", "log10_Ts"):
            fh.write(_line(tab[key]))
        for g in range(nG):
            fh.write(_line(tab["egy"][g]))
        for key in ("K", "H"):
            for g in range(nG):
                for it in range(nTs):
                    fh.write(_line(tab[key][g, it]))
        fh.write(_line(tab["log10_Thot"]))
        for key in ("delta", "P_tr", "f_cool", "E_bar"):
            for it in range(tab["log10_Thot"].size):
                fh.write(_line(tab[key][it]))
        fh.write(_line(tab["K_bg"]))
        fh.write(_line(tab["H_bg"]))
        fh.write(_line(tab["recomb_Zth"]))
        if tab["version"] >= FORMAT_VERSION_STICKING:
            fh.write(_line(tab["s_e"]))
    return path


def read_bin_table(path):
    with open(path) as fh:
        rows = [ln for ln in fh if not ln.startswith("#")]
    it = iter(rows)
    head = next(it).split()
    version, mcode = int(head[0]), int(head[1])
    a, W, Zmin, Zmax = (float(x) for x in head[2:6])
    nG, nU, nTs, nTh = (int(x) for x in next(it).split())
    vec = lambda: np.array(next(it).split(), dtype=float)
    tab = dict(version=version, material={v: k for k, v in MATERIAL_CODE.items()}[mcode], a=a, W=W,
               Zmin=Zmin, Zmax=Zmax, L0=vec(), L1=vec(), U=vec(), log10_Ts=vec())
    tab["egy"] = np.array([vec() for _ in range(nG)])
    for key in ("K", "H"):
        tab[key] = np.array([[vec() for _ in range(nTs)] for _ in range(nG)])
    tab["log10_Thot"] = vec()
    for key in ("delta", "P_tr", "f_cool", "E_bar"):
        tab[key] = np.array([vec() for _ in range(nTh)])
    if version >= 2:
        tab["K_bg"], tab["H_bg"] = vec(), vec()
    else:
        tab["K_bg"] = tab["H_bg"] = np.zeros(nU)
    tab["recomb_Zth"] = vec() if version >= 3 else recombination_thresholds(
        tab["material"], a, W, Zmin, Zmax)
    if version >= FORMAT_VERSION_STICKING:
        tab["s_e"] = vec()
    return tab


# ---------------------------------------------------------------------------
# Per-cell solver (mirrors RAMSES)
# ---------------------------------------------------------------------------

class GroupChargingTable:
    """One bin's table with the per-cell charge solve that RAMSES performs."""

    def __init__(self, tab, egy=None, kernels="pchip"):
        """kernels: interpolation of the photoemission and heating kernels in U,
        "pchip" (monotone cubic, Fritsch-Carlson slopes on the U-node index; the default, as RAMSES: no kinks at
        the U nodes, needed for an accurate recombination fit of wide P(Z)) or "linear" (RAMSES before its PCHIP
        kernels)."""
        if kernels not in ("linear", "pchip"):
            raise ValueError("kernels must be 'linear' or 'pchip'")
        self.t = tab
        self.a = tab["a"]
        self.material = tab["material"]
        self.kernels = kernels
        U = tab["U"]
        self.x0 = np.arcsinh(U[0] / U_SCALE)
        self.dx = (np.arcsinh(U[-1] / U_SCALE) - self.x0) / (U.size - 1)
        # charges at which IP(a, Z) crosses photon-energy edges: g(Z) has kinks there
        def edge(E):
            z = last_charge_with_ip_below(self.material, self.a, tab["W"], tab["Zmin"], tab["Zmax"], E)
            return z if z is not None and tab["Zmin"] < z < tab["Zmax"] else None
        self._edge_groups = [(edge(l0), edge(l1)) for l0, l1 in zip(tab["L0"], tab["L1"])]
        self._edge_bg = [z for z in (edge(E) for E in MATHIS_BREAKS) if z is not None]
        self.set_group_energies(tab["egy"][:, tab["egy"].shape[1] // 2] if egy is None else egy)

    # -- group collapse at the run's group energies (RAMSES: init / SED update)
    def set_group_energies(self, egy):
        """Per-group K_g(U), H_g(U) at mean energies ``egy`` [eV]: interpolate
        linearly in log10 T_s between the two table shapes bracketing egy_g."""
        t = self.t
        nG = t["L0"].size
        self.Kg = np.zeros((nG, t["U"].size))
        self.Hg = np.zeros_like(self.Kg)
        for g in range(nG):
            e = t["egy"][g]
            eg = min(max(float(egy[g]), e[0]), e[-1])
            j = int(np.clip(np.searchsorted(e, eg) - 1, 0, e.size - 2))
            w = 0.0 if e[j + 1] == e[j] else (eg - e[j]) / (e[j + 1] - e[j])
            self.Kg[g] = (1 - w) * t["K"][g, j] + w * t["K"][g, j + 1]
            self.Hg[g] = (1 - w) * t["H"][g, j] + w * t["H"][g, j + 1]
        if self.kernels == "pchip":     # Fritsch-Carlson slopes on the U-node index (once per run in RAMSES)
            self._mK, self._mH = _pchip_slopes(self.Kg), _pchip_slopes(self.Hg)
            self._mKb, self._mHb = _pchip_slopes(t["K_bg"]), _pchip_slopes(t["H_bg"])

    def _kern(self, y, m, i, w):
        """Kernel(s) y (last axis: U nodes) at node index i + w; m: PCHIP slopes."""
        if self.kernels == "linear":
            return (1 - w) * y[..., i] + w * y[..., i + 1]
        w2, w3 = w * w, w * w * w
        return ((2 * w3 - 3 * w2 + 1) * y[..., i] + (w3 - 2 * w2 + w) * m[..., i]
                + (3 * w2 - 2 * w3) * y[..., i + 1] + (w3 - w2) * m[..., i + 1])

    # -- interpolation helpers (linear in asinh U and log10 T)
    def _u_index(self, U):
        f = (np.arcsinh(U / U_SCALE) - self.x0) / self.dx
        f = min(max(f, 0.0), self.t["U"].size - 1.0)
        i = min(int(f), self.t["U"].size - 2)
        return i, f - i

    def _hot(self, key, T, i, w):
        lt = self.t["log10_Thot"]
        y = min(max(np.log10(T), lt[0]), lt[-1])
        dl = (lt[-1] - lt[0]) / (lt.size - 1)
        fT = (y - lt[0]) / dl
        k = min(int(fT), lt.size - 2)
        v = fT - k
        tab = self.t[key]
        return ((1 - v) * ((1 - w) * tab[k, i] + w * tab[k, i + 1])
                + v * ((1 - w) * tab[k + 1, i] + w * tab[k + 1, i + 1]))

    def _sticking(self, Z):
        """WD01 eqs. 27-30 with the continuous extension used by the solver;
        the table's s_e(U), linear in the U-node index, where it has one (cm20);
        the table's stick_pos / stick_neg (cations / neutrals and anions) where it
        has them (a law fixed per table radius, ``gsd.charging`` ``sticking=``)."""
        a = self.a
        if Z <= self.t["Zmin"]:
            return 0.0
        if "s_e" in self.t:
            i, w = self._u_index(Z * E2_EV_CM / a)
            return (1 - w) * self.t["s_e"][i] + w * self.t["s_e"][i + 1]
        if "stick_pos" in self.t:
            return self.t["stick_pos"] if Z > 0 else self.t["stick_neg"]
        base = 0.5 * (1.0 - np.exp(-a / 1e-7))
        if Z > 0:
            return base
        Nc = 468.0 * (a / 1e-7) ** 3
        return base / (1.0 + np.exp(20.0 - Nc))

    # -- rates at (continuous) charge Z
    def _up(self, Z, i, w):
        """photoemission + ion capture + gas-electron secondaries at Z (node i + w)."""
        c = self.cell
        a = self.a
        if self.kernels == "linear":
            photo = c["cN"] @ self._kern(self.Kg, None, i, w) + c["G0_bg"] * self._kern(self.t["K_bg"], None, i, w)
        else:
            photo = max(c["cN"] @ self._kern(self.Kg, self._mK, i, w)
                        + c["G0_bg"] * self._kern(self.t["K_bg"], self._mKb, i, w), 0.0)
        arrive_e = np.pi * a * a * c["ne"] * c["v_e"] * float(DS87_J_function_vec(Z, -1.0, a, c["T"]))
        ion = 0.0
        for n_i, z_i, v_i in c["ions"]:
            ion += np.pi * a * a * n_i * v_i * float(DS87_J_function_vec(Z, z_i, a, c["T"]))
        return photo + ion + arrive_e * self._hot("delta", c["T"], i, w)

    def _down(self, Z, i, w):
        """electron capture at Z (node i + w)."""
        c = self.cell
        a = self.a
        arrive_e = np.pi * a * a * c["ne"] * c["v_e"] * float(DS87_J_function_vec(Z, -1.0, a, c["T"]))
        return arrive_e * self._sticking(Z) * (1.0 - self._hot("P_tr", c["T"], i, w))

    def _rates(self, Z):
        i, w = self._u_index(Z * E2_EV_CM / self.a)
        return self._up(Z, i, w), self._down(Z, i, w), (i, w)

    def _g(self, Z):
        """ln Up(Z) - ln Down(Z+1): only the up rate at Z and the down rate at Z+1."""
        up = self._up(Z, *self._u_index(Z * E2_EV_CM / self.a))
        down = self._down(Z + 1.0, *self._u_index((Z + 1.0) * E2_EV_CM / self.a))
        return np.log(max(up, 1e-300)) - np.log(max(down, 1e-300))

    def _heating_cooling(self, Z):
        c = self.cell
        a = self.a
        i, w = self._u_index(Z * E2_EV_CM / a)
        if self.kernels == "linear":
            hsum = float(c["cN"] @ self._kern(self.Hg, None, i, w)) + c["G0_bg"] * self._kern(self.t["H_bg"], None, i, w)
        else:
            hsum = max(float(c["cN"] @ self._kern(self.Hg, self._mH, i, w))
                       + c["G0_bg"] * self._kern(self.t["H_bg"], self._mHb, i, w), 0.0)
        gamma = EV2ERG * hsum
        kT = KB_CGS * c["T"]
        ltil = float(_ds87_lambda(Z, -1.0, a, c["T"]))
        arrive = np.pi * a * a * c["ne"] * c["v_e"]
        lam = arrive * self._sticking(Z) * ltil * kT * self._hot("f_cool", c["T"], i, w)
        jt = float(DS87_J_function_vec(Z, -1.0, a, c["T"]))
        lam -= arrive * jt * self._hot("delta", c["T"], i, w) * self._hot("E_bar", c["T"], i, w) * EV2ERG
        return gamma, lam

    # -- the same on arrays of charges (the exact P(Z) at the charging_isrf_tables nodes); the
    #    scalar versions above are what RAMSES mirrors
    def _u_index_arr(self, U):
        f = np.clip((np.arcsinh(U / U_SCALE) - self.x0) / self.dx, 0.0, self.t["U"].size - 1.0)
        i = np.minimum(f.astype(int), self.t["U"].size - 2)
        return i, f - i

    def _sticking_arr(self, Z):
        a = self.a
        if "s_e" in self.t:
            i, w = self._u_index_arr(Z * E2_EV_CM / a)
            s = (1 - w) * self.t["s_e"][i] + w * self.t["s_e"][i + 1]
            return np.where(Z <= self.t["Zmin"], 0.0, s)
        if "stick_pos" in self.t:
            base, neg = self.t["stick_pos"], self.t["stick_neg"]
            return np.where(Z <= self.t["Zmin"], 0.0, np.where(Z > 0, base, neg))
        base = 0.5 * (1.0 - np.exp(-a / 1e-7))
        neg = base / (1.0 + np.exp(20.0 - 468.0 * (a / 1e-7) ** 3))
        return np.where(Z <= self.t["Zmin"], 0.0, np.where(Z > 0, base, neg))

    def _photo_arr(self, Y, m, Ybg, mbg, i, w):
        c = self.cell
        if self.kernels == "linear":
            return c["cN"] @ self._kern(Y, None, i, w) + c["G0_bg"] * self._kern(Ybg, None, i, w)
        return np.maximum(c["cN"] @ self._kern(Y, m, i, w) + c["G0_bg"] * self._kern(Ybg, mbg, i, w), 0.0)

    def _up_arr(self, Z, i, w):
        c, a = self.cell, self.a
        up = self._photo_arr(self.Kg, getattr(self, "_mK", None), self.t["K_bg"], getattr(self, "_mKb", None), i, w)
        for n_i, z_i, v_i in c["ions"]:
            up = up + np.pi * a * a * n_i * v_i * DS87_J_function_vec(Z, z_i, a, c["T"])
        arrive_e = np.pi * a * a * c["ne"] * c["v_e"] * DS87_J_function_vec(Z, -1.0, a, c["T"])
        return up + arrive_e * self._hot("delta", c["T"], i, w)

    def _down_arr(self, Z, i, w):
        c, a = self.cell, self.a
        arrive_e = np.pi * a * a * c["ne"] * c["v_e"] * DS87_J_function_vec(Z, -1.0, a, c["T"])
        return arrive_e * self._sticking_arr(Z) * (1.0 - self._hot("P_tr", c["T"], i, w))

    def _g_arr(self, Z):
        Z = np.asarray(Z, float)
        up = self._up_arr(Z, *self._u_index_arr(Z * E2_EV_CM / self.a))
        down = self._down_arr(Z + 1.0, *self._u_index_arr((Z + 1.0) * E2_EV_CM / self.a))
        return np.log(np.maximum(up, 1e-300)) - np.log(np.maximum(down, 1e-300))

    def _heating_cooling_arr(self, Z):
        c, a = self.cell, self.a
        Z = np.asarray(Z, float)
        i, w = self._u_index_arr(Z * E2_EV_CM / a)
        gamma = EV2ERG * self._photo_arr(self.Hg, getattr(self, "_mH", None), self.t["H_bg"],
                                         getattr(self, "_mHb", None), i, w)
        arrive = np.pi * a * a * c["ne"] * c["v_e"]
        lam = arrive * self._sticking_arr(Z) * _ds87_lambda_electron_arr(Z, a, c["T"]) * KB_CGS * c["T"] \
            * self._hot("f_cool", c["T"], i, w)
        lam = lam - arrive * DS87_J_function_vec(Z, -1.0, a, c["T"]) * self._hot("delta", c["T"], i, w) \
            * self._hot("E_bar", c["T"], i, w) * EV2ERG
        return gamma, lam

    def _capture_z1(self, Z):
        """pi a^2 (8kT / pi m_u)^1/2 J~(Z, z=1) [cm^3 s^-1]: capture rate
        coefficient of a singly charged ion of mass 1 amu at grain charge Z."""
        T = self.cell["T"]
        v = np.sqrt(8.0 * KB_CGS * T / (np.pi * M_U))
        return np.pi * self.a ** 2 * v * np.atleast_1d(DS87_J_function_vec(np.atleast_1d(Z), 1.0, self.a, T))

    def _recombination(self, Z, P, h=1.0):
        """Grain-assisted recombination coefficients alpha_i [cm^3 s^-1 per grain]
        of RECOMB_IONS for charge nodes Z with weights P (summing to one), each
        node standing for h integer charges: sum_Z P(Z) pi a^2 v_i J~(Z, 1)
        Theta(Z <= Zth_i), v_i = v(m_u) (m_u/m_i)^1/2; the node that straddles
        Zth_i counts for the fraction of its charges at or below it."""
        A = self._capture_z1(Z)
        mass = np.array([m for _, m, _ in RECOMB_IONS])
        frac = lambda z: np.clip((z + 0.5 - (Z - 0.5 * h)) / h, 0.0, 1.0)  # noqa: E731
        return np.array([np.sum(P * A * frac(z)) for z in self.t["recomb_Zth"]]) / np.sqrt(mass)

    def _recombination_discrete(self, nodes, weights, z_floor=None):
        """Grain-assisted recombination of a discrete P(Z) (charges ``nodes``, ascending, with
        ``weights`` summing to one): the sum over its window and over the charges below it,
        with P continued from the window by g, one charge at a time, while those terms can
        matter. On a positive grain the capture rate rises by up to exp(e^2/akT) per charge
        towards lower Z, so P A can peak far below the window, at Z <= 0 for a small grain in
        cold gas. The walk stops at Zmin, or once a bound on all the remaining terms is e^-CUT
        below each ion's sum (sums below 1e-300 count as 1e-300). The bound is
        P(Z) A(Zmin) (Z - Zmin): P falls and A rises towards Zmin. There is no floor relative to
        the geometric rate, unlike _recombination_fit: below such a floor a stopped sum misses
        the charges that set it, and jumps by ~exp(e^2/akT) wherever the stop moves by one
        charge. The weights keep the window's normalisation; the tail adds less than
        DISCRETE_TAIL to it. z_floor: the lowest charge the walk may reach (RAMSES: its window buffers' lowest,
        b_base + 1)."""
        t = self.t
        Zlo, zth = t["Zmin"], t["recomb_Zth"]
        Z = np.asarray(nodes, float)
        W = np.asarray(weights, float)
        S = np.array([np.sum(W * self._capture_z1(Z) * (Z <= z)) for z in zth])
        lnA_lo = self._ln_capture_z1(Zlo)
        z, lnP = Z[0], np.log(W[0])
        for _ in range(MAX_DISCRETE_STATES):
            if z <= Zlo or (z_floor is not None and z - 1.0 < z_floor):
                break
            z -= 1.0
            lnP -= self._g(z)
            S = S + np.exp(lnP) * self._capture_z1(z)[0] * (z <= zth)
            bound = lnP + lnA_lo + np.log(max(z - Zlo, 1.0))
            if np.all(bound < np.log(np.maximum(S, 1e-300)) - RECOMB_CUT):
                break
        mass = np.array([m for _, m, _ in RECOMB_IONS])
        return S / np.sqrt(mass)

    def _ln_capture_z1(self, Z):
        """ln of the capture coefficient of _capture_z1, in log form (DS87 with
        theta_nu = nu/(1+nu^-1/2)), so the repulsive branch cannot underflow."""
        T = self.cell["T"]
        tau = self.a * KB_CGS * T / E_STATC ** 2
        if Z > 0:
            lnJ = 2.0 * np.log(1.0 + 1.0 / np.sqrt(4.0 * tau + 3.0 * Z)) - Z / (1.0 + 1.0 / np.sqrt(Z)) / tau
        elif Z < 0:
            lnJ = np.log((1.0 - Z / tau) * (1.0 + np.sqrt(2.0 / (tau - 2.0 * Z))))
        else:
            lnJ = np.log(1.0 + np.sqrt(np.pi / (2.0 * tau)))
        return float(np.log(np.pi * self.a ** 2 * np.sqrt(8.0 * KB_CGS * T / (np.pi * M_U))) + lnJ)

    def _recombination_exact(self, Zs, sigma):
        """Reference: P(Z) from detailed balance on every integer charge from
        min(-60, Z* - 20 sigma) to Z* + 10 sigma, then the exact sum."""
        Zlo, Zhi = self.t["Zmin"], self.t["Zmax"] - 1.0
        Z = np.arange(max(Zlo, min(-60.0, np.floor(Zs - 20 * sigma))), min(Zhi, np.ceil(Zs + 10 * sigma)) + 1.0)
        lnP = np.concatenate([[0.0], np.cumsum([self._g(z) for z in Z[:-1]])])
        P = np.exp(lnP - lnP.max())
        return self._recombination(Z, P / P.sum())

    def _break_charges(self):
        """Charges where g(Z) has kinks in this cell: IP(a, Z) at the edges of
        the groups with photons (and the Mathis breaks with a background), and
        the onsets of gas-electron secondaries and electron pass-through at T."""
        t, c = self.t, self.cell
        out = set()
        for (z0, z1), n in zip(self._edge_groups, c["cN"]):
            if n > 0:
                out.update(z for z in (z0, z1) if z is not None)
        if c["G0_bg"] > 0:
            out.update(self._edge_bg)
        lt = t["log10_Thot"]
        y = min(max(np.log10(c["T"]), lt[0]), lt[-1])
        fT = (y - lt[0]) / ((lt[-1] - lt[0]) / (lt.size - 1))
        k = min(int(fT), lt.size - 2)
        v = fT - k
        pos = t["U"] > 0
        for key in ("delta", "P_tr"):
            row = ((1 - v) * t[key][k] + v * t[key][k + 1])[pos]
            if row.max() <= 0:
                continue
            for frac in (1e-4, 0.5):
                j = np.nonzero(row > frac * row.max())[0]
                if j.size:
                    zc = float(np.round(t["U"][pos][j[0]] * self.a / E2_EV_CM))
                    if t["Zmin"] < zc < t["Zmax"]:
                        out.add(zc)
        return sorted(out)

    def _recombination_fit(self, Zs, sigma):
        """Grain-assisted recombination of a wide P(Z) at ~10 g(Z) evaluations.

        g(Z) is represented piecewise on the integers: exact on [-K, K] and on
        segments too short to fit, elsewhere a Chebyshev interpolant (<= FIT_N
        nodes) on segments split at the kinks of g (_break_charges); ln A is
        fitted the same way. ln P between nodes follows from the antiderivative
        of the fit with the Euler-Maclaurin term, sum_{j=p}^{m-1} g(j) =
        G(m-1/2) - G(p-1/2) - [g'(m-1/2) - g'(p-1/2)]/24. P A is evaluated on
        nodes spaced so that ln(P A) is nearly linear in between (every integer
        where exact), and the sums over the integers between nodes are the
        geometric series of that log-linear interpolant; ion thresholds take
        partial sums. The domain is Z* - 8 sigma .. Z* + 5 sigma, extended down to
        the ion thresholds within RECOMB_REACH sigma and while P A Theta can
        exceed RECOMB_FLOOR of the geometric rate."""
        t = self.t
        Zlo, Zhi = t["Zmin"], t["Zmax"] - 1.0
        K = RECOMB_KINK
        top = min(Zhi, np.floor(Zs + RECOMB_UP * sigma))
        bot = max(Zlo, np.ceil(Zs - RECOMB_DOWN * sigma))
        brk = self._break_charges()
        pieces = []

        def cover(a, b):                       # pieces for the integers [a, b)
            cuts = sorted({a, b} | {z for z in brk if a < z < b} | {z for z in (-K, K + 1) if a < z < b})
            for p, q1 in zip(cuts[:-1], cuts[1:]):
                q = q1 - 1
                ints = q - p + 1
                nseg = int(min(RECOMB_FIT_N, max(3, np.ceil(ints / RECOMB_PER_NODE))))
                if (-K <= p and q <= K) or ints <= nseg:
                    pieces.append(_GPiece(p, q, vals=np.array([self._g(z) for z in np.arange(p, q + 1.0)])))
                else:
                    nA = int(min(RECOMB_FIT_NA, nseg))
                    pieces.append(_GPiece(p, q, coef=_cheb_coef(self._g, p, q, nseg),
                                          coefA=_cheb_coef(self._ln_capture_z1, p, q, nA)))

        cover(bot, top + 1.0)
        zth = t["recomb_Zth"]
        reach = zth[(zth >= Zlo) & (zth < bot) & (Zs - zth < RECOMB_REACH * sigma)]
        if reach.size:
            new = max(Zlo, np.floor(reach.min() - 2.0 * sigma))
            cover(new, bot)
            bot = new
        lnA_geo = np.log(np.pi * self.a ** 2 * np.sqrt(8.0 * KB_CGS * self.cell["T"] / (np.pi * M_U)))
        lnA_lo = self._ln_capture_z1(Zlo)
        for _ in range(40):
            pieces.sort(key=lambda pc: pc.p)
            Zn, lnP, lnA = [], [], []
            base = 0.0
            for pc in pieces:
                if pc.vals is not None:
                    for z in np.arange(pc.p, pc.q + 1.0):
                        Zn.append(z), lnP.append(base + pc.gsum(z)), lnA.append(self._ln_capture_z1(z))
                else:
                    z, loc = pc.p, []
                    while True:
                        la = pc.lnA(z)
                        Zn.append(z), lnP.append(base + pc.gsum(z)), lnA.append(la)
                        loc.append((z, la))
                        if z >= pc.q:
                            break
                        curv = abs(pc.dg(z))
                        if len(loc) >= 3:
                            (z0, a0), (z1, a1), (z2, a2) = loc[-3:]
                            curv += abs(2.0 * ((a2 - a1) / (z2 - z1) - (a1 - a0) / (z1 - z0)) / (z2 - z0))
                        else:
                            curv += 1.0
                        hz = max(1.0, min(np.floor(np.sqrt(8.0 * RECOMB_EPS / max(curv, 1e-12))), max(1.0, np.floor(sigma))))
                        z = min(z + hz, pc.q)
                base += pc.gsum(pc.q + 1.0)
            Zn, lnP, lnA = np.array(Zn), np.array(lnP), np.array(lnA)
            lnP -= lnP.max()
            v = lnP + lnA
            if bot > Zlo and bot > 0 and lnP[0] + max(lnA[0] - lnA_geo, 0.0) < RECOMB_FLOOR:
                break                          # positive charges below: A <= geometric, P below the floor
            done = bot <= Zlo or lnP[0] < -RECOMB_CUT
            # bound on the terms below the domain: P falls and A rises towards Zmin
            below = lnP[0] + lnA_lo + np.log(bot - Zlo + 1.0)
            for z in zth:
                m = Zn <= z
                if done and m.any() and v[0] > v[m].max() - RECOMB_CUT and max(v[m].max(), below) - lnA_geo > RECOMB_FLOOR:
                    done = False               # the ion's terms can exceed the floor and have not converged
            if done:
                break
            new = max(Zlo, np.floor(bot - max(4.0 * sigma, 20.0)))
            cover(new, bot)
            bot = new
        norm = _geometric_sum(Zn, lnP, np.inf)
        mass = np.array([m for _, m, _ in RECOMB_IONS])
        return np.array([_geometric_sum(Zn, v, z) for z in zth]) / norm / np.sqrt(mass)

    def _illinois(self, a_, b_, fa, fb):
        """Root of g in [a_, b_] with g(a_) > 0 > g(b_)."""
        side = 0
        Zs = b_
        for _ in range(200):
            Zs = (a_ * fb - b_ * fa) / (fb - fa)
            if abs(b_ - a_) < 1e-7 * max(1.0, abs(Zs)):
                break
            fz = self._g(Zs)
            if fz == 0:
                break
            if fz * fb > 0:
                b_, fb = Zs, fz
                if side == -1:
                    fa *= 0.5
                side = -1
            else:
                a_, fa = Zs, fz
                if side == 1:
                    fb *= 0.5
                side = 1
        return Zs

    def _root_from_guess(self, Zg, Zlo, Zhi):
        """Bracket the root by doubling steps (1, 2, 4, ...) from Zg, then Illinois."""
        fg = self._g(Zg)
        if fg == 0:
            return Zg
        step = 1.0
        if fg > 0:              # root above Zg
            a_, fa = Zg, fg
            while True:
                b_ = min(a_ + step, Zhi)
                fb = self._g(b_)
                if fb == 0:
                    return b_
                if fb < 0:
                    return self._illinois(a_, b_, fa, fb)
                if b_ == Zhi:
                    return Zhi
                a_, fa = b_, fb
                step *= 2.0
        b_, fb = Zg, fg         # root below Zg
        while True:
            a_ = max(b_ - step, Zlo)
            fa = self._g(a_)
            if fa == 0:
                return a_
            if fa > 0:
                return self._illinois(a_, b_, fa, fb)
            if a_ == Zlo:
                return Zlo
            b_, fb = a_, fa
            step *= 2.0

    def _set_cell(self, N_g, c_red, T, ne, ions, G0_bg):
        self._inputs = (N_g, c_red, ne, ions, G0_bg)
        self._gcache = {}               # g at the integer charges of this cell (_gi)
        self.cell = dict(cN=c_red * np.asarray(N_g, dtype=float), T=float(T), ne=float(ne), G0_bg=float(G0_bg),
                         v_e=np.sqrt(8 * KB_CGS * T / (np.pi * ME_CGS)),
                         ions=[(float(n), float(z), np.sqrt(8 * KB_CGS * T / (np.pi * m))) for n, z, m in ions])

    def _root(self, Z_guess, warm):
        """Z*, sigma = (-1/g')^1/2 and the pair (z-, z+, g(z-), g(z+)) that gave g'.

        warm (the previous result of this bin): g at Z*_prev +- 1/2; if they
        bracket the root, Z* is the root of the line through them, accepted as
        is after a wide P(Z), and otherwise if g there implies an error below
        1e-3 charges (else Illinois within the pair, e.g. across the kink of g
        near Z = 0), with g' taken again around the new root if it moved by
        more than 0.01; if they do not bracket it, the bracket
        search starts from the line's root. Without warm: bracket from
        Z_guess (or the whole range) and Illinois, then g' by central difference."""
        Zlo, Zhi = self.t["Zmin"], self.t["Zmax"] - 1.0
        if warm is not None:
            zm, zp = max(warm["Zstar"] - 0.5, Zlo), min(warm["Zstar"] + 0.5, Zhi)
            gm, gpl = self._g(zm), self._g(zp)
            slope = (gpl - gm) / (zp - zm)
            if slope < 0:
                Zs = zm - gm / slope
                if zm <= Zs <= zp:
                    if not warm["discrete"]:          # wide: g is smooth on the scale of the pair
                        return Zs, np.sqrt(-1.0 / slope), (zm, zp, gm, gpl)
                    gs = self._g(Zs)
                    if abs(gs / slope) > 1e-3 and gm > 0 > gpl:
                        Zs = self._illinois(zm, zp, gm, gpl)
                    if abs(Zs - warm["Zstar"]) <= 0.01:
                        return Zs, np.sqrt(-1.0 / slope), (zm, zp, gm, gpl)
                    return (Zs,) + self._slope(Zs)
                Z_guess = min(max(Zs, Zlo + 1.0), Zhi - 1.0)
        if Z_guess is not None and Zlo < Z_guess < Zhi:
            Zs = self._root_from_guess(float(Z_guess), Zlo, Zhi)
        else:
            glo, ghi = self._g(Zlo), self._g(Zhi)
            if glo <= 0:
                Zs = Zlo
            elif ghi >= 0:
                Zs = Zhi
            else:
                Zs = self._illinois(Zlo, Zhi, glo, ghi)
        return (Zs,) + self._slope(Zs)

    def _recomb_inputs(self):
        """ln of the inputs a recombination fit depends on (for its reuse test)."""
        c = self.cell
        vals = [c["T"], c["ne"]] + [n for n in c["cN"] if n > 0] + [n for n, _, _ in c["ions"] if n > 0]
        if c["G0_bg"] > 0:
            vals.append(c["G0_bg"])
        return np.log(vals)

    def refine_recombination(self, out, warm):
        """The fitted recombination of a wide P(Z) for the cell of the last solve
        (out): the warm fit reused, times the change of its Gaussian estimate,
        while the inputs drift by less than RECOMB_REUSE_TOL and every resolved
        estimate by less than RECOMB_REUSE_EPS; else a new fit. Returns whether
        it fitted."""
        Zs, sigma = out["Zstar"], out["Zsigma"]
        x, lg = self._recomb_inputs(), self._recombination_gauss(Zs, sigma)
        ref = warm.get("recomb_ref") if warm is not None else None
        if ref is not None and ref[0].size == x.size and np.max(np.abs(x - ref[0])) <= RECOMB_REUSE_TOL:
            est = ref[1] * np.exp(lg - ref[2])
            floor = RECOMB_RESOLVED * self._geometric_rates()
            m = (ref[1] > floor) | (est > floor)
            if not m.any() or np.max(np.abs(lg - ref[2])[m]) <= RECOMB_REUSE_EPS:
                out["alpha_rec"], out["recomb_ref"] = est, ref
                return False
        out["alpha_rec"] = self._recombination_fit(Zs, sigma)
        out["recomb_ref"] = (x, out["alpha_rec"], lg)
        return True

    def _geometric_rates(self):
        """pi a^2 v_i of RECOMB_IONS at the cell temperature [cm^3 s^-1]."""
        mass = np.array([m for _, m, _ in RECOMB_IONS])
        return np.pi * self.a ** 2 * np.sqrt(8.0 * KB_CGS * self.cell["T"] / (np.pi * mass * M_U))

    def _recombination_gauss(self, Zs, sigma):
        """ln of a cheap estimate of alpha_i for a wide P(Z), for the reuse of a fit:
        P Gaussian (mean Z* + 1/2, width sigma) times ln A linear around the peak
        of P A (two iterations), summed over Z <= Zth_i in closed form."""
        Zlo, Zhi = self.t["Zmin"], self.t["Zmax"] - 1.0
        mu = Zs + 0.5
        zp = mu
        for _ in range(2):
            z0 = min(max(zp, Zlo + 1.0), Zhi - 1.0)
            s = 0.5 * (self._ln_capture_z1(z0 + 1.0) - self._ln_capture_z1(z0 - 1.0))
            zp = mu + s * sigma ** 2
        zp = min(max(zp, Zlo), Zhi)
        mup = mu + s * sigma ** 2
        mass = np.array([m for _, m, _ in RECOMB_IONS])
        base = self._ln_capture_z1(zp) + s * (mu - zp) + 0.5 * s * s * sigma * sigma
        return np.array([base + _log_ndtr((z + 0.5 - mup) / sigma) - 0.5 * np.log(m)
                         for z, m in zip(self.t["recomb_Zth"], mass)])

    def _slope(self, Zs):
        """sigma from g' by central difference at Z* +- 1/2, and that pair."""
        zm, zp = max(Zs - 0.5, self.t["Zmin"]), min(Zs + 0.5, self.t["Zmax"] - 1.0)
        gm, gpl = self._g(zm), self._g(zp)
        slope = (gpl - gm) / (zp - zm)
        return (np.sqrt(-1.0 / slope) if slope < 0 else 0.0), (zm, zp, gm, gpl)

    def _chord(self, Zs, sigma):
        """sigma from the chord of g over the integers floor(Z*) - k + 1 .. floor(Z*) + k,
        k = round(sigma), re-centred from the new sigma until k is stable (at most
        SIGMA_CHORD_ITER chords; sigma: the starting width). The secant at Z* +- 1/2
        (_slope) is off by up to a factor ~3 where g has a kink next to Z*. Returns
        sigma and the chord (lo, hi, g(lo), g(hi)), or the starting sigma and None if
        no chord has g decreasing."""
        Zlo, Zhi = self.t["Zmin"], self.t["Zmax"] - 1.0
        k, chord = None, None
        for _ in range(SIGMA_CHORD_ITER):
            kn = max(1, _nint(sigma))
            if kn == k:
                break
            k = kn
            lo, hi = max(np.floor(Zs) - k + 1.0, Zlo), min(np.floor(Zs) + k, Zhi)
            if hi <= lo:
                break
            glo, ghi = self._g(lo), self._g(hi)
            slope = (ghi - glo) / (hi - lo)
            if slope >= 0:
                break
            sigma, chord = np.sqrt(-1.0 / slope), (lo, hi, glo, ghi)
        return sigma, chord

    def exact_distribution(self, Zs, width, tail=1e-12):
        """The exact P(Z) of the cell set by the last solve: g on every integer within
        8 width + 10 of Z*, trimmed where P < tail x its maximum (the reference that
        the Gaussian of a wide P(Z) approximates). Returns the charges and P."""
        Z = np.arange(max(self.t["Zmin"], np.floor(Zs - 8 * width - 10)),
                      min(self.t["Zmax"], np.ceil(Zs + 8 * width + 10)) + 1.0)
        lnf = np.concatenate([[0.0], np.cumsum(self._g_arr(Z[:-1]))])
        lnf -= lnf.max()
        k = lnf > np.log(tail)
        P = np.exp(lnf[k])
        return Z[k], P / P.sum()

    def _electron_affinity(self, Z):
        """The grain's electron affinity at charge Z [eV], for the autoionisation cooling at Zmin (the one place
        both closures take it from)."""
        return float(wy.electron_affinity(Z, self.a, self.material))

    def _averages(self, nodes, weights, discrete):
        """Gamma, Lambda (+ autoionisation at Zmin) averaged over the charge nodes."""
        gam = lam = 0.0
        for Z, wgt in zip(nodes, weights):
            g_, l_ = self._heating_cooling(Z)
            gam += wgt * g_
            lam += wgt * l_
        auto = 0.0
        # cm20: none, as in DustEM, whose electron sticking at Zmin is zero (its EA(Zmin) can be < 0)
        if discrete and nodes[0] == self.t["Zmin"] and self.material != "cm20":
            EA = self._electron_affinity(self.t["Zmin"])
            arrive = np.pi * self.a ** 2 * self.cell["ne"] * self.cell["v_e"]
            auto = weights[0] * arrive * float(DS87_J_function_vec(self.t["Zmin"], -1.0, self.a, self.cell["T"])) * EA * EV2ERG
        return float(gam), float(lam), float(auto)

    # -- the charge distribution (RAMSES rtgroups_solve_bin's distribution(), PR #86)
    def _ud(self, k):
        """Up and Down at the integer charge k, once per cell."""
        key = ("ud", k)
        v = self._gcache.get(key)
        if v is None:
            z = float(k)
            i, w = self._u_index(z * E2_EV_CM / self.a)
            v = self._gcache[key] = (self._up(z, i, w), self._down(z, i, w))
        return v

    def _gi(self, k):
        """g at the integer charge k, ln Up(k) - ln Down(k + 1) (RAMSES gwin), once per cell; the same operations as
        _g, so the same values."""
        g = self._gcache.get(k)
        if g is None:
            g = self._gcache[k] = np.log(max(self._ud(k)[0], 1e-300)) - np.log(max(self._ud(k + 1)[1], 1e-300))
        return g

    def _window(self, z0, z1, keep_base=False):
        """Exact P(Z) on the integers z0..z1, grown or trimmed at each end to one charge past where
        P < DISCRETE_TAIL of its maximum. Returns the charges, P, the g increments between them, and
        whether the growth stopped at MAX_DISCRETE_STATES with P still above the tail at an end that could
        grow (capped: the window need not hold the peak). Unless keep_base, the window's charges set RAMSES's
        buffer base (b_base), as a window built afresh does there."""
        Zlo, Zhi, ln_tail = _nint(self.t["Zmin"]), _nint(self.t["Zmax"]), np.log(DISCRETE_TAIL)
        lo, hi = _nint(z0), _nint(z1)
        if not keep_base:
            self._b_base = lo - MAX_DISCRETE_STATES - 2
        while True:
            ginc = [self._gi(k) for k in range(lo, hi)]
            lnf = np.concatenate([[0.0], np.cumsum(ginc)])
            lnf -= lnf.max()
            room = hi - lo + 1 < MAX_DISCRETE_STATES
            up_lo = lnf[0] > ln_tail and lo > Zlo
            up_hi = lnf[-1] > ln_tail and hi < Zhi
            if not (room and (up_lo or up_hi)):
                capped = (not room) and (up_lo or up_hi)
                break
            lo -= 1 if up_lo else 0
            hi += 1 if up_hi else 0
        k = np.nonzero(lnf > ln_tail)[0]
        i0, i1 = max(k[0] - 1, 0), min(k[-1] + 1, hi - lo)
        P = np.exp(lnf[i0:i1 + 1])
        return np.arange(lo + i0, lo + i1 + 1, dtype=float), P / P.sum(), np.array(ginc[i0:i1]), capped

    @staticmethod
    def _root_on_window(Z, ginc):
        """Z*: the root of g interpolated between the integers of a window; sig_l, the width of g's integer slope
        g(k*+1) - g(k*) there, and k* (sig_l = 0 and k* = lo - 1 if the window does not bracket the root)."""
        lo, hi = int(Z[0]), int(Z[-1])
        Zs, sig_l, kb = float(hi), 0.0, lo - 1
        if hi == lo or ginc[0] <= 0:
            Zs = float(lo)
        for i in range(hi - lo - 1):
            if ginc[i] > 0 and ginc[i + 1] <= 0:
                Zs = (lo + i) + ginc[i] / (ginc[i] - ginc[i + 1])
                sig_l = np.sqrt(-1.0 / (ginc[i + 1] - ginc[i]))
                kb = lo + i
                break
        return Zs, sig_l, kb

    def _lattice(self, kg):
        """The integer bracket ks of the root, g(ks) > 0 >= g(ks + 1), walked from kg one charge at a time, at
        most LAT_NWALK steps: (ks, found). Not found if the root is farther, or at an end of the charge range."""
        zl, zh = _nint(self.t["Zmin"]), _nint(self.t["Zmax"])
        ks = kg
        if kg < zl + 1 or kg + 2 > zh:
            return ks, False
        self._b_base = kg - MAX_DISCRETE_STATES - LAT_NWALK - 4
        if self._gi(kg) > 0:
            for _ in range(LAT_NWALK):
                if ks + 2 > zh:
                    return ks, False
                if self._gi(ks + 1) <= 0:
                    return ks, True
                ks += 1
        else:
            for _ in range(LAT_NWALK):
                if ks - 1 < zl:
                    return ks, False
                ks -= 1
                if self._gi(ks) > 0:
                    return ks, True
        return ks, False

    def _lattice_narrow(self, Zs, sig_l):
        """Whether the root's width on the lattice is narrow: its integer slope (sig_l) below SIGMA_DISCRETE,
        or else the integer chord from Zs and sig_l (_chord); and the width used, to start the window."""
        if sig_l < SIGMA_DISCRETE:
            return True, sig_l
        sig_n = sig_l
        if sig_l < HUGE:
            sig_n = self._chord(Zs, sig_l)[0]
        return sig_n < SIGMA_DISCRETE, sig_n

    def _distribution(self, Z_guess, warm):
        """P(Z) of the cell (RAMSES PR #86, CH1): one rule on the warm and the cold path, so that the result does
        not depend on the start. P(Z) is discrete (exact on a window of integers, _window) iff the root's width
        on the lattice is narrow (_lattice_narrow) and the converged window, not capped, has a width below
        SIGMA_DISCRETE; Z* is then the root of g interpolated between the integers. The bracket is walked on
        the lattice from the guess (the warm Z*, else Z_guess), or from the continuous root (_root) if it is
        farther. A warm discrete window is re-formed as a cold one. Otherwise P(Z) is wide: the continuous root
        within the lattice bracket, its width from the pair and the chord (from the pair's width), averaged at
        its mean Z* + 1/2 (_wide_nodes). At an end of the charge range the continuous root decides, as before
        PR #86 (a warm wide P(Z) then starts the chord from its last width).
        Returns dict(Zstar, sigma, pair, chord, discrete, nodes, weights)."""
        t = self.t
        Zmin, Zmax = t["Zmin"], t["Zmax"]
        got = okl = False
        Zs = sigma = pair = chord = Z = P = None
        ks = None
        if warm is not None and warm["discrete"]:
            Z, P, ginc, capped = self._window(warm["nodes"][0], warm["nodes"][-1])
            if _window_moments(Z, P)[1] < SIGMA_DISCRETE and not capped:
                Zs, sig_l, ks = self._root_on_window(Z, ginc)
                got, sig_n = self._lattice_narrow(Zs, sig_l)
                if got and ks >= Z[0]:
                    # the window a cold solve grows from k* and its width: the same P(Z) whatever the start
                    half = int(np.ceil(3.0 * max(sig_n, 0.5)))
                    Z, P, ginc, capped = self._window(max(Zmin, ks - half + 1.0), min(Zmax, ks + half + 0.0),
                                                      keep_base=True)
                    Zs, sig_l, ks = self._root_on_window(Z, ginc)
                    got = _window_moments(Z, P)[1] < SIGMA_DISCRETE and not capped
        if not got:
            zg = -HUGE if Z_guess is None else float(Z_guess)
            if warm is not None:
                zg = warm["Zstar"]
            if Zmin < zg < Zmax - 1.0:
                ks, okl = self._lattice(int(np.floor(zg)))
            if not okl:
                Zs, sigma, pair = self._root(Z_guess, warm)
                kg = int(np.floor(Zs))
                if Zmin < kg < Zmax - 1.0:
                    ks, okl = self._lattice(kg)
            if okl:
                g0, g1 = self._gi(ks), self._gi(ks + 1)
                sig_l = np.sqrt(-1.0 / (g1 - g0)) if g1 < g0 else HUGE
                Zs = ks + g0 / (g0 - g1)
                narrow, sig_n = self._lattice_narrow(Zs, sig_l)
                if narrow:
                    half = int(np.ceil(3.0 * max(sig_n, 0.5)))
                    Z, P, ginc, capped = self._window(max(Zmin, ks - half + 1.0), min(Zmax, ks + half + 0.0),
                                                      keep_base=True)
                    if _window_moments(Z, P)[1] < SIGMA_DISCRETE and not capped:
                        Zs = self._root_on_window(Z, ginc)[0]
                        got = True
        if got:
            discrete = True
        elif okl:
            Zs = self._illinois(float(ks), float(ks + 1), self._gi(ks), self._gi(ks + 1))
            sigma, pair = self._slope(Zs)
            chord = None
            if sigma >= SIGMA_DISCRETE:
                sigma, chord = self._chord(Zs, sigma)
            discrete = False
        else:
            # the root at an end of the charge range (_root already ran above, with the same result)
            chord = None
            if sigma >= SIGMA_DISCRETE:
                start = warm["Zsigma"] if warm is not None and not warm["discrete"] else sigma
                sigma, chord = self._chord(Zs, start)
            discrete = sigma < SIGMA_DISCRETE
            if discrete:
                half = int(np.ceil(3.0 * max(sigma, 0.5)))
                Zc = int(np.floor(Zs + 0.5))
                Z, P, _, _ = self._window(max(Zmin, Zc - half + 0.0), min(Zmax, Zc + half + 0.0))
        if not discrete:
            Z, P = _wide_nodes(Zs)
        return dict(Zstar=float(Zs), sigma=sigma, pair=pair, chord=chord, discrete=discrete, nodes=Z, weights=P)

    # -- the exact-P(Z) correction of the Gaussian Coulomb factors (RAMSES window_ratio, rtgroups_coulomb_ratio)
    @staticmethod
    def _coulomb_term(k, j, C1):
        """The Weingartner & Draine (1999) focusing factor of charge k for an impactor of charge j (the neutral
        grain's with RAMSES's pi, coulomb_focusing.PI_RAMSES, as RAMSES's sums)."""
        from pycalima.models.dust_charge.coulomb_focusing import PI_RAMSES
        if k == 0:
            return 1.0 + abs(j) * np.sqrt(PI_RAMSES * C1 / 2.0)
        if k * j > 0:
            x = -float(k * j) * C1
            return np.exp(-745.0) if x <= -745.0 else np.exp(x)
        return 1.0 - float(k * j) * C1

    def _coulomb_range(self, out):
        Zs, sg = out["Zstar"], max(out["Zsigma"], 0.5)
        rlo = max(_nint(self.t["Zmin"]), min(int(np.floor(Zs - 8.0 * sg - 10.0)), -3))
        rhi = min(_nint(self.t["Zmax"]), max(int(np.ceil(Zs + 8.0 * sg + 10.0)), 3))
        return rlo, rhi

    def _coulomb_ratio(self, out):
        """D_j over the cell's exact P(Z) over D_j of the Gaussian of <Z>, sigma (coulomb_focusing), for the
        impactor charges COULOMB_DCHARGE, of a discrete P(Z) (1 for a wide one): RAMSES window_ratio. P is
        continued outward from the window one charge at a time until it falls below RATIO_LN_CUT of its maximum
        or, while P is unimodal, the geometric bound on everything beyond (log-concave P) is below RATIO_EPS of
        the sums, within the charges of Z* -+ (8 sigma + 10) and -3..+3. Where that range does not fit RAMSES's
        window buffers, the sum over the whole range (_coulomb_ratio_range); where it does not fit them at all, 1."""
        from pycalima.models.dust_charge import coulomb_focusing as cf

        ratio = np.ones(len(COULOMB_DCHARGE))
        if not out["discrete"]:
            return ratio
        lo, hi = int(out["nodes"][0]), int(out["nodes"][-1])
        rlo, rhi = self._coulomb_range(out)
        if rhi - rlo + 3 > NBUF:
            return ratio
        k0, k1 = max(lo, rlo), min(hi, rhi)
        bb = self._b_base
        if k0 > k1 or rlo - bb < 1 or rhi - bb > NBUF:
            return self._coulomb_ratio_range(out)
        C1 = cf.E2 / (cf.KB * self.cell["T"] * self.a)
        sq = np.sqrt(cf.PI_RAMSES * C1 / 2.0)
        lp = {k0: 0.0}
        for k in range(k0, k1):
            lp[k + 1] = lp[k] + self._gi(k)
        pm = max(lp[k] for k in range(k0, k1 + 1))
        gk, fk = {}, {}
        acc = dict(ps=0.0, px=np.zeros(len(COULOMB_DCHARGE)))

        def term(k):
            gk[k], fk[k] = 0.0, np.zeros(len(COULOMB_DCHARGE))
            g1 = lp[k] - pm
            if g1 <= RATIO_LN_CUT:
                return
            g1 = np.exp(g1)
            gk[k] = g1
            acc["ps"] += g1
            for m, j in enumerate(COULOMB_DCHARGE):
                f = self._coulomb_term(k, j, C1)
                fk[k][m] = f
                acc["px"][m] += g1 * f

        def tail_done(k, lnr, down):
            # everything beyond the last charge k of a direction is negligible: P(k +- n) <= P(k) r^n, r < 1
            if not lnr < 0.0:
                return False
            r = np.exp(lnr)
            g1 = r / (1.0 - r)
            g2 = g1 / (1.0 - r)
            if gk[k] * g1 > RATIO_EPS * acc["ps"]:
                return False
            for m, j in enumerate(COULOMB_DCHARGE):
                if j > COULOMB_JMAX:
                    continue
                a_ = 1.0 + abs(j) * sq + float(abs(k) * abs(j)) * C1
                b_ = abs(j) * C1
                tm = gk[k] * (a_ * g1 + b_ * g2)
                if tm <= RATIO_EPS * acc["px"][m]:
                    continue
                if down and (acc["px"][m] + tm) * (1.0 + 4.0 * RATIO_EPS) < RATIO_CLAMP * acc["ps"]:
                    continue
                return False
            return True

        mono = True
        for k in range(k0, k1 + 1):
            term(k)
        while k1 < rhi and lp[k1] - pm > RATIO_LN_CUT:
            if mono and k1 > k0 and tail_done(k1, self._gi(k1 - 1), False):
                break
            lp[k1 + 1] = lp[k1] + self._gi(k1)
            k1 += 1
            if lp[k1] > pm:
                mono = False
            pm = max(pm, lp[k1])
            if mono:
                term(k1)
        while k0 > rlo and lp[k0] - pm > RATIO_LN_CUT:
            if mono and tail_done(k0, -self._gi(k0), True):
                break
            lp[k0 - 1] = lp[k0] - self._gi(k0 - 1)
            k0 -= 1
            if lp[k0] > pm:
                mono = False
            pm = max(pm, lp[k0])
            if mono:
                term(k0)
        ss, Dx = 0.0, np.zeros(len(COULOMB_DCHARGE))
        for k in range(k0, k1 + 1):
            gg = lp[k] - pm
            if gg <= RATIO_LN_CUT:
                continue
            if mono:
                ss += gk[k]
                Dx += gk[k] * fk[k]
            else:
                gg = np.exp(gg)
                ss += gg
                Dx += gg * np.array([self._coulomb_term(k, j, C1) for j in COULOMB_DCHARGE])
        Dg = cf.coulomb_factors(out["Zmean"], out["Zsigma"], self.cell["T"], self.a, COULOMB_JMAX)
        return (Dx / ss) / Dg[[j + 1 for j in COULOMB_DCHARGE]]

    def _coulomb_ratio_range(self, out):
        """_coulomb_ratio summed over every charge of its range down to 1e-300 of the maximum (RAMSES
        rtgroups_coulomb_ratio); 1 where the range exceeds the buffers."""
        from pycalima.models.dust_charge import coulomb_focusing as cf

        lo, hi = self._coulomb_range(out)
        if hi - lo + 3 > NBUF:
            return np.ones(len(COULOMB_DCHARGE))
        lnP = np.concatenate([[0.0], np.cumsum([self._gi(k) for k in range(lo, hi)])])
        pmax = lnP.max()
        C1 = cf.E2 / (cf.KB * self.cell["T"] * self.a)
        s, Dx = 0.0, np.zeros(len(COULOMB_DCHARGE))
        for k, lp in zip(range(lo, hi + 1), lnP):
            g = lp - pmax
            if g <= RATIO_LN_CUT:
                continue
            g = np.exp(g)
            s += g
            Dx += g * np.array([self._coulomb_term(k, j, C1) for j in COULOMB_DCHARGE])
        Dg = cf.coulomb_factors(out["Zmean"], out["Zsigma"], self.cell["T"], self.a, COULOMB_JMAX)
        return (Dx / s) / Dg[[j + 1 for j in COULOMB_DCHARGE]]

    def _dlnT_response(self, out, pair, dlnT, chord=None, sigma=None, shift=0.0):
        """d/d ln T of Zmean, Zsigma, Gamma, Lambda (+ auto). Discrete P(Z):
        analytic (_dlnT_discrete). Wide: one more set of rates at T (1 + dlnT),
        the pair g(z+-) re-evaluated, so the shifted root is consistent with the
        T call by construction, and the width from the same chord (or pair), then
        Gamma and Lambda re-averaged on the shifted Gaussian nodes. sigma: the Gaussian's width (the result's,
        unless _wide_moments set it); shift: _wide_moments' shift of <Z>, which the rates at T (1 + dlnT) carry; the
        d/d ln T of <Z> and sigma stay the Gaussian's."""
        if out["discrete"]:
            return self._dlnT_discrete(out)
        T0 = self.cell["T"]
        N_g, c_red, ne, ions, G0_bg = self._inputs
        self._set_cell(N_g, c_red, T0 * (1.0 + dlnT), ne, ions, G0_bg)
        zm, zp, gm, gpl = pair
        gm1, gp1 = self._g(zm), self._g(zp)
        s0, s1 = (gpl - gm) / (zp - zm), (gp1 - gm1) / (zp - zm)
        Zs1 = out["Zstar"] + (zm - gm1 / s1) - (zm - gm / s0)
        sg = out["Zsigma"] if sigma is None else sigma
        sig1 = np.sqrt(-1.0 / s1) if s1 < 0 else sg
        if chord is not None:
            lo, hi = chord[:2]
            c1 = (self._g(hi) - self._g(lo)) / (hi - lo)
            sig1 = np.sqrt(-1.0 / c1) if c1 < 0 else sg
        nodes1, weights1 = _wide_nodes(Zs1 + shift)
        gam1, lam1, _ = self._averages(nodes1, weights1, False)
        self._set_cell(N_g, c_red, T0, ne, ions, G0_bg)
        h = np.log1p(dlnT)
        return dict(dZmean_dlnT=(Zs1 - out["Zstar"]) / h, dZsigma_dlnT=(sig1 - sg) / h,
                    dGamma_dlnT=(gam1 - out["Gamma"]) / h, dLambda_dlnT=(lam1 - out["Lambda"]) / h)

    def _hot_dlnT(self, key, i, w):
        """d/d ln T of _hot: its slope in log10 T (zero where log T is clamped)."""
        lt = self.t["log10_Thot"]
        y = np.log10(self.cell["T"])
        if not lt[0] < y < lt[-1]:
            return 0.0
        dl = (lt[-1] - lt[0]) / (lt.size - 1)
        k = min(int((y - lt[0]) / dl), lt.size - 2)
        tab = self.t[key]
        a0 = (1 - w) * tab[k, i] + w * tab[k, i + 1]
        a1 = (1 - w) * tab[k + 1, i] + w * tab[k + 1, i + 1]
        return (a1 - a0) / dl / np.log(10.0)

    def _rates_dlnT(self, Z):
        """At charge Z: up, down, the cooling lam of _heating_cooling, and their
        d/d ln T (tau ~ T, v ~ T^1/2; photoemission does not depend on T)."""
        c, a = self.cell, self.a
        i, w = self._u_index(Z * E2_EV_CM / a)
        up, down = self._up(Z, i, w), self._down(Z, i, w)
        arr = np.pi * a * a * c["ne"] * c["v_e"]
        tau_e = a * KB_CGS * c["T"] / E_STATC ** 2
        je, dje = _ds87_J_dlnT(Z, -1.0, tau_e)
        lt, dlt = _ds87_lambda_dlnT(Z, tau_e)
        dion = 0.0
        for n_i, z_i, v_i in c["ions"]:
            ji, dji = _ds87_J_dlnT(Z, z_i, a * KB_CGS * c["T"] / (z_i * z_i * E_STATC ** 2))
            dion += np.pi * a * a * n_i * v_i * ji * (0.5 + dji)
        st = self._sticking(Z)
        de, dde = self._hot("delta", c["T"], i, w), self._hot_dlnT("delta", i, w)
        pt, dpt = self._hot("P_tr", c["T"], i, w), self._hot_dlnT("P_tr", i, w)
        fc, dfc = self._hot("f_cool", c["T"], i, w), self._hot_dlnT("f_cool", i, w)
        eb, deb = self._hot("E_bar", c["T"], i, w), self._hot_dlnT("E_bar", i, w)
        kT = KB_CGS * c["T"]
        dup = dion + arr * je * (de * (0.5 + dje) + dde)
        ddown = arr * je * st * ((1.0 - pt) * (0.5 + dje) - dpt)
        lam = arr * (st * lt * kT * fc - je * de * eb * EV2ERG)
        dlam = arr * (st * lt * kT * (fc * (1.5 + dlt) + dfc) - je * EV2ERG * (de * eb * (0.5 + dje) + dde * eb + de * deb))
        return up, dup, down, ddown, lam, dlam

    def _dlnT_discrete(self, out):
        """Analytic d/d ln T for a discrete P(Z) on its window: d ln P(Z_k) is the
        cumulative sum of d ln Up(Z_j) - d ln Down(Z_j+1) (centred on its mean),
        and Gamma (no T dependence per charge) and Lambda move with it, Lambda
        also through each charge's cooling rate."""
        Z, P = out["nodes"], out["weights"]
        r = np.array([self._rates_dlnT(z) for z in Z])            # up, dup, down, ddown, lam, dlam
        gam = np.array([self._heating_cooling(z)[0] for z in Z])
        c = np.concatenate([[0.0], np.cumsum(r[:-1, 1] / np.maximum(r[:-1, 0], 1e-300)
                                             - r[1:, 3] / np.maximum(r[1:, 2], 1e-300))])
        c -= P @ c
        dmean = P @ (c * Z)
        dvar = P @ (c * Z * Z) - 2.0 * out["Zmean"] * dmean
        dlam = P @ (c * r[:, 4]) + P @ r[:, 5]
        if Z[0] == self.t["Zmin"]:
            tau_e = self.a * KB_CGS * self.cell["T"] / E_STATC ** 2
            dlam += out["Lambda_auto"] * (c[0] + 0.5 + _ds87_J_dlnT(Z[0], -1.0, tau_e)[1])
        return dict(dZmean_dlnT=dmean, dZsigma_dlnT=0.5 * dvar / out["Zsigma"] if out["Zsigma"] > 0 else 0.0,
                    dGamma_dlnT=P @ (c * gam), dLambda_dlnT=dlam)

    # -- alpha and D of a positive grain from its P(Z) toward Z = 0 (RAMSES tail_sums, calima/charge-lg5 commits 2, 4)
    def _tail_sums(self, m, sl, kw, gw, lnPm, zmean, want_a, want_d):
        """The grain-assisted recombination alpha_i and the Coulomb factors D_j of a positive grain's P(Z) whose
        tail toward Z = 0 carries them. alpha and D(+j) come from P(Z) near and below Z = 0, where g bends at the
        kink of the Coulomb factors, and a closure that continues g from the window puts far too much P there
        (D(+1) too large by up to e^6, alpha too small by up to e^80).
          m: the mode of P(Z); sl: its central width; kw, gw: g on the window charges (consecutive);
          lnPm, zmean: ln P(m) (normalised) and <Z> of the closure.
        The tail is used if, with g continued linearly from the window's lower edge, the alpha or D(+1)
        integrand at Z = 1, 0, -2 comes within e^-TAIL_CUT of its value on the window (at its lowest charges and
        the mode). Then g is also taken at the kink charges -2 .. 1 and at two charges between them and the
        window, PCHIP between the nodes, linear beyond them; ln P(lo) - ln P(m) is minus the sum of that g over
        lo .. m-1, in closed form on each piece, and P(Z) = P(m) exp(...) is carried up from lo = -6 only as far
        as the sums go: alpha_i (case-A thresholds) and D(+j) until their terms are past their peak and below
        e^-TAIL_STOP of their sums; D(-1) = 1 + C1 <Z> plus its exact difference from that on Z <= 0. D floored
        at 1e-10. Returns (used, alpha or None, D or None)."""
        n = len(kw)
        if not (want_a or want_d) or n < 2:
            return False, None, None
        t = self.t
        klo, khi = _nint(t["Zmin"]), _nint(t["Zmax"])
        if not (m >= 1 and kw[0] >= -1):
            return False, None, None
        T = self.cell["T"]
        C1 = E_STATC ** 2 / (KB_CGS * T * self.a)
        sq = np.sqrt(np.pi * C1 / 2.0)
        geoA = np.pi * self.a ** 2 * np.sqrt(8.0 * KB_CGS * T / (np.pi * M_U))

        def capture(z):
            return geoA * float(DS87_J_function_vec(float(z), 1.0, self.a, T))

        def lnw(kz):
            # the steepest wanted integrand weight at charge kz (ln): alpha's capture rate, D(+1)
            v = -HUGE
            if want_a:
                v = np.log(max(capture(kz), 1e-300))
            if want_d:
                v = max(v, -kz * C1 if kz > 0 else (np.log(1.0 + sq) if kz == 0 else np.log(1.0 - kz * C1)))
            return v

        lpw = np.concatenate([[0.0], np.cumsum(gw)])
        peak = max(lpw[0] + lnw(kw[0]), lpw[1] + lnw(kw[0] + 1), lpw[m - kw[0]] + lnw(m))
        sl_e = gw[1] - gw[0]
        if not sl_e < 0.0:
            sl_e = -1.0 / max(sl * sl, 1e-12)
        used = False
        for zt in (1, 0, -2):
            if zt >= kw[0] or zt < klo:
                continue
            dz = kw[0] - zt
            lpt = -(dz * gw[0] - sl_e * float(dz) * float(dz + 1) / 2.0)
            if lpt + lnw(zt) > peak - TAIL_CUT:
                used = True
                break
        if not used:
            return False, None, None
        # the nodes of g: the window, the kink charges -2 .. 1, two between them and the window
        nodes = {int(k): float(g) for k, g in zip(kw, gw)}
        for kq in range(-2, 2):
            if kw[0] <= kq <= kw[-1] or kq < klo or kq + 1 > khi:
                continue
            nodes[kq] = self._gi(kq)
        a0, b0 = 1, int(kw[0])
        for q in (1, 2):
            kq = _nint(a0 + (b0 - a0) * q / 3.0)
            if kq <= a0 or kq >= b0 or kq in nodes:
                continue
            nodes[kq] = self._gi(kq)
        xn = np.array(sorted(nodes), dtype=float)
        yn = np.array([nodes[int(x)] for x in xn])
        nn = xn.size
        dk = np.zeros(nn)
        dk[0] = (yn[1] - yn[0]) / (xn[1] - xn[0])
        dk[-1] = (yn[-1] - yn[-2]) / (xn[-1] - xn[-2])
        for i in range(1, nn - 1):
            s0 = (yn[i] - yn[i - 1]) / (xn[i] - xn[i - 1])
            s1 = (yn[i + 1] - yn[i]) / (xn[i + 1] - xn[i])
            if s0 * s1 > 0.0:
                w1 = 2.0 * (xn[i + 1] - xn[i]) + (xn[i] - xn[i - 1])
                w2 = (xn[i + 1] - xn[i]) + 2.0 * (xn[i] - xn[i - 1])
                dk[i] = (w1 + w2) / (w1 / s0 + w2 / s1)

        def gmod(kz):
            xx = float(kz)
            if xx <= xn[0]:
                return yn[0] + dk[0] * (xx - xn[0])
            if xx >= xn[-1]:
                return yn[-1] + dk[-1] * (xx - xn[-1])
            i = int(np.searchsorted(xn, xx, side="right")) - 1
            hh = xn[i + 1] - xn[i]
            ww = (xx - xn[i]) / hh
            return ((2.0 * ww ** 3 - 3.0 * ww ** 2 + 1.0) * yn[i] + (ww ** 3 - 2.0 * ww ** 2 + ww) * hh * dk[i]
                    + (-2.0 * ww ** 3 + 3.0 * ww ** 2) * yn[i + 1] + (ww ** 3 - ww ** 2) * hh * dk[i + 1])

        def sp(pw, nt):
            r = float(nt)
            if nt < 0:
                return 0.0
            if pw == 0:
                return r + 1.0
            if pw == 1:
                return 0.5 * r * (r + 1.0)
            if pw == 2:
                return r * (r + 1.0) * (2.0 * r + 1.0) / 6.0
            return (0.5 * r * (r + 1.0)) ** 2

        def gsum(ka, kb):
            # sum of gmod over ka .. kb in closed form: the linear continuations and each Hermite cubic
            tot = 0.0
            if kb < ka:
                return tot
            k2 = min(kb, _nint(xn[0]) - 1)
            if k2 >= ka:
                tot += float(k2 - ka + 1) * (yn[0] + dk[0] * (0.5 * float(ka + k2) - xn[0]))
            k1 = max(ka, _nint(xn[-1]))
            if kb >= k1:
                tot += float(kb - k1 + 1) * (yn[-1] + dk[-1] * (0.5 * float(k1 + kb) - xn[-1]))
            for i0 in range(nn - 1):
                k1 = max(ka, _nint(xn[i0]))
                k2 = min(kb, _nint(xn[i0 + 1]) - 1)
                if k2 < k1:
                    continue
                hh = xn[i0 + 1] - xn[i0]
                c0, c1_ = yn[i0], dk[i0]
                c2 = (-3.0 * yn[i0] - 2.0 * hh * dk[i0] + 3.0 * yn[i0 + 1] - hh * dk[i0 + 1]) / hh ** 2
                c3 = (2.0 * yn[i0] + hh * dk[i0] - 2.0 * yn[i0 + 1] + hh * dk[i0 + 1]) / hh ** 3
                ta_, tb_ = k1 - _nint(xn[i0]), k2 - _nint(xn[i0])
                tot += (c0 * (sp(0, tb_) - sp(0, ta_ - 1)) + c1_ * (sp(1, tb_) - sp(1, ta_ - 1))
                        + c2 * (sp(2, tb_) - sp(2, ta_ - 1)) + c3 * (sp(3, tb_) - sp(3, ta_ - 1)))
            return tot

        lo = min(max(klo, -6), int(kw[0]))
        hi = max(min(khi, m + int(np.ceil(8.0 * sl + 5.0))), int(kw[-1]) + 1)
        lp = -gsum(lo, m - 1)
        zth = t["recomb_Zth"]
        order = sorted(range(len(zth)), key=lambda j: zth[j])     # stable, as the insertion sort
        jp, adone = 0, not want_a
        al = np.zeros(len(zth))
        sa, cr, tap, tdp = 0.0, 0.0, 0.0, 0.0
        dd = np.zeros(len(COULOMB_DCHARGE))
        q1 = np.exp(-C1) if C1 < 745.0 else 0.0
        stop = np.exp(-TAIL_STOP)
        for k in range(lo, hi + 1):
            if k > lo:
                lp += gmod(k - 1)
            ef = lp + lnPm
            if not adone:
                while jp < len(zth) and not k <= zth[order[jp]]:
                    al[order[jp]] = sa
                    jp += 1
                if jp >= len(zth):
                    adone = True
            if ef < -745.0:
                continue
            p = np.exp(ef)
            ta = 0.0
            if not adone:
                ta = p * capture(k)
                sa += ta
            td = 0.0
            if want_d:
                if k > 0:
                    qk = q1 ** k
                    td = p * qk
                    dd[1] += td
                    dd[2] += td * qk
                    dd[3] += td * qk * qk
                elif k == 0:
                    dd[1] += p * (1.0 + sq)
                    dd[2] += p * (1.0 + 2.0 * sq)
                    dd[3] += p * (1.0 + 3.0 * sq)
                    cr += p * sq                                 # D(-1): 1 + sq against 1 + 0 C1
                else:
                    dd[1] += p * (1.0 - k * C1)
                    dd[2] += p * (1.0 - 2.0 * k * C1)
                    dd[3] += p * (1.0 - 3.0 * k * C1)
                    f = np.exp(k * C1) if k * C1 > -745.0 else 0.0   # electrons repelled
                    cr += p * (f - (1.0 + k * C1))
            if k > 0 and ((adone or (ta < tap and ta <= stop * sa))
                          and (not want_d or (td < tdp and td <= stop * dd[1]))):
                break
            tap, tdp = ta, td
        alpha = D = None
        if want_a:
            while jp < len(zth):
                al[order[jp]] = sa
                jp += 1
            alpha = al / np.sqrt(np.array([mm for _, mm, _ in RECOMB_IONS]))
        if want_d:
            dd[0] = 1.0 + C1 * zmean + cr
            D = np.maximum(dd, 1e-10)
        return True, alpha, D

    def _wide_moments(self, m, gm1, g0, sl):
        """<Z>, sigma and ln P(m) of a wide P(Z) from a model of g (RAMSES wide_moments, calima/charge-lg5 commit
        6). The Gaussian of the central pair g(m-1), g(m) puts <Z> ~0.5 charges low in diffuse gas and is off in
        sigma by up to 4 %: g bends over the width. g is also taken at m -+ h, h = max(2, round(1.5 sl)), PCHIP
        through the four nodes (a single cubic is thrown off by the kinks of g), linear beyond; ln P is its lattice
        sum, ln P(m + z) - ln P(m) ~ int_{-1/2}^{z-1/2} g (Euler-Maclaurin); the moments by 8-point Gauss-Hermite
        quadrature around its peak (the root of g(z - 1/2), by Newton), with importance weights. Returns (used,
        <Z>, sigma, ln P(m)); not used where m -+ h does not fit the bin's charges or there is no peak."""
        t = self.t
        klo, khi = _nint(t["Zmin"]), _nint(t["Zmax"])
        h = max(2, _nint(1.5 * sl))
        if m - h < klo or m + h + 1 > khi:
            return False, None, None, None
        xn = np.array([-float(h), -1.0, 0.0, float(h)])
        yn = np.array([self._gi(m - h), gm1, g0, self._gi(m + h)])
        dk = np.zeros(4)
        dk[0] = (yn[1] - yn[0]) / (xn[1] - xn[0])
        dk[3] = (yn[3] - yn[2]) / (xn[3] - xn[2])
        for i in (1, 2):
            s0 = (yn[i] - yn[i - 1]) / (xn[i] - xn[i - 1])
            s1 = (yn[i + 1] - yn[i]) / (xn[i + 1] - xn[i])
            if s0 * s1 > 0.0:
                w1 = 2.0 * (xn[i + 1] - xn[i]) + (xn[i] - xn[i - 1])
                w2 = (xn[i + 1] - xn[i]) + 2.0 * (xn[i] - xn[i - 1])
                dk[i] = (w1 + w2) / (w1 / s0 + w2 / s1)

        def coef(i):
            hh = xn[i + 1] - xn[i]
            c2 = (-3.0 * yn[i] - 2.0 * hh * dk[i] + 3.0 * yn[i + 1] - hh * dk[i + 1]) / hh ** 2
            c3 = (2.0 * yn[i] + hh * dk[i] - 2.0 * yn[i + 1] + hh * dk[i + 1]) / hh ** 3
            return c2, c3

        def pint(i, x):
            c2, c3 = coef(i)
            return yn[i] * x + dk[i] * x * x / 2.0 + c2 * x ** 3 / 3.0 + c3 * x ** 4 / 4.0

        ci = [0.0]
        for i in range(3):
            ci.append(ci[i] + pint(i, xn[i + 1] - xn[i]))

        def piece(x):
            return 0 if x <= xn[1] else (1 if x <= xn[2] else 2)

        def Gint(x):
            if x <= xn[0]:
                return yn[0] * (x - xn[0]) + 0.5 * dk[0] * (x - xn[0]) ** 2
            if x >= xn[3]:
                return ci[3] + yn[3] * (x - xn[3]) + 0.5 * dk[3] * (x - xn[3]) ** 2
            j = piece(x)
            return ci[j] + pint(j, x - xn[j])

        def gval(x):
            if x <= xn[0]:
                return yn[0] + dk[0] * (x - xn[0]), dk[0]
            if x >= xn[3]:
                return yn[3] + dk[3] * (x - xn[3]), dk[3]
            j = piece(x)
            c2, c3 = coef(j)
            tt = x - xn[j]
            return yn[j] + dk[j] * tt + c2 * tt * tt + c3 * tt ** 3, dk[j] + 2.0 * c2 * tt + 3.0 * c3 * tt * tt

        z = 0.5 - g0 / (g0 - gm1)
        for _ in range(30):
            gz, dgz = gval(z - 0.5)
            if not dgz < 0.0:
                break
            dz = min(max(-gz / dgz, -sl), sl)
            z = min(max(z + dz, xn[0] + 0.5), xn[3] + 0.5)
            if abs(dz) < 1e-10:
                break
        gz, dgz = gval(z - 0.5)
        if not dgz < 0.0 or abs(gz) > 1e-6:
            return False, None, None, None
        swd = np.sqrt(-1.0 / dgz)
        Lz = Gint(z - 0.5)
        zi = z + np.sqrt(2.0) * swd * GH_X
        lr = np.array([Gint(zz - 0.5) for zz in zi]) - Lz + GH_X ** 2
        lmax = lr.max()
        rs = mu = 0.0
        rw = np.zeros(GH_X.size)
        for i in range(GH_X.size):
            rw[i] = GH_W[i] * np.exp(lr[i] - lmax)
            rs += rw[i]
            mu += rw[i] * zi[i]
        mu /= rs
        var = 0.0
        for i in range(GH_X.size):
            var += rw[i] * (zi[i] - mu) ** 2
        var /= rs
        lnpm = Gint(-0.5) - (Lz + lmax + np.log(np.sqrt(2.0) * swd * rs))
        return True, float(m + mu), float(np.sqrt(max(var, 0.0))), float(lnpm)

    # -- LG5: the lattice closure (RAMSES rtgroups_solve_lg, calima/charge-lg5 commits 3, 5)
    def _irates(self, k):
        """The rates of the integer charge k (RAMSES eval_int_k), each evaluated on first use and kept for the cell:
        up, down; gam, lam (Gamma, Lambda); dup, ddown, dlam (d/d ln T); A, the capture coefficient of a z = 1 ion."""
        key = ("r", k)
        v = self._gcache.get(key)
        if v is None:
            v = self._gcache[key] = _ChargeRates(self, k)
        return v

    def _lg_pos(self, k):
        """g(k) > 0, i.e. Up(k) > Down(k + 1), both floored as in g (no logarithm)."""
        return max(self._ud(k)[0], 1e-300) > max(self._ud(k + 1)[1], 1e-300)

    def _lg_dg(self, k):
        """d g(k) / d ln T."""
        a, b = self._irates(k), self._irates(k + 1)
        return a["dup"] / max(a["up"], 1e-300) - b["ddown"] / max(b["down"], 1e-300)

    def _lg_mode(self, Z_guess):
        """The mode m of P(Z), the smallest charge with g(m) <= 0, bracketed on the integers from the guess (by
        steps 1, 2, 4, ... and bisection, after a secant jump; from the continuous root if the guess is outside
        (Zmin, Zmax - 1) or the bracket leaves RAMSES's buffers)."""
        t = self.t
        klo, khi = _nint(t["Zmin"]), _nint(t["Zmax"])
        nw, half = 7, NBUF // 2
        zg = -HUGE if Z_guess is None else float(Z_guess)
        m = None
        for att in (1, 2):
            if not t["Zmin"] < zg < t["Zmax"] - 1.0:
                zg = self._root(None if zg == -HUGE else zg, None)[0]
            kg = min(max(int(np.floor(zg)), klo), khi - 1)
            base = kg - half
            m = kg
            if self._lg_pos(kg) and not self._lg_pos(kg + 1):
                return kg + 1
            if kg + 1 < khi:
                g0, g1 = self._gi(kg), self._gi(kg + 1)
                if g1 < g0:
                    zr = kg - g0 / (g1 - g0)
                    zr = min(max(zr, float(kg - half + nw + 4)), float(kg + half - nw - 4))
                    kg = min(max(int(np.floor(zr)), klo), khi - 1)
            okb, found, step = True, False, 1
            if self._lg_pos(kg):
                lo, hi = kg, khi
                while True:
                    kk = min(kg + step, khi - 1)
                    if kk + 1 - base > NBUF - nw - 1:
                        okb = False
                        break
                    if not self._lg_pos(kk):
                        hi, found = kk, True
                        break
                    lo = kk
                    if kk == khi - 1:
                        break
                    step *= 2
                default = khi
            else:
                lo, hi = klo, kg
                while True:
                    kk = max(kg - step, klo)
                    if kk - base < nw + 1:
                        okb = False
                        break
                    if self._lg_pos(kk):
                        lo, found = kk, True
                        break
                    hi = kk
                    if kk == klo:
                        break
                    step *= 2
                default = klo
            if not okb:
                zg = -HUGE
                if att == 1:
                    continue
                return m
            if not found:
                return default
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if self._lg_pos(mid):
                    lo = mid
                else:
                    hi = mid
            return hi
        return m

    def _solve_lattice(self, Z_guess, want_a, want_tail_a, wD, want_d, wide_moments=True):
        """LG5 (RAMSES rtgroups_solve_lg): the full rates at the 5 integer charges around the mode m of P(Z), 7 when
        the central width sigma_l = (-1/(g(m) - g(m-1)))^1/2 is in [LG_SIGMA7, SIGMA_DISCRETE), instead of a root,
        a chord width and a window grown to its tails.
          - sigma_l < SIGMA_DISCRETE: P(Z) exact on the window, its tails from g continued linearly from the two
            increments at each edge (down to LG_EPS of the peak); Gamma, Lambda over it (continued linearly in Z
            on the tails); the Coulomb factors D and alpha over it; d/d ln T analytic;
          - else (wide): the Gaussian of the central pair (mean Z* + 1/2, Z* = m - g(m)/s, sigma^2 = -1/s), Gamma
            and Lambda linear between the integers around the mean; alpha the Gaussian estimate; D the Gaussian
            ones (D_abs False);
          - either: where the tail toward Z = 0 carries alpha (want_tail_a for a wide P(Z)) or D, those from it
            (_tail_sums).
        Returns the result dict of solve (D, D_abs, alpha_rec only where asked)."""
        t = self.t
        klo, khi = _nint(t["Zmin"]), _nint(t["Zmax"])
        m = self._lg_mode(Z_guess)
        hw = 2
        while True:
            nw = 2 * hw + 1
            k0 = min(max(m - hw, klo), khi - nw + 1)
            gg = [self._gi(k0 + j) for j in range(nw - 1)]
            dg = [self._lg_dg(k0 + j) for j in range(nw - 1)] if want_d else None
            tm = min(max(m - k0, 1), nw - 2)
            sc = gg[tm] - gg[tm - 1]
            if not sc < 0.0:
                sc = -1e-6
            sl = np.sqrt(-1.0 / sc)
            if hw == 2 and LG_SIGMA7 <= sl < SIGMA_DISCRETE:
                hw = 3
                continue
            break
        R = lambda k: self._irates(k)    # noqa: E731
        out = dict(discrete=bool(sl < SIGMA_DISCRETE), Lambda_auto=0.0, lg_window=nw)
        kw = [k0 + j for j in range(nw - 1)]
        if not out["discrete"]:
            gb, mz = gg[tm], float(k0 + tm)
            mu, var = 0.5 - gb / sc, -1.0 / sc
            out["Zmean"] = mz + mu
            out["Zsigma"] = float(np.sqrt(var))
            out["Zstar"] = out["Zmean"] - 0.5
            # the closure's P at the mode: the lattice distribution j g(m) + s j (j - 1)/2, summed as its Gaussian
            jst = 0.5 - gb / sc
            lmx = jst * gb + 0.5 * sc * jst * (jst - 1.0)
            lnpm = -lmx - 0.5 * np.log(2.0 * np.pi / (-sc))
            if wide_moments:          # <Z>, sigma and P(m) of a model of g over the width; Z* stays the Gaussian's
                used, zm, zs, lnpm_w = self._wide_moments(k0 + tm, gg[tm - 1], gb, sl)
                if used:
                    out["Zmean"], out["Zsigma"], lnpm = zm, zs, lnpm_w
            jt = min(max(int(np.floor(out["Zmean"])) - k0, 0), nw - 2)
            fr = out["Zmean"] - float(k0 + jt)
            r0, r1 = R(k0 + jt), R(k0 + jt + 1)
            out["Gamma"] = max((1.0 - fr) * r0["gam"] + fr * r1["gam"], 0.0)
            out["Lambda"] = (1.0 - fr) * r0["lam"] + fr * r1["lam"]
            if want_d:
                dgb, dsc = dg[tm], dg[tm] - dg[tm - 1]
                dmu = -dgb / sc + gb * dsc / (sc * sc)
                out.update(dZmean_dlnT=dmu, dZsigma_dlnT=0.5 * (dsc / (sc * sc)) / out["Zsigma"],
                           dGamma_dlnT=(r1["gam"] - r0["gam"]) * dmu,
                           dLambda_dlnT=(r1["lam"] - r0["lam"]) * dmu + (1.0 - fr) * r0["dlam"] + fr * r1["dlam"])
            used, at, Dt = self._tail_sums(m, sl, kw, gg, lnpm, out["Zmean"], want_tail_a, wD)
            out["D_abs"] = bool(used and wD)
            if out["D_abs"]:
                out["D"] = Dt
            if want_tail_a:
                out["alpha_rec"] = at if used else np.exp(self._recombination_gauss(out["Zstar"], out["Zsigma"]))
            out["alpha_tail"] = bool(used and want_tail_a)
            out["nodes"], out["weights"] = np.array([out["Zmean"]]), np.array([1.0])
            return out
        # narrow: P exact on the window, tails from the edge increments
        slo = shi = sc
        if nw >= 4:
            slo, shi = gg[1] - gg[0], gg[nw - 2] - gg[nw - 3]
            slo = slo if slo < 0.0 else sc
            shi = shi if shi < 0.0 else sc
        dslo = dshi = 0.0
        if want_d:
            dslo = dg[1] - dg[0]
            dshi = dg[nw - 2] - dg[nw - 3] if nw >= 4 else dslo
        lp, cc = {0: 0.0}, {0: 0.0}
        for j in range(nw - 1):
            lp[j + 1] = lp[j] + gg[j]
            if want_d:
                cc[j + 1] = cc[j] + dg[j]
        pm = max(lp[j] for j in range(nw))
        pr = {j: np.exp(lp[j] - pm) for j in range(nw)}
        jhi = nw - 1
        if k0 + jhi < khi:
            rr, er = np.exp(gg[nw - 2] + shi), np.exp(shi)
            for n in range(1, LG_TAIL + 1):
                if k0 + jhi + 1 > khi or (pr[jhi] < LG_EPS and rr < 1.0):
                    break
                pr[jhi + 1] = pr[jhi] * rr
                if want_d:
                    cc[jhi + 1] = cc[jhi] + dg[nw - 2] + dshi * n
                jhi += 1
                rr *= er
        jlo = 0
        if k0 > klo:
            rr, er = np.exp(-gg[0] + slo), np.exp(slo)
            for n in range(1, LG_TAIL + 1):
                if k0 + jlo - 1 < klo or (pr[jlo] < LG_EPS and rr < 1.0):
                    break
                pr[jlo - 1] = pr[jlo] * rr
                if want_d:
                    cc[jlo - 1] = cc[jlo] - (dg[0] - dslo * n)
                jlo -= 1
                rr *= er
        ssum = 0.0
        for j in range(jlo, jhi + 1):
            ssum += pr[j]
        for j in range(jlo, jhi + 1):
            pr[j] = pr[j] / ssum

        def rates_at(j):
            # Gamma, Lambda (and dLambda / d ln T) of window index j, continued linearly beyond the window
            if j < 0:
                a, b, x = R(k0), R(k0 + 1), float(j)
                return (max(a["gam"] + (b["gam"] - a["gam"]) * x, 0.0), a["lam"] + (b["lam"] - a["lam"]) * x,
                        a["dlam"] + (b["dlam"] - a["dlam"]) * x)
            if j > nw - 1:
                a, b, x = R(k0 + nw - 1), R(k0 + nw - 2), float(j - (nw - 1))
                return (max(a["gam"] + (a["gam"] - b["gam"]) * x, 0.0), a["lam"] + (a["lam"] - b["lam"]) * x,
                        a["dlam"] + (a["dlam"] - b["dlam"]) * x)
            a = R(k0 + j)
            return a["gam"], a["lam"], a["dlam"]

        mu = m2 = gam = lam = 0.0
        rat = {}
        for j in range(jlo, jhi + 1):
            x = float(k0 + j)
            mu += pr[j] * x
            m2 += pr[j] * x * x
            rat[j] = rates_at(j)
            gam += pr[j] * rat[j][0]
            lam += pr[j] * rat[j][1]
        out.update(Zmean=mu, Zsigma=float(np.sqrt(max(m2 - mu * mu, 0.0))), Gamma=gam, Lambda=lam,
                   Zstar=float(k0 + tm) + gg[tm - 1] / (gg[tm - 1] - gg[tm]) - 1.0)
        dje = 0.0
        if k0 + jlo == klo and self.material != "cm20":      # cm20: no autoionisation, as _averages
            EA = self._electron_affinity(t["Zmin"])
            arrive = np.pi * self.a ** 2 * self.cell["ne"] * self.cell["v_e"]
            je, dje = _ds87_J_dlnT(t["Zmin"], -1.0, self.a * KB_CGS * self.cell["T"] / E_STATC ** 2)
            out["Lambda_auto"] = pr[jlo] * arrive * je * EA * EV2ERG
        C1 = E_STATC ** 2 / (KB_CGS * self.cell["T"] * self.a)
        sq = np.sqrt(np.pi * C1 / 2.0)
        if wD:
            # impactor charges -1, 1, 2, 3: attractive 1 + |k j| C1, neutral 1 + |j| sq, repulsive exp(-|k j| C1)
            q1 = np.exp(-C1) if C1 < 745.0 else 0.0
            D = np.zeros(len(COULOMB_DCHARGE))
            j0 = max(jlo, -k0 + 1)
            qk = 1.0
            for j in range(j0, jhi + 1):                   # k > 0
                k = k0 + j
                qk = q1 ** k if j == j0 else qk * q1
                D[0] += pr[j] * (1.0 + k * C1)
                D[1] += pr[j] * qk
                D[2] += pr[j] * qk * qk
                D[3] += pr[j] * qk * qk * qk
            j1 = min(jhi, -k0 - 1)
            for j in range(j1, jlo - 1, -1):               # k < 0
                k = k0 + j
                qk = q1 ** (-k) if j == j1 else qk * q1
                D[0] += pr[j] * qk
                D[1] += pr[j] * (1.0 - k * C1)
                D[2] += pr[j] * (1.0 - 2.0 * k * C1)
                D[3] += pr[j] * (1.0 - 3.0 * k * C1)
            if jlo <= -k0 <= jhi:                          # k = 0
                p0 = pr[-k0]
                D[0] += p0 * (1.0 + sq)
                D[1] += p0 * (1.0 + sq)
                D[2] += p0 * (1.0 + 2.0 * sq)
                D[3] += p0 * (1.0 + 3.0 * sq)
            out.update(D=D, D_abs=True)
        if want_a:
            al = np.zeros(len(t["recomb_Zth"]))
            for j in range(jlo, jhi + 1):
                x = float(k0 + j)
                qa = R(k0 + j)["A"] if 0 <= j <= nw - 1 else float(self._capture_z1(x)[0])
                for jj, zth in enumerate(t["recomb_Zth"]):
                    if x <= zth:
                        al[jj] += pr[j] * qa
            out["alpha_rec"] = al / np.sqrt(np.array([mm for _, mm, _ in RECOMB_IONS]))
        used, at, Dt = self._tail_sums(m, sl, kw, gg, np.log(max(pr[m - k0], 1e-300)), out["Zmean"], want_a, wD)
        if used:
            if want_a:
                out["alpha_rec"] = at
            if wD:
                out["D"] = Dt
        out["alpha_tail"] = bool(used and want_a)
        if want_d:
            s0 = 0.0
            for j in range(jlo, jhi + 1):
                s0 += pr[j] * cc[j]
            for j in range(jlo, jhi + 1):
                cc[j] -= s0
            dmu = dvar = dgam = dlam = 0.0
            for j in range(jlo, jhi + 1):
                x = float(k0 + j)
                dmu += pr[j] * cc[j] * x
                dvar += pr[j] * cc[j] * x * x
                dgam += pr[j] * cc[j] * rat[j][0]
                dlam += pr[j] * (cc[j] * rat[j][1] + rat[j][2])
            if out["Lambda_auto"] != 0.0:
                dlam += out["Lambda_auto"] * (cc[jlo] + 0.5 + dje)
            dvar -= 2.0 * out["Zmean"] * dmu
            out.update(dZmean_dlnT=dmu, dZsigma_dlnT=0.5 * dvar / out["Zsigma"] if out["Zsigma"] > 0 else 0.0,
                       dGamma_dlnT=dgam, dLambda_dlnT=dlam)
        out["nodes"] = np.arange(k0 + jlo, k0 + jhi + 1, dtype=float)
        out["weights"] = np.array([pr[j] for j in range(jlo, jhi + 1)])
        return out

    def _solve_lg(self, Z_guess, warm, recomb, dlnT, coulomb, wide_moments=True):
        """solve(lattice=True): LG5 (_solve_lattice), started from the warm Z* if given (as dust_interface), else
        Z_guess. Its alpha: over the narrow P(Z) (and its tail); for a wide one, recomb "gauss" the tail's or the
        Gaussian estimate, "fit" / "exact" as solve does. Its D: its own (D_abs), or for a wide P(Z) whose tail
        does not carry them the Gaussian factors of <Z>, sigma."""
        from pycalima.models.dust_charge import coulomb_focusing as cf

        zg = warm["Zstar"] if warm is not None else Z_guess
        out = self._solve_lattice(zg, want_a=recomb is not None, want_tail_a=recomb == "gauss", wD=bool(coulomb),
                                  want_d=dlnT is not None, wide_moments=wide_moments)
        if recomb is None:
            out["alpha_rec"] = None
        elif not out["discrete"] and recomb == "exact":
            out["alpha_rec"] = self._recombination_exact(out["Zstar"], out["Zsigma"])
        elif not out["discrete"] and recomb != "gauss":
            self.refine_recombination(out, warm)
        if coulomb:
            out["cratio"] = None
            if not out.get("D_abs", False):
                Dg = cf.coulomb_factors(out["Zmean"], out["Zsigma"], self.cell["T"], self.a,
                                        COULOMB_JMAX)[[j + 1 for j in COULOMB_DCHARGE]]
                out.update(D=np.maximum(Dg, cf.D_FLOOR), D_abs=False)
        else:
            out.pop("D", None)
            out.pop("D_abs", None)
        return out

    def solve(self, N_g, c_red, T, ne, ions=(), G0_bg=0.0, Z_guess=None, recomb="fit", warm=None, dlnT=None,
              coulomb=False, lattice=False, wide_moments=True):
        """Equilibrium charge and rates for one cell.

        N_g : photon densities per group [cm^-3]; c_red : RT speed of light
        [cm/s]; T [K]; ne [cm^-3]; ions : iterable of (n_i [cm^-3], z_i,
        m_i [g]); G0_bg : UV background in Habing units. Z_guess : start of
        the search (RAMSES passes the bin's previous Z*, or the uniform-ISRF table's <Z>); None, or a value
        outside (Zmin, Zmax - 1), searches the whole range. warm : the previous
        result for this bin (e.g. the last cooling sub-step), the start of the search instead. The result does not
        depend on the start (_distribution), except for the chord of a wide P(Z) whose root is at an end of the
        charge range. recomb: how the
        recombination of a wide P(Z) is summed, "fit" (_recombination_fit) or
        "exact" (every integer charge, the reference), "gauss" (the Gaussian
        estimate, _recombination_gauss), or None to skip it.
        With warm, the recombination fit of a wide P(Z) (``recomb_ref`` in the
        result: inputs, alpha and Gaussian estimate where it was fitted) is
        reused, times the change of the Gaussian estimate, while every input
        it depends on is within RECOMB_REUSE_TOL (in ln) of the fitted state
        and the estimate of every resolved alpha moved by less than
        RECOMB_REUSE_EPS.

        dlnT : if given (e.g. 1e-5), also return the d/d ln T of Zmean, Zsigma,
        Gamma and Lambda + Lambda_auto (_dlnT_response), which serve the cooling
        solver's T (1 +- dlnT) calls without solving again. coulomb : also return the Coulomb factors
        ``D`` for the impactor charges COULOMB_DCHARGE (-1, 1, 2, 3), as RAMSES dust_interface takes them: the
        Gaussian factors of <Z>, sigma (coulomb_focusing) times ``cratio``, the exact-P(Z) correction of a
        discrete P(Z) (_coulomb_ratio; 1 for a wide one), floored at 1e-10; ``D_abs`` is whether D is the
        solve's own (not yet: False). Returns a dict with
        Zmean, Zsigma, Gamma, Lambda and Lambda_auto [erg/s per grain], Zstar,
        whether P(Z) was discrete, alpha_rec: the grain-assisted recombination
        coefficients of RECOMB_IONS [cm^3 s^-1 per grain], and the charge nodes
        and weights over which Gamma and Lambda were averaged.
        """
        self._set_cell(N_g, c_red, T, ne, ions, G0_bg)
        self._b_base = None
        if lattice:
            return self._solve_lg(Z_guess, warm, recomb, dlnT, coulomb, wide_moments)
        d = self._distribution(Z_guess, warm)
        Zs, discrete, nodes, weights = d["Zstar"], d["discrete"], d["nodes"], d["weights"]
        shift, tail = 0.0, (False, None, None)
        if discrete:
            zmean, zsig = _window_moments(nodes, weights)
        else:
            # a wide P(Z): its mean and width from a model of g over the width (_wide_moments; else the Gaussian of
            # Z* + 1/2 and the chord's sigma), the rates there; alpha and D from its tail where it carries them
            sigma = float(d["sigma"])
            zmean, zsig = Zs + 0.5, sigma
            if wide_moments or recomb == "gauss" or coulomb:
                mq = min(max(int(np.ceil(Zs)), _nint(self.t["Zmin"]) + 2), _nint(self.t["Zmax"]) - 3)
                kw = [mq - 2, mq - 1, mq, mq + 1]
                gw = [self._gi(k) for k in kw]
                lnpm = -0.5 * ((float(mq) - Zs - 0.5) / sigma) ** 2 - np.log(np.sqrt(2.0 * np.pi) * sigma)
                if wide_moments:
                    used, zm, zs, lnpm_w = self._wide_moments(mq, gw[1], gw[2], sigma)
                    if used:
                        shift, zmean, zsig, lnpm = zm - zmean, zm, zs, lnpm_w
                if (recomb == "gauss" or coulomb) and mq >= 1:
                    tail = self._tail_sums(mq, sigma, kw, gw, lnpm, zmean, recomb == "gauss", coulomb)
            nodes, weights = np.array([zmean]), np.array([1.0])
        gam, lam, auto = self._averages(nodes, weights, discrete)
        out = dict(Zstar=Zs, discrete=discrete, Zmean=zmean, Zsigma=zsig, Gamma=gam, Lambda=lam,
                   Lambda_auto=auto, nodes=np.asarray(nodes, float), weights=np.asarray(weights, float))
        if recomb is None:
            out["alpha_rec"] = None
        elif discrete:
            out["alpha_rec"] = self._recombination_discrete(nodes, weights, z_floor=self._b_base + 1)
        elif recomb == "exact":
            out["alpha_rec"] = self._recombination_exact(Zs, float(d["sigma"]))
        elif recomb == "gauss":
            out["alpha_rec"] = tail[1] if tail[0] else np.exp(self._recombination_gauss(Zs, float(d["sigma"])))
            if warm is not None and "recomb_ref" in warm:
                out["recomb_ref"] = warm["recomb_ref"]       # kept for a later refine_recombination
        else:
            self.refine_recombination(out, warm)
        out["alpha_tail"] = bool(tail[0] and recomb == "gauss")
        if coulomb:
            from pycalima.models.dust_charge import coulomb_focusing as cf
            if tail[0]:
                out.update(cratio=None, D=tail[2], D_abs=True)
            else:
                cr = self._coulomb_ratio(out)
                Dg = cf.coulomb_factors(zmean, zsig, self.cell["T"], self.a,
                                        COULOMB_JMAX)[[j + 1 for j in COULOMB_DCHARGE]]
                out.update(cratio=cr, D=np.maximum(Dg * cr, cf.D_FLOOR), D_abs=False)
        if dlnT is not None:
            out.update(self._dlnT_response(out, d["pair"], dlnT, d["chord"], d["sigma"], shift))
        return out


class _ChargeRates:
    """The rates of one integer charge of a GroupChargingTable's cell, each evaluated on first use (LG5 reads only
    what its call needs: Up and Down for the bracket, Gamma and Lambda on its window, d/d ln T and A when asked)."""

    __slots__ = ("gt", "k", "v")

    def __init__(self, gt, k):
        self.gt, self.k, self.v = gt, k, {}

    def __getitem__(self, key):
        v = self.v
        if key not in v:
            gt, z = self.gt, float(self.k)
            if key in ("up", "down"):
                v["up"], v["down"] = gt._ud(self.k)
            elif key in ("gam", "lam"):
                v["gam"], v["lam"] = gt._heating_cooling(z)
            elif key in ("dup", "ddown", "dlam"):
                _, v["dup"], _, v["ddown"], _, v["dlam"] = gt._rates_dlnT(z)
            elif key == "A":
                v["A"] = float(gt._capture_z1(z)[0])
            else:
                raise KeyError(key)
        return v[key]


def _nint(x):
    """Fortran nint: the nearest integer, halves away from zero."""
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def _window_moments(Z, P):
    """<Z> and sigma of a discrete P(Z), summed in charge order as RAMSES window_moments."""
    zmean = s2 = 0.0
    for z, p in zip(Z, P):
        zmean += p * z
        s2 += p * z * z
    return float(zmean), float(np.sqrt(max(s2 - zmean * zmean, 0.0)))


def _log_ndtr(x):
    """ln of the standard normal CDF (erfcx in the far lower tail)."""
    from scipy.special import erfcx
    from math import erfc, log, sqrt
    if x < -5.0:
        return log(0.5 * erfcx(-x / sqrt(2.0))) - 0.5 * x * x
    return log(0.5 * erfc(-x / sqrt(2.0)))


def _wide_nodes(Zs):
    """A wide (Gaussian) P(Z) is averaged at its mean Z* + 1/2: Gamma and Lambda vary
    little over sigma (against 3 Gauss-Hermite nodes: median 0.1-3%, worst ~20%)."""
    return np.array([Zs + 0.5]), np.array([1.0])


def _ds87_J_dlnT(Z, q, tau):
    """DS87 J-tilde (theta_nu = nu/(1+nu^-1/2)) at nu = Z/q and d ln J / d ln tau."""
    nu = Z / q
    if nu == 0:
        r = np.sqrt(np.pi / (2.0 * tau))
        return 1.0 + r, -0.5 * r / (1.0 + r)
    if nu < 0:
        A, s = 1.0 - nu / tau, np.sqrt(2.0 / (tau - 2.0 * nu))
        return A * (1.0 + s), (nu / tau) / A - 0.5 * s * tau / (tau - 2.0 * nu) / (1.0 + s)
    th, s = nu / (1.0 + 1.0 / np.sqrt(nu)), 1.0 / np.sqrt(4.0 * tau + 3.0 * nu)
    return (1.0 + s) ** 2 * np.exp(-th / tau), -4.0 * tau * s ** 3 / (1.0 + s) + th / tau


def _ds87_lambda_dlnT(Z, tau):
    """DS87 Lambda-tilde for electrons at grain charge Z and d ln Lambda / d ln tau."""
    nu = -Z
    if nu == 0:
        r = np.sqrt(np.pi / (2.0 * tau))
        L = 2.0 + 1.5 * r
        return L, -0.75 * r / L
    if nu < 0:
        A, s = 2.0 - nu / tau, 1.0 / np.sqrt(tau - nu)
        return A * (1.0 + s), (nu / tau) / A - 0.5 * tau * s ** 3 / (1.0 + s)
    th, A, s = nu / (1.0 + 1.0 / np.sqrt(nu)), 2.0 + nu / tau, 1.0 / np.sqrt(1.5 / tau + 3.0 * nu)
    return A * (1.0 + s) * np.exp(-th / tau), (-nu / tau) / A + 0.75 * s ** 3 / tau / (1.0 + s) + th / tau


def solve_cell(gts, n_grains, N_g, c_red, T, ne, ions=(), G0_bg=0.0, Z_guess=None, warms=None, dlnT=None):
    """All dust bins of one cell, as RAMSES dust_interface: each bin solved with the
    Gaussian estimate of the recombination of a wide P(Z), then the fit (with its
    warm reuse) only for the wide bins that carry more than RECOMB_IMPORTANT of
    the cell's grain-assisted recombination of some ion. gts: GroupChargingTable
    per bin; n_grains: grain number density per bin (only ratios matter);
    Z_guess, warms: per bin (or None). Each result has "refined": fitted or reused."""
    nb = len(gts)
    Z_guess = [None] * nb if Z_guess is None else Z_guess
    warms = [None] * nb if warms is None else warms
    res = [gt.solve(N_g, c_red, T, ne, ions, G0_bg=G0_bg, Z_guess=zg, recomb="gauss", warm=w, dlnT=dlnT)
           for gt, zg, w in zip(gts, Z_guess, warms)]
    rate = np.array([n * r["alpha_rec"] for n, r in zip(n_grains, res)])
    tot = rate.sum(axis=0)
    share = np.divide(rate, tot, out=np.zeros_like(rate), where=tot > 0)
    for gt, r, w, sh in zip(gts, res, warms, share):
        r["refined"] = bool((not r["discrete"]) and sh.max() > RECOMB_IMPORTANT)
        if r["refined"]:
            r["fitted"] = gt.refine_recombination(r, w)
    return res


def _pchip_slopes(y):
    """Fritsch-Carlson monotone slopes on a unit-spaced grid (last axis)."""
    y = np.asarray(y, dtype=float)
    d = np.diff(y, axis=-1)
    m = np.zeros_like(y)
    prod = d[..., :-1] * d[..., 1:]
    den = np.where(prod > 0, d[..., :-1] + d[..., 1:], 1.0)
    m[..., 1:-1] = np.where(prod > 0, 2.0 * d[..., :-1] * d[..., 1:] / den, 0.0)
    m[..., 0], m[..., -1] = d[..., 0], d[..., -1]
    return m


def _cheb_coef(f, a, b, n):
    """Chebyshev coefficients of the degree n-1 interpolant of f on [a, b]."""
    k = np.arange(n)
    x = np.cos(np.pi * (k + 0.5) / n)
    y = np.array([f(0.5 * (a + b) + 0.5 * (b - a) * xi) for xi in x])
    c = np.array([2.0 / n * np.sum(y * np.cos(np.pi * j * (k + 0.5) / n)) for j in range(n)])
    c[0] *= 0.5
    return c


class _GPiece:
    """g(Z) on the integers [p, q]: exact values, or Chebyshev coefficients of g
    (with its antiderivative and derivative) and of ln A."""

    def __init__(self, p, q, vals=None, coef=None, coefA=None):
        self.p, self.q, self.vals, self.c, self.cA = p, q, vals, coef, coefA
        if coef is not None:
            self.ci = np.polynomial.chebyshev.chebint(coef)
            self.cd = np.polynomial.chebyshev.chebder(coef)

    def _t(self, z):
        return (2.0 * z - self.p - self.q) / (self.q - self.p)

    def dg(self, z):
        return float(2.0 / (self.q - self.p) * np.polynomial.chebyshev.chebval(self._t(z), self.cd))

    def lnA(self, z):
        return float(np.polynomial.chebyshev.chebval(self._t(z), self.cA))

    def gsum(self, m):
        """sum_{j=p}^{m-1} g(j), p <= m <= q + 1 (Euler-Maclaurin for fits)."""
        if self.vals is not None:
            return float(np.sum(self.vals[: int(m - self.p)]))
        if m == self.p:
            return 0.0
        a, b = self.p - 0.5, m - 0.5
        G = lambda z: 0.5 * (self.q - self.p) * np.polynomial.chebyshev.chebval(self._t(z), self.ci)  # noqa: E731
        return float(G(b) - G(a) - (self.dg(b) - self.dg(a)) / 24.0)


def _geometric_sum(Zn, lnf, zmax):
    """sum over the integers Zn[0] .. min(zmax, Zn[-1]) of exp(lnf), with ln f
    linear between the nodes Zn (exact on unit steps)."""
    tot = 0.0
    for k in range(len(Zn) - 1):
        z0, z1 = Zn[k], Zn[k + 1]
        if z0 > zmax:
            break
        d = z1 - z0
        m = min(d, zmax - z0 + 1)
        lr = (lnf[k + 1] - lnf[k]) / d
        if abs(lr) < 1e-12:
            tot += m * np.exp(lnf[k])
        elif lr < 0:
            tot += np.exp(lnf[k]) * np.expm1(lr * m) / np.expm1(lr)
        else:
            tot += np.exp(lnf[k] + lr * (m - 1)) * np.expm1(-lr * m) / np.expm1(-lr)
    if Zn[-1] <= zmax:
        tot += np.exp(lnf[-1])
    return tot


def uniform_table(gt, G0):
    """(Zmean, Zsigma, Gamma, Lambda + Lambda_auto) on (UNIF_LT, UNIF_LNE) for a
    uniform background G0 and no local RT photons; ions are H+ with n = n_e."""
    nG = gt.t["L0"].size
    out = np.zeros((4, UNIF_LT.size, UNIF_LNE.size))
    for i, lt in enumerate(UNIF_LT):
        for j, lne in enumerate(UNIF_LNE):
            ne = 10.0 ** lne
            r = gt.solve(np.zeros(nG), 0.0, 10.0 ** lt, ne, [(ne, 1.0, M_HION)], G0_bg=G0)
            out[:, i, j] = (r["Zmean"], r["Zsigma"], r["Gamma"], r["Lambda"] + r["Lambda_auto"])
    return out


def uniform_lookup(tab, T, ne):
    """Bilinear interpolation of ``uniform_table`` in (log T, log n_e), clamped."""
    x = min(max(np.log10(T), UNIF_LT[0]), UNIF_LT[-1])
    y = min(max(np.log10(max(ne, 1e-300)), UNIF_LNE[0]), UNIF_LNE[-1])
    fx = (x - UNIF_LT[0]) / (UNIF_LT[1] - UNIF_LT[0])
    fy = (y - UNIF_LNE[0]) / (UNIF_LNE[1] - UNIF_LNE[0])
    i = min(int(fx), UNIF_LT.size - 2)
    j = min(int(fy), UNIF_LNE.size - 2)
    u, v = fx - i, fy - j
    return ((1 - u) * (1 - v) * tab[:, i, j] + u * (1 - v) * tab[:, i + 1, j]
            + (1 - u) * v * tab[:, i, j + 1] + u * v * tab[:, i + 1, j + 1])


def _ds87_lambda_electron_arr(Z, a, T):
    """DS87_lambda_scalar(Z, -1, a, T) on an array of grain charges."""
    nu = -np.asarray(Z, float)
    tau = max(a * KB_CGS * T / E_STATC ** 2, 1e-300)
    neg = (2.0 - nu / tau) * (1.0 + 1.0 / np.sqrt(np.maximum(tau - nu, 1e-300)))
    nup = np.maximum(nu, 1e-300)
    theta = nup / (1.0 + 1.0 / np.sqrt(nup))
    pos = (2.0 + nup / tau) * (1.0 + 1.0 / np.sqrt(1.5 / tau + 3.0 * nup)) * np.exp(-theta / tau)
    return np.where(nu == 0.0, 2.0 + 1.5 * np.sqrt(np.pi / (2.0 * tau)), np.where(nu < 0.0, neg, pos))


def _ds87_lambda(Z, q, a, T):
    from pycalima.models.dust_charge.shared_physics import DS87_lambda_scalar
    return DS87_lambda_scalar(Z, q, a, T)
