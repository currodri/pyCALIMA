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
the same file, formulas and iteration, so the two agree to round-off.

Per-cell algorithm (``GroupChargingTable.solve``)
-------------------------------------------------
With Up(Z) = photoemission + ion capture + gas-electron secondaries and
Down(Z) = electron capture, detailed balance gives f(Z+1)/f(Z) = Up(Z)/Down(Z+1).

1. Root Z* of g(Z) = ln Up(Z) - ln Down(Z+1), decreasing in Z, on
   [Zmin, Zmax] by the Illinois method; with a starting guess (RAMSES: the
   bin's previous Z*), on a bracket grown from it in steps 1, 2, 4, ...
   Warm start (``warm``: the previous result, e.g. the last cooling sub-step):
   g at Z*_prev +- 1/2 gives Z* (root of the line through them, checked) and
   g' in ~3 evaluations; a discrete result's charge window is reused instead
   (step 3, no root search).
2. sigma^2 = -1 / g'(Z*) (Gaussian limit), g' by central difference; where that
   reads >= ``SIGMA_DISCRETE``, g' is taken instead from the chord of g over the
   integers within ~ +-sigma of Z* (``_chord``), which a kink of g near Z* (at Z ~ 0,
   or where a photoemission threshold crosses a group edge) does not throw off.
3. If sigma < ``SIGMA_DISCRETE`` the distribution is built exactly over integer
   charges around Z*, the window grown or trimmed at each end to one charge
   past where P < ``DISCRETE_TAIL`` of its maximum (warm: while its exact
   width is < ``SIGMA_DISCRETE``, see ``_warm_window``); otherwise it is Gaussian with mean Z* + 1/2, and charge
   averages are taken at its mean.
4. Heating Gamma = sum_g c N_g H_g(U); recombination cooling from DS87
   Lambda-tilde, scaled for electron pass-through, less the energy the
   gas-excited secondaries return; autoionisation cooling at Zmin.
5. Optionally (``dlnT``) the d/d ln T of <Z>, sigma_Z, Gamma and Lambda from one
   more set of rates at T (1 + dlnT) on the same charges, to serve the cooling
   solver's T (1 +- dlnT) calls without further solves.
6. Grain-assisted recombination alpha_i = sum_Z P(Z) pi a^2 v_i J~ Theta_i
   (``recomb``): exact for a discrete P(Z); for a wide one, the segmented
   Chebyshev reconstruction of ln P in ``_recombination_fit``, reused by warm
   solves as the fit times the change of a Gaussian estimate
   (``_recombination_gauss``) while the cell drifts little.

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

from pycalima.models.dust_charge import _charge_kernels as _ck
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
C_CGS = 2.99792458e10
HABING_FLUX = 1.6e-3   # erg cm^-2 s^-1 in 6-13.6 eV: G0 = 1 (RAMSES advected_G0)
E2_EV_CM = E_STATC ** 2 / EV2ERG      # e^2 in eV cm: U [V] = Z * E2_EV_CM / a
MATERIAL_CODE = {"graphite": 1, "silicate": 2}
SIGMA_DISCRETE = 3.0                  # below this, P(Z) is built exactly
MAX_DISCRETE_STATES = 121
DISCRETE_TAIL = 1e-5                  # a discrete P(Z) ends one charge past where P < this x max
SIGMA_CHORD_ITER = 3                  # at most this many chords for the width of a wide P(Z)
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


def build_bin_table(material, a, group_L0=DEFAULT_GROUP_L0, group_L1=DEFAULT_GROUP_L1,
                    U=None, Ts=TS_GRID, T_hot=THOT_GRID, n_E=N_E_PER_GROUP, label=""):
    """Table for one dust bin (dict of arrays; see ``write_bin_table``)."""
    material = "graphite" if material == "graphite" else "silicate"
    U = u_grid() if U is None else np.asarray(U, dtype=float)
    L0 = np.asarray(group_L0, dtype=float)
    L1 = np.asarray(group_L1, dtype=float)
    if np.any(L1 <= L0):
        raise ValueError("every group needs a finite upper edge above its lower one")
    Zs = U * a / E2_EV_CM
    Zmin = float(most_negative_allowed_charge_graphite(a) if material == "graphite"
                 else most_negative_allowed_charge_silicate(a))
    W = wy.MATERIALS[material]["W"]
    Zmax = float(most_positive_allowed_charge(a, W, float(L1.max())))
    Zmax = min(Zmax, np.floor(U[-1] * a / E2_EV_CM))
    Zmin = max(Zmin, np.ceil(U[0] * a / E2_EV_CM))

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

    # UV background: Mathis ISRF at G0 = 1 (Habing flux in 6-13.6 eV)
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field
    Eb = np.geomspace(1.0, 13.6, 400)
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

    return dict(version=FORMAT_VERSION, label=label, material=material, a=float(a), W=float(W),
                Zmin=Zmin, Zmax=Zmax, L0=L0, L1=L1, U=U, log10_Ts=np.log10(Ts), egy=egy, K=K, H=H,
                log10_Thot=np.log10(T_hot), delta=delta, P_tr=P_tr, f_cool=f_cool, E_bar=E_bar,
                K_bg=K_bg, H_bg=H_bg, recomb_Zth=recombination_thresholds(material, a, W, Zmin, Zmax))


def grain_ionisation_potential(material, a, W, Z):
    """IP(a, Z) [eV]: energy to remove an electron from a grain of charge Z
    (valence IP for Z >= 0, electron affinity of charge Z + 1 below)."""
    Z = np.asarray(Z, dtype=float)
    ea = electron_affinity_graphite_vec if material == "graphite" else electron_affinity_silicate_vec
    return np.where(Z >= 0, ionisation_potential_valence_vec(W, Z, a), ea(Z + 1.0, a))


def recombination_thresholds(material, a, W, Zmin, Zmax):
    """Largest grain charge Z at which X+ can take an electron from the grain,
    IP(a, Z) <= IP(X) (Weingartner & Draine 2001b case A), per RECOMB_IONS;
    Zmin - 1 where no charge allows it."""
    Z = np.arange(Zmin, Zmax + 1.0)
    ipz = grain_ionisation_potential(material, a, W, Z)
    out = []
    for _, _, ip in RECOMB_IONS:
        ok = np.nonzero(ipz <= ip)[0]
        out.append(float(Z[ok[-1]]) if ok.size else Zmin - 1.0)
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
    return tab


# ---------------------------------------------------------------------------
# Per-cell solver (mirrors RAMSES)
# ---------------------------------------------------------------------------

class GroupChargingTable:
    """One bin's table with the per-cell charge solve that RAMSES performs."""

    def __init__(self, tab, egy=None, kernels="linear"):
        """kernels: interpolation of the photoemission and heating kernels in U,
        "linear" (as RAMSES now) or "pchip" (monotone cubic; no kinks at the U
        nodes, needed for an accurate recombination fit of wide P(Z))."""
        if kernels not in ("linear", "pchip"):
            raise ValueError("kernels must be 'linear' or 'pchip'")
        self.t = tab
        self.a = tab["a"]
        self.material = tab["material"]
        self.kernels = kernels
        U = tab["U"]
        self.x0 = np.arcsinh(U[0] / U_SCALE)
        self.dx = (np.arcsinh(U[-1] / U_SCALE) - self.x0) / (U.size - 1)
        self.e2a = E2_EV_CM / self.a
        self.compiled = _ck.NUMBA     # g(Z) and its root in numba (_charge_kernels); False: Python
        self._nb = self._nbh = None
        # charges at which IP(a, Z) crosses photon-energy edges: g(Z) has kinks there
        Zall = np.arange(tab["Zmin"], tab["Zmax"] + 1.0)
        ipz = grain_ionisation_potential(self.material, self.a, tab["W"], Zall)
        def edge(E):
            k = np.nonzero(ipz <= E)[0]
            return float(Zall[k[-1]]) if k.size and 0 < k[-1] < Zall.size - 1 else None
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
        """WD01 eqs. 27-30 with the continuous extension used by the solver."""
        a = self.a
        if Z <= self.t["Zmin"]:
            return 0.0
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
        if self.compiled and self._nb is not None:
            return float(_ck.g_scalar(float(Z), *self._nb))
        up = self._up(Z, *self._u_index(Z * E2_EV_CM / self.a))
        down = self._down(Z + 1.0, *self._u_index((Z + 1.0) * E2_EV_CM / self.a))
        return np.log(max(up, 1e-300)) - np.log(max(down, 1e-300))

    def _heating_cooling(self, Z):
        c = self.cell
        a = self.a
        up, down, (i, w) = self._rates(Z)
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

    # -- the same on arrays of charges (exact_distribution, the WDB06tab nodes); the scalar
    #    versions above are what RAMSES mirrors
    def _u_index_arr(self, U):
        f = np.clip((np.arcsinh(U / U_SCALE) - self.x0) / self.dx, 0.0, self.t["U"].size - 1.0)
        i = np.minimum(f.astype(int), self.t["U"].size - 2)
        return i, f - i

    def _sticking_arr(self, Z):
        a = self.a
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
        if self.compiled and self._nb is not None:
            return _ck.g_array(np.ascontiguousarray(Z.ravel()), *self._nb).reshape(Z.shape)
        up = self._up_arr(Z, *self._u_index_arr(Z * E2_EV_CM / self.a))
        down = self._down_arr(Z + 1.0, *self._u_index_arr((Z + 1.0) * E2_EV_CM / self.a))
        return np.log(np.maximum(up, 1e-300)) - np.log(np.maximum(down, 1e-300))

    def _heating_cooling_arr(self, Z):
        c, a = self.cell, self.a
        Z = np.asarray(Z, float)
        if self.compiled and self._nb is not None:
            if self._nbh is None:
                self._nbh = _ck.heat_arrays(self)
            gam, lam = _ck.heating_cooling_array(np.ascontiguousarray(Z.ravel()), *self._nbh)
            return gam.reshape(Z.shape), lam.reshape(Z.shape)
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
        if self.compiled and self._nb is not None:
            return float(_ck.illinois(float(a_), float(b_), float(fa), float(fb), *self._nb))
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
        if self.compiled and self._nb is not None:
            return float(_ck.root_from_guess(float(Zg), float(Zlo), float(Zhi), *self._nb))
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
        self.cell = dict(cN=c_red * np.asarray(N_g, dtype=float), T=float(T), ne=float(ne), G0_bg=float(G0_bg),
                         v_e=np.sqrt(8 * KB_CGS * T / (np.pi * ME_CGS)),
                         ions=[(float(n), float(z), np.sqrt(8 * KB_CGS * T / (np.pi * m))) for n, z, m in ions])
        self._nb = _ck.cell_arrays(self) if self.compiled else None
        self._nbh = None                # heating arguments, built on first use

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
            kn = max(1, int(round(sigma)))
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

    def _averages(self, nodes, weights, discrete):
        """Gamma, Lambda (+ autoionisation at Zmin) averaged over the charge nodes."""
        gam = lam = 0.0
        for Z, wgt in zip(nodes, weights):
            g_, l_ = self._heating_cooling(Z)
            gam += wgt * g_
            lam += wgt * l_
        auto = 0.0
        if discrete and nodes[0] == self.t["Zmin"]:
            EA = float((electron_affinity_graphite_vec if self.material == "graphite"
                        else electron_affinity_silicate_vec)(self.t["Zmin"], self.a))
            arrive = np.pi * self.a ** 2 * self.cell["ne"] * self.cell["v_e"]
            auto = weights[0] * arrive * float(DS87_J_function_vec(self.t["Zmin"], -1.0, self.a, self.cell["T"])) * EA * EV2ERG
        return float(gam), float(lam), float(auto)

    def _distribution(self, Zs, sigma):
        """P(Z): exact over the integers around Z* (_window) if sigma < SIGMA_DISCRETE,
        else a Gaussian of mean Z* + 1/2, averaged at that mean (_wide_nodes). Returns
        nodes, weights, discrete, and the g increments of the discrete window."""
        if sigma < SIGMA_DISCRETE:
            half = int(np.ceil(3 * max(sigma, 0.5)))
            Zc = int(np.floor(Zs + 0.5))
            Zint, P, ginc = self._window(max(self.t["Zmin"], Zc - half), min(self.t["Zmax"], Zc + half))
            return Zint, P, True, ginc
        return _wide_nodes(Zs) + (False, None)

    def _window(self, z0, z1):
        """Exact P(Z) on the integers z0..z1, grown or trimmed at each end to one
        charge past where P < DISCRETE_TAIL of its maximum. Returns the charges,
        P, and the g increments between them."""
        Zlo, Zhi, ln_tail = self.t["Zmin"], self.t["Zmax"], np.log(DISCRETE_TAIL)
        Z = np.arange(float(z0), float(z1) + 1.0)
        ginc = [self._g(z) for z in Z[:-1]]
        while True:
            lnf = np.concatenate([[0.0], np.cumsum(ginc)])
            lnf -= lnf.max()
            room = Z.size < MAX_DISCRETE_STATES
            lo = room and lnf[0] > ln_tail and Z[0] > Zlo
            hi = room and lnf[-1] > ln_tail and Z[-1] < Zhi
            if not (lo or hi):
                break
            if lo:
                ginc.insert(0, self._g(Z[0] - 1.0))
                Z = np.r_[Z[0] - 1.0, Z]
            if hi:
                ginc.append(self._g(Z[-1]))
                Z = np.r_[Z, Z[-1] + 1.0]
        k = np.nonzero(lnf > ln_tail)[0]
        i0, i1 = max(k[0] - 1, 0), min(k[-1] + 1, Z.size - 1)
        P = np.exp(lnf[i0:i1 + 1])
        return Z[i0:i1 + 1], P / P.sum(), np.array(ginc[i0:i1])

    def _warm_window(self, warm):
        """A discrete result's window, regrown in this cell (no root search):
        (Z*, charges, P, g increments), or None once P(Z) is as wide as
        SIGMA_DISCRETE. Z* is the root of g interpolated between the integers.
        The width tested is the exact one, not (-1/g')^1/2 as in _distribution:
        next to the kink of g at Z ~ 0 the latter can read above SIGMA_DISCRETE
        for a narrower P(Z), where the Gaussian is off by ~1/2 in <Z>. So a
        discrete P(Z) stays discrete there, rather than flipping each sub-step."""
        Z, P, ginc = self._window(warm["nodes"][0], warm["nodes"][-1])
        if _width(Z, P) >= SIGMA_DISCRETE:
            return None
        j = np.nonzero((ginc[:-1] > 0) & (ginc[1:] <= 0))[0]
        if j.size:
            Zs = Z[j[0]] + ginc[j[0]] / (ginc[j[0]] - ginc[j[0] + 1])
        else:
            Zs = Z[0] if ginc.size == 0 or ginc[0] <= 0 else Z[-1]
        return float(Zs), Z, P, ginc

    @staticmethod
    def _moments(nodes, weights, discrete, Zs, sigma):
        if discrete:
            return float(np.sum(weights * nodes)), float(_width(nodes, weights))
        return float(Zs + 0.5), float(sigma)

    def _dlnT_response(self, out, pair, dlnT, chord=None):
        """d/d ln T of Zmean, Zsigma, Gamma, Lambda (+ auto). Discrete P(Z):
        analytic (_dlnT_discrete). Wide: one more set of rates at T (1 + dlnT),
        the pair g(z+-) re-evaluated, so the shifted root is consistent with the
        T call by construction, and the width from the same chord (or pair), then
        Gamma and Lambda re-averaged on the shifted Gaussian nodes."""
        if out["discrete"]:
            return self._dlnT_discrete(out)
        T0 = self.cell["T"]
        N_g, c_red, ne, ions, G0_bg = self._inputs
        self._set_cell(N_g, c_red, T0 * (1.0 + dlnT), ne, ions, G0_bg)
        zm, zp, gm, gpl = pair
        gm1, gp1 = self._g(zm), self._g(zp)
        s0, s1 = (gpl - gm) / (zp - zm), (gp1 - gm1) / (zp - zm)
        Zs1 = out["Zstar"] + (zm - gm1 / s1) - (zm - gm / s0)
        sig1 = np.sqrt(-1.0 / s1) if s1 < 0 else out["Zsigma"]
        if chord is not None:
            lo, hi = chord[:2]
            c1 = (self._g(hi) - self._g(lo)) / (hi - lo)
            sig1 = np.sqrt(-1.0 / c1) if c1 < 0 else out["Zsigma"]
        nodes1, weights1 = _wide_nodes(Zs1)
        gam1, lam1, _ = self._averages(nodes1, weights1, False)
        self._set_cell(N_g, c_red, T0, ne, ions, G0_bg)
        h = np.log1p(dlnT)
        return dict(dZmean_dlnT=(Zs1 - out["Zstar"]) / h, dZsigma_dlnT=(sig1 - out["Zsigma"]) / h,
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

    def charge_distribution(self, N_g, c_red, T, ne, ions=(), G0_bg=0.0, Z_guess=None):
        """The charge part of a cold solve: Zstar, discrete, Zmean, Zsigma and the charge nodes and
        weights, as solve(..., recomb=None) returns them, without the heating and cooling averages
        (for callers that evaluate those on the exact P(Z), e.g. the table builder)."""
        self._set_cell(N_g, c_red, T, ne, ions, G0_bg)
        Zs, sigma, _ = self._root(Z_guess, None)
        if sigma >= SIGMA_DISCRETE:
            sigma, _ = self._chord(Zs, sigma)
        nodes, weights, discrete, _ = self._distribution(Zs, sigma)
        zmean, zsig = self._moments(nodes, weights, discrete, Zs, sigma)
        return dict(Zstar=Zs, discrete=discrete, Zmean=zmean, Zsigma=zsig,
                    nodes=np.asarray(nodes, float), weights=np.asarray(weights, float))

    def solve(self, N_g, c_red, T, ne, ions=(), G0_bg=0.0, Z_guess=None, recomb="fit", warm=None, dlnT=None,
              sens=False):
        """Equilibrium charge and rates for one cell.

        N_g : photon densities per group [cm^-3]; c_red : RT speed of light
        [cm/s]; T [K]; ne [cm^-3]; ions : iterable of (n_i [cm^-3], z_i,
        m_i [g]); G0_bg : UV background in Habing units. Z_guess : start of
        the root search (RAMSES passes the bin's previous Z*); None, or a value
        outside (Zmin, Zmax - 1), searches the whole range. warm : the previous
        result for this bin (e.g. the last cooling sub-step): the root then costs
        two g evaluations (_root), or none if P(Z) was discrete (_warm_window). recomb: how the
        recombination of a wide P(Z) is summed, "fit" (_recombination_fit) or
        "exact" (every integer charge, the reference), "gauss" (the Gaussian
        estimate, _recombination_gauss), or None to skip it.
        With warm, the recombination fit of a wide P(Z) (``recomb_ref`` in the
        result: inputs, alpha and Gaussian estimate where it was fitted) is
        reused, times the change of the Gaussian estimate, while every input
        it depends on is within RECOMB_REUSE_TOL (in ln) of the fitted state
        and the estimate of every resolved alpha moved by less than
        RECOMB_REUSE_EPS.

        sens : also return the analytic d/d ln S and d/d ln N of Zmean, Zsigma, Gamma and
        Lambda + Lambda_auto (_sensitivities) and the inputs and kernels the predictor needs
        ("anchor"; predict), with dlnT (default 1e-5) for the T direction.
        dlnT : if given (e.g. 1e-5), also return the d/d ln T of Zmean, Zsigma,
        Gamma and Lambda + Lambda_auto (_dlnT_response), which serve the cooling
        solver's T (1 +- dlnT) calls without solving again. Returns a dict with
        Zmean, Zsigma, Gamma, Lambda and Lambda_auto [erg/s per grain], Zstar,
        whether P(Z) was discrete, alpha_rec: the grain-assisted recombination
        coefficients of RECOMB_IONS [cm^3 s^-1 per grain], and the charge nodes
        and weights over which Gamma and Lambda were averaged.
        """
        self._set_cell(N_g, c_red, T, ne, ions, G0_bg)
        dist = self._warm_window(warm) if warm is not None and warm["discrete"] else None
        chord = None
        if dist is None:
            Zs, sigma, pair = self._root(Z_guess, warm)
            if sigma >= SIGMA_DISCRETE:
                # warm and wide: start the chord from the last width, usually one chord
                start = warm["Zsigma"] if warm is not None and not warm["discrete"] else sigma
                sigma, chord = self._chord(Zs, start)
            nodes, weights, discrete, _ = self._distribution(Zs, sigma)
        else:
            (Zs, nodes, weights, _), sigma, discrete, pair = dist, None, True, None
        zmean, zsig = self._moments(nodes, weights, discrete, Zs, sigma)
        gam, lam, auto = self._averages(nodes, weights, discrete)
        out = dict(Zstar=Zs, discrete=discrete, Zmean=zmean, Zsigma=zsig, Gamma=gam, Lambda=lam,
                   Lambda_auto=auto, nodes=np.asarray(nodes, float), weights=np.asarray(weights, float))
        if recomb is None:
            out["alpha_rec"] = None
        elif discrete:
            out["alpha_rec"] = self._recombination(nodes, weights)
        elif recomb == "exact":
            out["alpha_rec"] = self._recombination_exact(Zs, sigma)
        elif recomb == "gauss":
            out["alpha_rec"] = np.exp(self._recombination_gauss(Zs, sigma))
            if warm is not None and "recomb_ref" in warm:
                out["recomb_ref"] = warm["recomb_ref"]       # kept for a later refine_recombination
        else:
            self.refine_recombination(out, warm)
        if sens and dlnT is None:
            dlnT = 1e-5
        if dlnT is not None:
            out.update(self._dlnT_response(out, pair, dlnT, chord))
        if sens:
            out.update(self._sensitivities(out))
            out["anchor"] = self._anchor(out)
        return out

    def _sensitivities(self, out):
        """d/d ln S (every photon density, local and background, x e^theta) and d/d ln N (n_e and
        every ion density x e^theta) of Zmean, Zsigma, Gamma and Lambda + Lambda_auto, analytic:
        the predictor's linearisation (predict). Photoemission and the heating scale with S; the
        ion capture, the gas-electron secondaries, electron capture, the cooling and the
        autoionisation with N. Discrete P(Z): d ln P(Z_k) is the cumulative sum of
        d ln Up(Z_j) - d ln Down(Z_j + 1), centred on its mean (as _dlnT_discrete). Wide: the root
        moves by -(dg/dtheta)/(dg/dZ), with dg/dZ = -1/sigma^2, sigma by the change of that slope
        across the pair Z* -+ 1/2, and Gamma, Lambda (at Z* + 1/2) along their slope in Z."""
        a = self.a
        mK, mKb = getattr(self, "_mK", None), getattr(self, "_mKb", None)
        res = {}
        if out["discrete"]:
            Z, P = np.asarray(out["nodes"], float), np.asarray(out["weights"], float)
            i, w = self._u_index_arr(Z * E2_EV_CM / a)
            up = np.maximum(self._up_arr(Z, i, w), 1e-300)
            photo = self._photo_arr(self.Kg, mK, self.t["K_bg"], mKb, i, w)
            gam, lam = self._heating_cooling_arr(Z)
            auto = out.get("Lambda_auto", 0.0)
            for key, fup, fdn, fgam, flam in (("S", photo / up, 0.0, 1.0, 0.0), ("N", (up - photo) / up, 1.0, 0.0, 1.0)):
                c = np.concatenate([[0.0], np.cumsum(fup[:-1] - fdn)])
                c -= P @ c
                dmean = P @ (c * Z)
                dvar = P @ (c * Z * Z) - 2.0 * out["Zmean"] * dmean
                dlam = P @ (c * lam) + flam * (P @ lam)
                if auto:
                    dlam += auto * (c[0] + flam)
                res[f"dZmean_dln{key}"] = dmean
                res[f"dZsigma_dln{key}"] = 0.5 * dvar / out["Zsigma"] if out["Zsigma"] > 0 else 0.0
                res[f"dGamma_dln{key}"] = P @ (c * gam) + fgam * (P @ gam)
                res[f"dLambda_dln{key}"] = dlam
            return res
        Zs, sigma = out["Zstar"], out["Zsigma"]

        def dg(z):                     # dg/dln S, dg/dln N at charge z (up at z, down at z + 1)
            i, w = self._u_index(z * E2_EV_CM / a)
            up = max(self._up(z, i, w), 1e-300)
            photo = float(self._photo_arr(self.Kg, mK, self.t["K_bg"], mKb, i, w))
            return photo / up, (up - photo) / up - 1.0

        g0, gm, gp = dg(Zs), dg(Zs - 0.5), dg(Zs + 0.5)
        (gam_lo, gam_hi), (lam_lo, lam_hi) = self._heating_cooling_arr(np.array([Zs, Zs + 1.0]))
        for k, key in enumerate(("S", "N")):
            dZs = sigma * sigma * g0[k]
            res[f"dZmean_dln{key}"] = dZs
            res[f"dZsigma_dln{key}"] = 0.5 * sigma ** 3 * (gp[k] - gm[k])
            res[f"dGamma_dln{key}"] = (gam_hi - gam_lo) * dZs + (out["Gamma"] if key == "S" else 0.0)
            res[f"dLambda_dln{key}"] = (lam_hi - lam_lo) * dZs + (out["Lambda"] if key == "N" else 0.0)
        return res

    def _anchor(self, out):
        """What predict needs from this solve: the inputs, and the per-group photoemission kernel
        at Z* (the change of the photoemission rate there is the S of the linearisation)."""
        N_g, c_red, ne, ions, G0_bg = self._inputs
        i, w = self._u_index(out["Zstar"] * E2_EV_CM / self.a)
        mK, mKb = getattr(self, "_mK", None), getattr(self, "_mKb", None)
        return dict(N_g=np.array(N_g, float), c_red=float(c_red), T=self.cell["T"], ne=float(ne), G0_bg=float(G0_bg),
                    n_ion=np.array([n for n, _, _ in ions], float), k_star=np.array(self._kern(self.Kg, mK, i, w), float),
                    kbg_star=float(self._kern(self.t["K_bg"], mKb, i, w)))


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


PREDICT_TOL = 0.1          # predict: largest |d ln S|, |d ln N|, |d ln T| and residual of the ion densities
PREDICT_TOL_SHAPE = 0.3    # predict: largest change of a group's photon density beyond d ln S
PREDICT_SIGMA_MARGIN = 0.2   # predict: no prediction that would bring sigma this close to SIGMA_DISCRETE
PREDICT_WIDE_ZERO = 1.0    # predict: no prediction of a wide P(Z) with |Z* + 1/2| below this (kink of the cooling)
PREDICT_WIDE_DZ = 0.25     # predict: no prediction moving the Z* of a wide P(Z) by more than max(1, this x sigma)


def predict(r, N_g, c_red, T, ne, ions=(), G0_bg=0.0, tol=PREDICT_TOL, tol_shape=PREDICT_TOL_SHAPE):
    """First-order prediction of a solve from an earlier one of the same cell and bin (r: a
    solve(..., sens=True) result), or None where it must be solved again. The changes since r:
    d ln S of the photoemission rate at r's Z* (all groups and the background, with r's
    kernels), d ln N of n_e, d ln T; refused if one of them, or the change of the ion densities
    beyond d ln N, exceeds tol, if a group's photon density changed by more than tol_shape
    beyond d ln S, if the predicted sigma comes within PREDICT_SIGMA_MARGIN of SIGMA_DISCRETE
    (a change of the closure), or if, for a wide P(Z), Z* + 1/2 of r or of the prediction comes
    within PREDICT_WIDE_ZERO + 1/2 of the kink of the cooling at zero charge (r's slopes span
    [Z*, Z* + 1]) or the two are on either side of it, or the prediction moves Z* by more than
    max(1, PREDICT_WIDE_DZ sigma). Returns a result dict like solve's (Lambda including the
    autoionisation, the d/d ln T and the anchor of r), with "predicted" True."""
    an = r["anchor"]
    N_g = np.asarray(N_g, float)
    P0 = an["c_red"] * float(an["N_g"] @ an["k_star"]) + an["G0_bg"] * an["kbg_star"]
    P1 = c_red * float(N_g @ an["k_star"]) + G0_bg * an["kbg_star"]
    if P0 > 0.0 and P1 > 0.0:
        dS = np.log(P1 / P0)
    elif P0 <= 0.0 and P1 <= 0.0:
        dS = 0.0
    else:
        return None
    dN, dT = np.log(ne / an["ne"]), np.log(T / an["T"])
    if max(abs(dS), abs(dN), abs(dT)) > tol:
        return None
    n_ion = np.array([n for n, _, _ in ions], float)
    big = np.maximum(n_ion, an["n_ion"]) > 1e-6 * ne
    if big.any():
        if np.any((n_ion[big] <= 0.0) | (an["n_ion"][big] <= 0.0)):
            return None
        if np.max(np.abs(np.log(n_ion[big] / an["n_ion"][big]) - dN)) > tol:
            return None
    lit = np.maximum(N_g, an["N_g"]) > 1e-30 * max(np.max(np.maximum(N_g, an["N_g"])), 1e-300)
    if lit.any():
        if np.any((N_g[lit] <= 0.0) | (an["N_g"][lit] <= 0.0)):
            return None
        if np.max(np.abs(np.log(N_g[lit] / an["N_g"][lit]) - dS)) > tol_shape:
            return None
    out = dict(predicted=True, discrete=r["discrete"], anchor=an,
               Lambda_auto=0.0, nodes=r.get("nodes"), weights=r.get("weights"))
    lam0 = r["Lambda"] + r.get("Lambda_auto", 0.0)
    for k, base in (("Zmean", r["Zmean"]), ("Zsigma", r["Zsigma"]), ("Gamma", r["Gamma"]), ("Lambda", lam0)):
        out[k] = base + r[f"d{k}_dlnS"] * dS + r[f"d{k}_dlnN"] * dN + r[f"d{k}_dlnT"] * dT
    out["Zstar"] = r["Zstar"] + (out["Zmean"] - r["Zmean"])
    if abs(out["Zsigma"] - SIGMA_DISCRETE) < PREDICT_SIGMA_MARGIN or (out["Zsigma"] < SIGMA_DISCRETE) != r["discrete"]:
        return None
    if not r["discrete"]:
        # a wide P(Z) is averaged at Z* + 1/2 and its slopes taken over [Z*, Z* + 1]: neither end
        # may come near the kink of the cooling at zero charge, or cross it
        z0, z1 = r["Zstar"] + 0.5, out["Zstar"] + 0.5
        if z0 * z1 <= 0.0 or min(abs(z0), abs(z1)) < PREDICT_WIDE_ZERO + 0.5:
            return None
        # beyond a fraction of the width, the rates' curvature in Z makes the linearisation drift
        if abs(z1 - z0) > max(1.0, PREDICT_WIDE_DZ * r["Zsigma"]):
            return None
    for k in ("dZmean_dlnT", "dZsigma_dlnT", "dGamma_dlnT", "dLambda_dlnT"):
        out[k] = r[k]
    return out


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


def _width(Z, P):
    """Standard deviation of the charge distribution P on the charges Z."""
    mean = P @ Z
    return np.sqrt(max(P @ Z ** 2 - mean ** 2, 0.0))


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
