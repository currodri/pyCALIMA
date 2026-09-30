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
   [Zmin, Zmax] by the Illinois method.
2. sigma^2 = -1 / g'(Z*) (Gaussian limit), g' by central difference.
3. If sigma < ``SIGMA_DISCRETE`` the distribution is built exactly over integer
   charges around Z*; otherwise it is Gaussian with mean Z* + 1/2, and charge
   averages use 3-point Gauss-Hermite nodes.
4. Heating Gamma = sum_g c N_g H_g(U); recombination cooling from DS87
   Lambda-tilde, scaled for electron pass-through, less the energy the
   gas-excited secondaries return; autoionisation cooling at Zmin.

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
    most_negative_allowed_charge_graphite,
    most_negative_allowed_charge_silicate,
    most_positive_allowed_charge,
    photodetachment_cross_section_vec,
)

FORMAT_VERSION = 1
E2_EV_CM = E_STATC ** 2 / EV2ERG      # e^2 in eV cm: U [V] = Z * E2_EV_CM / a
MATERIAL_CODE = {"graphite": 1, "silicate": 2}
SIGMA_DISCRETE = 3.0                  # below this, P(Z) is built exactly
MAX_DISCRETE_STATES = 121

# default grids
N_U, U_MIN, U_MAX, U_SCALE = 401, -1000.0, 1000.0, 0.5   # volts, asinh spacing
TS_GRID = np.geomspace(3e3, 3e6, 31)                     # group-shape temperatures [K]
THOT_GRID = np.geomspace(1e3, 1e9, 61)                   # hot-gas table temperatures [K]
N_E_PER_GROUP = 64

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
                log10_Thot=np.log10(T_hot), delta=delta, P_tr=P_tr, f_cool=f_cool, E_bar=E_bar)


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
    return tab


# ---------------------------------------------------------------------------
# Per-cell solver (mirrors RAMSES)
# ---------------------------------------------------------------------------

class GroupChargingTable:
    """One bin's table with the per-cell charge solve that RAMSES performs."""

    def __init__(self, tab, egy=None):
        self.t = tab
        self.a = tab["a"]
        self.material = tab["material"]
        U = tab["U"]
        self.x0 = np.arcsinh(U[0] / U_SCALE)
        self.dx = (np.arcsinh(U[-1] / U_SCALE) - self.x0) / (U.size - 1)
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
    def _rates(self, Z):
        c = self.cell
        a = self.a
        U = Z * E2_EV_CM / a
        i, w = self._u_index(U)
        photo = c["cN"] @ ((1 - w) * self.Kg[:, i] + w * self.Kg[:, i + 1])
        arrive_e = np.pi * a * a * c["ne"] * c["v_e"] * float(DS87_J_function_vec(Z, -1.0, a, c["T"]))
        ion = 0.0
        for n_i, z_i, v_i in c["ions"]:
            ion += np.pi * a * a * n_i * v_i * float(DS87_J_function_vec(Z, z_i, a, c["T"]))
        sec = arrive_e * self._hot("delta", c["T"], i, w)
        down = arrive_e * self._sticking(Z) * (1.0 - self._hot("P_tr", c["T"], i, w))
        return photo + ion + sec, down, (i, w)

    def _g(self, Z):
        up, _, _ = self._rates(Z)
        _, down, _ = self._rates(Z + 1.0)
        return np.log(max(up, 1e-300)) - np.log(max(down, 1e-300))

    def _heating_cooling(self, Z):
        c = self.cell
        a = self.a
        up, down, (i, w) = self._rates(Z)
        gamma = EV2ERG * float(c["cN"] @ ((1 - w) * self.Hg[:, i] + w * self.Hg[:, i + 1]))
        kT = KB_CGS * c["T"]
        ltil = float(_ds87_lambda(Z, -1.0, a, c["T"]))
        arrive = np.pi * a * a * c["ne"] * c["v_e"]
        lam = arrive * self._sticking(Z) * ltil * kT * self._hot("f_cool", c["T"], i, w)
        jt = float(DS87_J_function_vec(Z, -1.0, a, c["T"]))
        lam -= arrive * jt * self._hot("delta", c["T"], i, w) * self._hot("E_bar", c["T"], i, w) * EV2ERG
        return gamma, lam

    def solve(self, N_g, c_red, T, ne, ions=()):
        """Equilibrium charge and rates for one cell.

        N_g : photon densities per group [cm^-3]; c_red : RT speed of light
        [cm/s]; T [K]; ne [cm^-3]; ions : iterable of (n_i [cm^-3], z_i,
        m_i [g]). Returns a dict with Zmean, Zsigma, Gamma, Lambda and
        Lambda_auto [erg/s per grain], Zstar and whether P(Z) was discrete.
        """
        self.cell = dict(cN=c_red * np.asarray(N_g, dtype=float), T=float(T), ne=float(ne),
                         v_e=np.sqrt(8 * KB_CGS * T / (np.pi * ME_CGS)),
                         ions=[(float(n), float(z), np.sqrt(8 * KB_CGS * T / (np.pi * m))) for n, z, m in ions])
        Zlo, Zhi = self.t["Zmin"], self.t["Zmax"] - 1.0
        glo, ghi = self._g(Zlo), self._g(Zhi)
        if glo <= 0:
            Zs = Zlo
        elif ghi >= 0:
            Zs = Zhi
        else:  # Illinois
            a_, b_, fa, fb, side = Zlo, Zhi, glo, ghi, 0
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
        h = 0.5
        gp = (self._g(min(Zs + h, Zhi)) - self._g(max(Zs - h, Zlo))) / (min(Zs + h, Zhi) - max(Zs - h, Zlo))
        sigma = np.sqrt(-1.0 / gp) if gp < 0 else 0.0

        out = dict(Zstar=Zs)
        if sigma < SIGMA_DISCRETE:
            half = int(np.ceil(6 * max(sigma, 0.5))) + 2
            half = min(half, (MAX_DISCRETE_STATES - 1) // 2)
            Zc = int(np.floor(Zs + 0.5))
            z0 = int(max(self.t["Zmin"], Zc - half))
            z1 = int(min(self.t["Zmax"], Zc + half))
            Zint = np.arange(z0, z1 + 1, dtype=float)
            lnf = np.zeros(Zint.size)
            for k in range(Zint.size - 1):
                lnf[k + 1] = lnf[k] + self._g(Zint[k])
            P = np.exp(lnf - lnf.max())
            P /= P.sum()
            nodes, weights = Zint, P
            out["discrete"] = True
            mean = float(np.sum(P * Zint))
            out["Zmean"], out["Zsigma"] = mean, float(np.sqrt(max(np.sum(P * Zint ** 2) - mean ** 2, 0.0)))
        else:
            mu = Zs + 0.5
            nodes = np.array([mu - np.sqrt(3) * sigma, mu, mu + np.sqrt(3) * sigma])
            weights = np.array([1 / 6, 2 / 3, 1 / 6])
            out["discrete"] = False
            out["Zmean"], out["Zsigma"] = float(mu), float(sigma)
        gam = lam = 0.0
        for Z, wgt in zip(nodes, weights):
            g_, l_ = self._heating_cooling(Z)
            gam += wgt * g_
            lam += wgt * l_
        auto = 0.0
        if out["discrete"] and nodes[0] == self.t["Zmin"]:
            EA = float((electron_affinity_graphite_vec if self.material == "graphite"
                        else electron_affinity_silicate_vec)(self.t["Zmin"], self.a))
            arrive = np.pi * self.a ** 2 * self.cell["ne"] * self.cell["v_e"]
            auto = weights[0] * arrive * float(DS87_J_function_vec(self.t["Zmin"], -1.0, self.a, T)) * EA * EV2ERG
        out.update(Gamma=float(gam), Lambda=float(lam), Lambda_auto=float(auto))
        return out


def _ds87_lambda(Z, q, a, T):
    from pycalima.models.dust_charge.shared_physics import DS87_lambda_scalar
    return DS87_lambda_scalar(Z, q, a, T)
