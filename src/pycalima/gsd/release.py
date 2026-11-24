"""Photo-release of molecules stuck to nano-grains: the sticking efficiency S_eff of the carbon pair rule.

Decided 2026-11-23 (status report, "How carbon moves"). A molecule that sticks to a nano-grain (1-20 nm) stays only
if it is buried before a photon releases it:

    S_eff = 1 / (1 + R_abs (P_th + Y_pd) t_bury),

with R_abs the adsorbed molecule's photon absorption rate, P_th the single-photon thermal release probability,
Y_pd the non-thermal photodesorption yield (unmeasured: an input labelled 'calibrated', default 0, calibrated in A5
on molecular-cloud parity and dense-core depletion; expected 1e-7 to 1e-6) and t_bury the host's burial time.
Molecules on grains >= 20 nm stick (S = 1).

P_th (``p_thermal``) follows the literature review of 2026-11-23 (overview/lit/carbon_transfer_release.md), each
step labelled:
- binding energy E_b = 0.052 N_C eV (PAHs on graphite, Zacharia et al. 2004; the host's curvature is ignored);
- the photon's energy E is shared over the molecule and its host, which reach the peak temperature T_m with
  int_0^T_m C dT = E, the harmonic sp2 heat capacity of every carbon atom (``carbon_heat_capacity``; H ignored);
- desorption at the Arrhenius rate k = nu exp(-E_b / k T_e) with nu = 2.5e17 s^-1 and the microcanonical correction
  T_e = T_m (1 - 0.2 E_b / E) (Lange et al. 2023);
- the aggregate cools by its IR emission (THEMIS-like a-C optics; hosts up to 20 nm are far below the IR wavelengths,
  so the power per atom does not depend on size, the Rayleigh limit, and is tabulated once); since k is steep in T,
  P_th = k(T_e) tau_e with tau_e the time over which k falls by e, (k T_e^2 / E_b) / |dT/dt|, capped at 1.
Release is thus a single-photon effect of small molecules on ~1 nm hosts: a 3 nm host's peak temperature (~100-140 K
for a UV photon) gives P_th ~ 0.

t_bury (``burial_time``) is the time for the host to grow by one graphitic layer (0.34 nm, labelled) at its rate
of growth da/dt; without growth it is infinite and any release keeps the molecule off the host.
"""
from __future__ import annotations

import numpy as np

from .carbon import CARRIERS, M_C

K_B = 1.380649e-16
EV = 1.602176634e-12
H_P = 6.62607015e-27
C_L = 2.99792458e10
E_BIND_PER_C = 0.052        # eV per C atom (Zacharia et al. 2004)
NU_DES = 2.5e17             # s^-1 (Lange et al. 2023)
T_E_FACTOR = 0.2            # T_e = T_m (1 - 0.2 E_b / E) (Lange et al. 2023)
BURIAL_DEPTH = 0.34e-7      # cm: one graphitic layer (labelled)
Y_PD_DEFAULT = 0.0          # 'calibrated' (A5)
_T = np.geomspace(3.0, 1e4, 400)
_LAM_IR = np.geomspace(1.0, 3000.0, 200)        # um


def binding_energy(nc):
    """E_b [eV] of a PAH of nc carbon atoms on a carbon surface."""
    return E_BIND_PER_C * np.asarray(nc, float)


def _enthalpy_per_c():
    """int_0^T C dT per carbon atom [erg] on the temperature grid ``_T`` (harmonic sp2 heat capacity)."""
    from pycalima.models.dust_radiation.carbon_heat_capacity import per_sp2_carbon

    c = np.array([float(np.squeeze(per_sp2_carbon(t))) for t in _T]) * K_B
    h = np.concatenate([[0.5 * c[0] * _T[0]], 0.5 * (c[1:] + c[:-1]) * np.diff(_T)])
    return np.cumsum(h), c


_H, _C, _P1 = None, None, None


def peak_temperature(energy_ev, n_atoms):
    """T_m [K] of n_atoms carbon atoms after absorbing energy_ev [eV] from 0 K (the thermal approximation)."""
    global _H, _C
    if _H is None:
        _H, _C = _enthalpy_per_c()
    e = np.asarray(energy_ev, float) * EV / np.asarray(n_atoms, float)
    return np.interp(e, _H, _T)


def _ir_per_atom():
    """4 pi int C_abs B_lambda dlambda per carbon atom [erg/s] on ``_T``, from a 1 nm a-C grain (Rayleigh limit)."""
    n_ref = 4.0 / 3.0 * np.pi * 1.6 * (1e-7) ** 3 / M_C
    c_abs = CARRIERS["aromatic"].cross_sections(np.array([n_ref * M_C]), _LAM_IR)["C_abs"][:, 0] / n_ref
    lam = _LAM_IR * 1e-4
    out = np.empty(_T.size)
    for k, t in enumerate(_T):
        x = np.minimum(H_P * C_L / (lam * K_B * t), 700.0)
        out[k] = 4.0 * np.pi * np.trapezoid(c_abs * 2.0 * H_P * C_L ** 2 / lam ** 5 / np.expm1(x), lam)
    return out


def ir_power(n_atoms, T):
    """IR power [erg/s] of an a-C grain of n_atoms carbon atoms at temperature T (Rayleigh limit)."""
    global _P1
    if _P1 is None:
        _P1 = _ir_per_atom()
    return np.asarray(n_atoms, float) * np.exp(np.interp(np.log(T), np.log(_T), np.log(np.maximum(_P1, 1e-300))))


def p_thermal(nc_mol, a_host, energy_ev, rho=1.6):
    """Single-photon thermal release probability of a molecule of nc_mol C atoms on a host of radius a_host [cm]
    after one photon of energy_ev [eV] (module doc)."""
    n_host = 4.0 / 3.0 * np.pi * rho * a_host ** 3 / M_C
    n_tot = nc_mol + n_host
    e_b = float(binding_energy(nc_mol))
    if e_b >= energy_ev / T_E_FACTOR:
        return 0.0
    t_m = float(peak_temperature(energy_ev, n_tot))
    t_e = t_m * (1.0 - T_E_FACTOR * e_b / energy_ev)
    if t_e <= 0.0:
        return 0.0
    k = NU_DES * np.exp(-e_b * EV / (K_B * t_e))
    if k <= 0.0:
        return 0.0
    c_tot = n_tot * float(np.interp(t_m, _T, _C))
    dtdt = float(ir_power(n_tot, t_m)) / c_tot
    tau = (K_B * t_e ** 2 / (e_b * EV)) / max(dtdt, 1e-300)
    return float(min(1.0, k * tau))


def burial_time(a_dot):
    """t_bury [s]: one graphitic layer of growth at the host's da/dt [cm/s] (infinite without growth)."""
    a_dot = np.asarray(a_dot, float)
    return np.where(a_dot > 0.0, BURIAL_DEPTH / np.maximum(a_dot, 1e-300), np.inf)


def s_eff(r_abs, p_th, t_bury, y_pd=Y_PD_DEFAULT):
    """S_eff = 1 / (1 + R_abs (P_th + Y_pd) t_bury): 1 with no release, 0 with release and no burial."""
    rate = np.asarray(r_abs, float) * (np.asarray(p_th, float) + y_pd)          # releases per second
    t = np.asarray(t_bury, float)
    with np.errstate(invalid="ignore", over="ignore"):
        x = np.where(rate > 0.0, rate * t, 0.0)
    return 1.0 / (1.0 + x)
