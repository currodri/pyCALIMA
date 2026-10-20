"""Statistical electron sticking on a neutral PAH: e- + PAH -> PAH-.

A captured electron leaves the anion vibrationally hot, at internal energy
E = EA + E_n + eps above the anion ground state (E_n the neutral's vibrational
energy, eps the electron's kinetic energy). The anion then either re-emits
the electron (autodetachment) or radiates IR photons until E < EA, where it is
bound. The sticking coefficient is the probability of the second outcome,
averaged over the capture flux.

- **Autodetachment** by detailed balance with the capture cross section
  sigma_c (Weisskopf/Klots; the thermionic-emission form):

      k_d(E) = (2 m_e g_n / (g_a pi^2 hbar^3)) int sigma_c(eps) eps rho_n(E - EA - eps) deps / rho_a(E)

  g_n and g_a are the electronic degeneracies of the neutral and the anion,
  and the 2 is the electron spin.
- **Capture.** The default sigma_c is the one behind
  ``chemistry.electron_capture`` for Z = 0: 0.82 pi a^2 [1 + sqrt(2 e^2 / (a eps))],
  whose flux average is the DS87 J~ = 1 + sqrt(pi / 2 tau). The sticking
  coefficient therefore multiplies that capture rate consistently.
- **Stabilisation.** Harmonic IR down-jumps (``cooling.mode_emission_rates``)
  of the anion modes. P_stab(E) = 1 below EA, and above it::

      P_stab(E) = sum_i k_i(E) P_stab(E - h nu_i) / (sum_i k_i(E) + k_d(E)),

  exact for a cascade of down-jumps alone (no photon absorbed in between).

Energies are in cm^-1 on a Beyer-Swinehart grid, as in ``dos``.
"""

from __future__ import annotations

import numpy as np

from pycalima.pahchem import cooling, dos

try:  # optional accelerator (the ``accel`` extra); the NumPy path gives the same numbers
    from numba import njit
    HAVE_NUMBA = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

ME_G = 9.1093837e-28
HBAR_CGS = 1.054571817e-27
E2_CGS = 4.80320471e-10 ** 2          # e^2 [erg cm]
ERG_PER_CM = 1.98644586e-16           # h c [erg cm]


def capture_cross_section(eps_cm, radius_cm, disc_factor=0.82):
    """sigma_c(eps) [cm^2] of a neutral conducting disc, eps in cm^-1."""
    eps = np.asarray(eps_cm, dtype=float) * ERG_PER_CM
    return disc_factor * np.pi * radius_cm**2 * (1.0 + np.sqrt(2.0 * E2_CGS / (radius_cm * eps)))


def _stabilisation_numpy(k_modes, k_tot, steps, i_ea):
    n = k_modes.shape[1]
    P = np.ones(n)
    for i in range(i_ea + 1, n):
        # a window with no anion states carries its neighbour's value (sparse mode sets only)
        P[i] = (k_modes[:, i] @ P[np.maximum(i - steps, 0)]) / k_tot[i] if k_tot[i] > 0 else P[i - 1]
    return P


if HAVE_NUMBA:
    @njit(cache=True)
    def _stabilisation(k_modes, k_tot, steps, i_ea):
        n = k_modes.shape[1]
        P = np.ones(n)
        for i in range(i_ea + 1, n):
            if k_tot[i] > 0:
                s = 0.0
                for m in range(steps.size):
                    s += k_modes[m, i] * P[max(i - steps[m], 0)]
                P[i] = s / k_tot[i]
            else:
                P[i] = P[i - 1]
        return P
else:  # pragma: no cover
    _stabilisation = _stabilisation_numpy


class Attachment:
    """Electron sticking on one neutral PAH, from its and its anion's modes.

    Parameters
    ----------
    neutral, anion : PAHSpecies
        The anion must carry Einstein A coefficients.
    ea_ev : float
        Adiabatic electron affinity [eV].
    radius_cm : float
        Radius in ``capture_cross_section`` (``chemistry.disc_radius_cm``).
    e_excess_cm : float
        How far above EA the grid extends.
    g_neutral, g_anion : int
        Electronic degeneracies (closed-shell neutral 1, doublet anion 2).
    """

    def __init__(self, neutral, anion, ea_ev, radius_cm, e_excess_cm=15000.0, grain_cm=2.0,
                 g_neutral=1, g_anion=2, window_cm=20.0):
        self.ea_cm = ea_ev * dos.CM_PER_EV
        self.grain = float(grain_cm)
        self.radius_cm = radius_cm
        e_top = self.ea_cm + e_excess_cm
        self.energy_cm, counts_a = dos.beyer_swinehart(anion.freqs_cm, e_top, grain_cm)
        _, counts_n = dos.beyer_swinehart(neutral.freqs_cm, e_excess_cm, grain_cm)
        self._counts_n = counts_n
        self.i_ea = int(round(self.ea_cm / grain_cm))
        n = self.energy_cm.size
        # int sigma_c eps rho_n(x - eps) deps on the excess grid x = j grain (rho deps = counts)
        eps = np.arange(1, counts_n.size) * grain_cm
        g = capture_cross_section(eps, radius_cm) * eps * ERG_PER_CM            # [erg cm^2]
        conv = np.zeros(counts_n.size)
        conv[1:] = np.convolve(g, counts_n)[:counts_n.size - 1]                  # x = eps + E_n
        pre = 2.0 * ME_G * g_neutral / (g_anion * np.pi**2 * HBAR_CGS**3)
        # numerator and rho_a averaged over the same window, as the IR rates are (cooling)
        half = max(int(round(0.5 * window_cm / grain_cm)), 1)
        m = n - self.i_ea - 1
        num = np.zeros(n)
        num[self.i_ea + 1:] = conv[1:m + 1]
        num_w = cooling._window_sum(num, half)
        rho_a_erg = cooling._window_sum(counts_a, half) / (grain_cm * ERG_PER_CM)   # [erg^-1] x window bins
        self.k_detach = np.zeros(n)
        hi = slice(self.i_ea + 1, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.k_detach[hi] = np.where(rho_a_erg[hi] > 0, pre * num_w[hi] / rho_a_erg[hi], 0.0)
        k_modes = cooling.mode_emission_rates(self.energy_cm, counts_a, anion.freqs_cm, anion.einstein_a,
                                              window_cm=window_cm)
        self.k_ir = k_modes.sum(axis=0)
        steps = np.maximum(np.rint(np.asarray(anion.freqs_cm) / grain_cm).astype(np.int64), 1)
        self.p_stab = _stabilisation(np.ascontiguousarray(k_modes), self.k_ir + self.k_detach, steps, self.i_ea)

    def sticking(self, temperature_k, neutral_energy_cm=None, neutral_weights=None, t_vib_k=None):
        """s_e = <P_stab(EA + E_n + eps)> over the capture flux and the neutral's E_n.

        The electrons are Maxwellian at ``temperature_k``, weighted by
        sigma_c(eps) eps exp(-eps/kT). The neutral's vibrational energy is
        canonical at ``t_vib_k`` (default ``temperature_k``, as in a thermal
        laboratory cell) or the distribution ``neutral_weights`` on
        ``neutral_energy_cm`` (e.g. an ``energy_me.SteadyState`` P(E)).

        Weight that lands above the grid (the hot tail of an irradiated P(E))
        is given the P_stab of the top bin; more than 1e-3 of it raises.
        """
        kt = dos.KB_CM * temperature_k
        n_excess = self.energy_cm.size - self.i_ea
        x = np.arange(n_excess) * self.grain
        eps = x[1:]
        w_e = np.zeros(n_excess)
        w_e[1:] = capture_cross_section(eps, self.radius_cm) * eps * np.exp(-eps / kt)
        w_n = np.zeros(n_excess)
        if neutral_weights is None:
            c = self._counts_n[:n_excess]
            w_n[:c.size] = c * np.exp(-x[:c.size] / (dos.KB_CM * (t_vib_k or temperature_k)))
            if w_n[c.size - 1] > 1e-8 * w_n.max():
                raise ValueError("canonical neutral distribution is cut by the grid; raise e_excess_cm")
            total = w_n.sum()
        else:
            p = np.asarray(neutral_weights, dtype=float)
            idx = np.rint(np.asarray(neutral_energy_cm) / self.grain).astype(int)
            ok = idx < n_excess
            np.add.at(w_n, idx[ok], p[ok])
            total = p.sum()
        w = np.convolve(w_n, w_e)[:n_excess]
        overflow = total * w_e.sum() - w.sum()
        if overflow > 1e-3 * total * w_e.sum():
            raise ValueError("thermal weights extend above the grid; raise e_excess_cm")
        return float((w @ self.p_stab[self.i_ea:] + overflow * self.p_stab[-1]) / (w.sum() + overflow))
