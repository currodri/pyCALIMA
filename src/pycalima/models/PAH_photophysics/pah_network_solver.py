"""
pah_network_solver.py — Steady-state and time-evolution solver for the PAH
hydrogen + charge network.

Species grid
------------
  Z  ∈ {-1, 0, +1, +2}  (charge states)
  Nh ∈ [0, Nh0 + n_superh_max]  with n_superh_max = 4 by default

Total species per PAH: nZ × nNh = 4 × (Nh0 + 5).
  C54H18  →  4 × 23 = 92 species
  C96H24  →  4 × 29 = 116 species

Processes
---------
  H-addition         (Z, Nh) + H → (Z, Nh+1)               all Nh < Nh_max
  H-loss (UV)        (Z, Nh)     → (Z, Nh-1)               photodissociation
  H2-loss (UV)       (Z, Nh)     → (Z, Nh-2)               when k_H2loss > 0
  H2-abstraction     (Z, Nh) + H → (Z, Nh-1) + H2          Eley-Rideal; Nh > Nh0
  Photoionisation    (Z, Nh)     → (Z+1, Nh)               Z < +2
  Recombination      (Z, Nh) + e⁻→ (Z-1, Nh)              Z > 0
  Attachment         (Z=0, Nh) + e⁻ → (Z=-1, Nh)          Z = 0

Two solver modes
----------------
  solve_equilibrium(...)
      Steady-state: A @ n = 0 with ∑n = n_PAH_total.
      method='direct'  — replace one equation with the conservation constraint,
                         call scipy.sparse.linalg.spsolve.  < 1 ms for N ~ 100.
      method='newton'  — scipy.optimize.root (hybr / MINPACK) with the analytical
                         sparse Jacobian.  Equivalent for the linear problem;
                         preferred when extending to non-linear coupling.

  solve_evolution(n0, dt, ...)
      Out-of-equilibrium integration from initial condition n0 over a time dt.
      method='expm'    — scipy.sparse.linalg.expm_multiply: exact solution of
                         dn/dt = A n for the linear network in one call.
                         No time stepping, no accumulated truncation error.
      method='ode'     — scipy.integrate.solve_ivp (Radau, stiff-safe).
                         Handles non-linear residuals when subclassing.

Quick-start
-----------
    from models.PAH_photophysics.pah_network_solver import make_c54_solver
    import numpy as np

    solver = make_c54_solver()

    # Zero photodissociation for demo; replace with tables from model_data/
    k_Hloss  = np.zeros((solver.nZ, solver.nNh))
    k_H2loss = np.zeros((solver.nZ, solver.nNh))
    k_ion    = np.array([1e-7, 5e-7, 2e-7, 0.0])   # [s⁻¹] per charge state
    k_rec    = np.array([0.0, 0.0, 1e-6, 1e-6])     # [cm³/s]
    k_att    = 1e-9                                  # [cm³/s]

    env = dict(n_H=1e3, n_e=1e-2, T=100.0, a_pah_cm=3.7e-8)

    # ── Steady-state ────────────────────────────────────────────────────
    n_eq = solver.solve_equilibrium(k_Hloss, k_H2loss, k_ion, k_rec, k_att,
                                    **env, n_PAH_total=1.0)

    # ── Time evolution from a flat initial condition ────────────────────
    n0   = np.zeros((solver.nZ, solver.nNh))
    n0[solver._iz[0], solver.Nh_vals.tolist().index(solver.Nh0)] = 1.0  # all neutral Nh0
    n_t  = solver.solve_evolution(n0, dt=1e6, k_Hloss=k_Hloss,
                                  k_H2loss=k_H2loss, k_ion=k_ion, k_rec=k_rec,
                                  k_att=k_att, **env)
    solver.print_distribution(n_t)
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, expm_multiply
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .pah_hydrogen_chemistry import (
    collisional_rate,
    reaction_efficiency_neutral,
    h2_abstraction_rate_coefficient,
    K_CATION_CM3S,
    K_ANION_CM3S,
)

N_SUPERH_MAX: int = 4
_Z_VALUES = np.array([-1, 0, 1, 2], dtype=int)


class PAHNetworkSolver:
    """
    Two-mode solver for the PAH hydrogen + charge network.

    Parameters
    ----------
    Nc           : Carbon count (e.g. 54 for circumcoronene C54H18).
    Nh0          : Normal fully-hydrogenated H count (e.g. 18).
    parent_solo  : Solo H atoms in the fully-hydrogenated parent (e.g. 6).
    parent_duo   : Duo  H atoms in the fully-hydrogenated parent (e.g. 12).
    n_superh_max : Extra H atoms above Nh0 included in the network (default 4).
    """

    Z_VALUES = _Z_VALUES

    def __init__(
        self,
        Nc: int,
        Nh0: int,
        parent_solo: int,
        parent_duo: int,
        n_superh_max: int = N_SUPERH_MAX,
    ) -> None:
        self.Nc           = Nc
        self.Nh0          = Nh0
        self.parent_solo  = parent_solo
        self.parent_duo   = parent_duo
        self.n_superh_max = n_superh_max

        self.Z_vals  = _Z_VALUES.copy()
        self.nZ      = len(self.Z_vals)
        self.Nh_vals = np.arange(0, Nh0 + n_superh_max + 1, dtype=int)
        self.nNh     = len(self.Nh_vals)
        self.N       = self.nZ * self.nNh

        # Z-value → array-index mapping  (e.g. -1 → 0, 0 → 1, +1 → 2, +2 → 3)
        self._iz: dict[int, int] = {int(z): i for i, z in enumerate(self.Z_vals)}

    # ------------------------------------------------------------------ #
    # Index helpers                                                        #
    # ------------------------------------------------------------------ #

    def idx(self, iz: int, inh: int) -> int:
        """Flat species index from charge-index iz and H-count index inh."""
        return iz * self.nNh + inh

    def species_at(self, k: int) -> tuple[int, int]:
        """Return (Z, Nh) for flat species index k."""
        return int(self.Z_vals[k // self.nNh]), int(self.Nh_vals[k % self.nNh])

    # ------------------------------------------------------------------ #
    # Internal rate pre-computation                                        #
    # ------------------------------------------------------------------ #

    def _hadd_rates(self, a_pah_cm: float, T: float, n_H: float) -> np.ndarray:
        """
        H-addition rate [s⁻¹] for every (iz, inh).  Shape (nZ, nNh).
        Already multiplied by n_H.
        """
        out = np.empty((self.nZ, self.nNh))
        k_coll_T = float(collisional_rate(a_pah_cm, T))

        for iz, Z in enumerate(self.Z_vals):
            if Z > 0:
                out[iz, :] = K_CATION_CM3S * n_H
            elif Z < 0:
                out[iz, :] = K_ANION_CM3S * n_H
            else:
                for inh, Nh in enumerate(self.Nh_vals):
                    n_extra = int(Nh) - self.Nh0
                    P = float(reaction_efficiency_neutral(n_extra, T))
                    out[iz, inh] = k_coll_T * P * n_H

        return out

    def _er_rates(self, T: float, n_H: float) -> np.ndarray:
        """
        Eley-Rideal H2-abstraction rate [s⁻¹] for super-H states (Nh > Nh0).
        Shape (nZ, nNh); zero for Nh ≤ Nh0.
        """
        out = np.zeros((self.nZ, self.nNh))
        k_er = float(h2_abstraction_rate_coefficient(T)) * n_H
        superH = self.Nh_vals > self.Nh0
        out[:, superH] = k_er
        return out

    # ------------------------------------------------------------------ #
    # Rate matrix assembly                                                 #
    # ------------------------------------------------------------------ #

    def build_rate_matrix(
        self,
        k_Hloss:  np.ndarray,    # (nZ, nNh) [s⁻¹]
        k_H2loss: np.ndarray,    # (nZ, nNh) [s⁻¹]
        k_hadd_s: np.ndarray,    # (nZ, nNh) [s⁻¹]  — already × n_H
        k_er_s:   np.ndarray,    # (nZ, nNh) [s⁻¹]  — already × n_H
        k_ion:    np.ndarray,    # (nZ,)     [s⁻¹]
        k_rec_s:  np.ndarray,    # (nZ,)     [s⁻¹]  — already × n_e
        k_att_s:  float,         #           [s⁻¹]  — already × n_e
    ) -> sparse.csr_matrix:
        """
        Assemble the N×N sparse transition rate matrix A where dN/dt = A @ n.

          A[i, j] > 0  (i ≠ j): rate flowing from species j into species i.
          A[j, j] < 0          : total outflow rate from species j.

        The conservation constraint is NOT encoded here; it is imposed by the
        solver backends.
        """
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def _add(from_j: int, to_i: int, rate: float) -> None:
            if rate <= 0.0:
                return
            rows.append(to_i);   cols.append(from_j); data.append(rate)    # gain
            rows.append(from_j); cols.append(from_j); data.append(-rate)   # loss

        for iz in range(self.nZ):
            Z = int(self.Z_vals[iz])
            for inh in range(self.nNh):
                Nh = int(self.Nh_vals[inh])
                j  = self.idx(iz, inh)

                # H-addition: (iz, inh) → (iz, inh+1)
                if inh + 1 < self.nNh:
                    _add(j, self.idx(iz, inh + 1), k_hadd_s[iz, inh])

                # H-loss UV: (iz, inh) → (iz, inh-1)
                if inh > 0:
                    _add(j, self.idx(iz, inh - 1), k_Hloss[iz, inh])

                # H2-loss UV: (iz, inh) → (iz, inh-2)
                if inh >= 2:
                    _add(j, self.idx(iz, inh - 2), k_H2loss[iz, inh])

                # H2 abstraction (ER): (iz, inh) → (iz, inh-1), only Nh > Nh0
                if Nh > self.Nh0 and inh > 0:
                    _add(j, self.idx(iz, inh - 1), k_er_s[iz, inh])

                # Photoionisation: (iz, inh) → (iz+1, inh)
                if iz + 1 < self.nZ:
                    _add(j, self.idx(iz + 1, inh), k_ion[iz])

                # Recombination: (iz, inh) + e⁻ → (iz-1, inh), Z > 0
                if Z > 0:
                    _add(j, self.idx(self._iz[Z - 1], inh), k_rec_s[iz])

                # Electron attachment: Z=0 → Z=-1
                if Z == 0:
                    _add(j, self.idx(self._iz[-1], inh), k_att_s)

        return sparse.csr_matrix((data, (rows, cols)), shape=(self.N, self.N),
                                 dtype=float)

    # ------------------------------------------------------------------ #
    # Shared rate preparation                                              #
    # ------------------------------------------------------------------ #

    def _prepare_matrix(
        self,
        k_Hloss:  np.ndarray,
        k_H2loss: np.ndarray,
        k_ion:    np.ndarray,
        k_rec:    np.ndarray,
        k_att:    float,
        n_H:      float,
        n_e:      float,
        T:        float,
        a_pah_cm: float,
    ) -> sparse.csr_matrix:
        k_hadd_s = self._hadd_rates(a_pah_cm, T, n_H)
        k_er_s   = self._er_rates(T, n_H)
        k_rec_s  = np.asarray(k_rec, dtype=float) * n_e
        k_att_s  = float(k_att) * n_e
        return self.build_rate_matrix(
            np.asarray(k_Hloss,  dtype=float),
            np.asarray(k_H2loss, dtype=float),
            k_hadd_s, k_er_s,
            np.asarray(k_ion, dtype=float),
            k_rec_s, k_att_s,
        )

    # ------------------------------------------------------------------ #
    # Mode 1: Steady-state equilibrium                                     #
    # ------------------------------------------------------------------ #

    def solve_equilibrium(
        self,
        k_Hloss:     np.ndarray,
        k_H2loss:    np.ndarray,
        k_ion:       np.ndarray,
        k_rec:       np.ndarray,
        k_att:       float,
        n_H:         float,
        n_e:         float,
        T:           float,
        a_pah_cm:    float,
        n_PAH_total: float = 1.0,
        method:      str   = 'direct',
    ) -> np.ndarray:
        """
        Solve for the steady-state PAH abundance distribution n(Z, Nh).

        Parameters
        ----------
        k_Hloss  : (nZ, nNh) [s⁻¹]   UV H-loss photodissociation rates.
        k_H2loss : (nZ, nNh) [s⁻¹]   UV H2-loss photodissociation rates.
        k_ion    : (nZ,)     [s⁻¹]   Photoionisation rate per charge state.
                                      k_ion[iz] drives Z_vals[iz] → Z_vals[iz]+1.
                                      k_ion[-1] (Z=+2→+3) is unused.
        k_rec    : (nZ,)     [cm³/s] Recombination rate coefficients.
                                      Only entries with Z_vals[iz] > 0 are used.
        k_att    : float     [cm³/s] Electron attachment coefficient (Z=0 → -1).
        n_H      : float     [cm⁻³]  Atomic hydrogen density.
        n_e      : float     [cm⁻³]  Electron density.
        T        : float     [K]     Gas temperature.
        a_pah_cm : float     [cm]    PAH effective radius (use afromNc).
        n_PAH_total : float  [cm⁻³] Conservation normalisation.
        method   : 'direct' | 'newton'  Solver backend.

        Returns
        -------
        n2d : ndarray (nZ, nNh)
            Equilibrium abundances.  n2d[iz, inh] ↔ (Z_vals[iz], Nh_vals[inh]).
        """
        A = self._prepare_matrix(k_Hloss, k_H2loss, k_ion, k_rec, k_att,
                                  n_H, n_e, T, a_pah_cm)

        if method == 'direct':
            n_flat = self._equilibrium_direct(A, n_PAH_total)
        elif method == 'newton':
            n_flat = self._equilibrium_newton(A, n_PAH_total)
        else:
            raise ValueError(f"method must be 'direct' or 'newton', got {method!r}")

        return np.maximum(n_flat.reshape(self.nZ, self.nNh), 0.0)

    def _equilibrium_direct(
        self, A: sparse.csr_matrix, n_PAH_total: float
    ) -> np.ndarray:
        """
        Direct sparse solve: replace last equation with ∑n = n_PAH_total.
        Uses SuiteSparse UMFPACK via scipy.sparse.linalg.spsolve.
        """
        A_lil = A.tolil()
        A_lil[-1, :] = 1.0
        b = np.zeros(self.N)
        b[-1] = n_PAH_total
        return spsolve(A_lil.tocsr(), b)

    def _equilibrium_newton(
        self,
        A:           sparse.csr_matrix,
        n_PAH_total: float,
        tol:         float = 1e-12,
        max_iter:    int   = 50,
    ) -> np.ndarray:
        """
        Newton-Raphson for the modified system F(x) = 0:

            F_i(x) = (A @ x)_i          for i < N-1
            F_{N-1}(x) = sum(x) - n_PAH_total

        The Jacobian J = dF/dx equals A with its last row replaced by [1,…,1].
        Because the network is linear in x, J is constant and Newton-Raphson
        converges in exactly one step: x* = J⁻¹ b.

        The solve uses scipy.sparse.linalg.spsolve (sparse LU / UMFPACK) so it
        is robust to the extreme rate-magnitude spread (~12 orders of magnitude)
        that makes iterative Newton implementations (MINPACK hybr) fail.

        For genuinely non-linear extensions (self-consistent n_e, etc.) override
        this method and add a residual-norm loop around the spsolve call.
        """
        J_lil = A.tolil()
        J_lil[-1, :] = 1.0          # conservation row
        J_csr = J_lil.tocsr()

        b = np.zeros(self.N)
        b[-1] = n_PAH_total

        # Newton step: Δx = J⁻¹ (b − F(x₀)); for linear F this is the exact solution.
        x = spsolve(J_csr, b)

        # Verify residual (should be machine precision for linear system)
        res = np.abs(A @ x)
        res[-1] = abs(x.sum() - n_PAH_total)
        if res.max() > tol * n_PAH_total * self.N:
            import warnings
            warnings.warn(
                f"Newton residual {res.max():.2e} exceeds tolerance {tol:.2e}",
                RuntimeWarning, stacklevel=4,
            )

        return x

    # ------------------------------------------------------------------ #
    # Mode 2: Time evolution                                               #
    # ------------------------------------------------------------------ #

    def solve_evolution(
        self,
        n0:       np.ndarray,
        dt:       float,
        k_Hloss:  np.ndarray,
        k_H2loss: np.ndarray,
        k_ion:    np.ndarray,
        k_rec:    np.ndarray,
        k_att:    float,
        n_H:      float,
        n_e:      float,
        T:        float,
        a_pah_cm: float,
        method:   str   = 'expm',
        rtol:     float = 1e-8,
        atol:     float = 1e-14,
    ) -> np.ndarray:
        """
        Integrate the network from initial condition n0 over time dt.

        Parameters
        ----------
        n0       : ndarray (nZ, nNh)  Initial abundance distribution.
        dt       : float  [s]         Integration time span.
        k_Hloss, k_H2loss, k_ion, k_rec, k_att, n_H, n_e, T, a_pah_cm :
                   Same as solve_equilibrium.
        method   : 'expm' | 'ode'     Integrator backend.

            'expm'  — scipy.sparse.linalg.expm_multiply.
                      Computes exp(A·dt)·n0 exactly for the linear network
                      via a Krylov-based matrix-free algorithm.  No step-size
                      control needed; handles arbitrarily stiff A in one call.

            'ode'   — scipy.integrate.solve_ivp (Radau, implicit, stiff-safe).
                      Adaptive time stepping with the sparse Jacobian.
                      Use when overriding _rhs() to add non-linear terms.

        rtol, atol : ODE tolerances (ignored for method='expm').

        Returns
        -------
        n_out : ndarray (nZ, nNh)
            Abundance distribution at time t0 + dt.
        """
        A = self._prepare_matrix(k_Hloss, k_H2loss, k_ion, k_rec, k_att,
                                  n_H, n_e, T, a_pah_cm)
        n0_flat = np.asarray(n0, dtype=float).ravel()

        if method == 'expm':
            n_flat = self._evolve_expm(A, n0_flat, dt)
        elif method == 'ode':
            n_flat = self._evolve_ode(A, n0_flat, dt, rtol, atol)
        else:
            raise ValueError(f"method must be 'expm' or 'ode', got {method!r}")

        return np.maximum(n_flat.reshape(self.nZ, self.nNh), 0.0)

    def _evolve_expm(
        self, A: sparse.csr_matrix, n0: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Exact solution n(dt) = exp(A·dt) · n0.

        expm_multiply uses a Krylov subspace method with no explicit time
        stepping; it is exact up to floating-point and is the fastest option
        for the linear network (stiffness has no cost).
        """
        return expm_multiply(A * dt, n0)

    def _evolve_ode(
        self,
        A:    sparse.csr_matrix,
        n0:   np.ndarray,
        dt:   float,
        rtol: float,
        atol: float,
    ) -> np.ndarray:
        """
        Integrate dN/dt = A @ n over [0, dt] with Radau (implicit, stiff-safe).

        The sparse Jacobian structure is passed so the solver avoids dense
        finite-differences and uses sparse LU factorisation internally.
        """
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return A @ y

        sol = solve_ivp(
            rhs,
            t_span=[0.0, dt],
            y0=n0,
            method='Radau',
            jac=A,              # constant sparse Jacobian
            jac_sparsity=A,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol.y[:, -1]

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def summary(self, n2d: np.ndarray) -> dict:
        """
        Abundance-weighted summary statistics.

        Returns dict: mean_Z, mean_Nh, f_Z (nZ,), f_Nh (nNh,), peak_Z, peak_Nh.
        """
        n_tot = n2d.sum()
        w = n2d / n_tot
        mean_Z  = float(np.einsum('ij,i->', w, self.Z_vals.astype(float)))
        mean_Nh = float(np.einsum('ij,j->', w, self.Nh_vals.astype(float)))
        peak_iz, peak_inh = np.unravel_index(n2d.argmax(), n2d.shape)
        return {
            "mean_Z":  mean_Z,
            "mean_Nh": mean_Nh,
            "f_Z":     w.sum(axis=1),
            "f_Nh":    w.sum(axis=0),
            "peak_Z":  int(self.Z_vals[peak_iz]),
            "peak_Nh": int(self.Nh_vals[peak_inh]),
        }

    def print_distribution(self, n2d: np.ndarray, n_PAH_total: float = 1.0) -> None:
        """Print a compact (Nh × Z) fraction table to stdout."""
        fracs = n2d / n_PAH_total
        z_header = "  ".join(f"   Z={z:+d}" for z in self.Z_vals)
        print(f"  Nh  | {z_header}")
        print("-" * (8 + 10 * self.nZ))
        for inh, Nh in enumerate(self.Nh_vals):
            row = "  ".join(f"{fracs[iz, inh]:.3e}" for iz in range(self.nZ))
            tag = " ← Nh0" if Nh == self.Nh0 else (
                  " (superH)" if Nh > self.Nh0 else "")
            print(f"  {Nh:2d}  | {row}{tag}")
        info = self.summary(n2d)
        print(
            f"\n  <Z> = {info['mean_Z']:+.3f}   <Nh> = {info['mean_Nh']:.2f}   "
            f"peak: C{self.Nc}H{info['peak_Nh']} Z={info['peak_Z']:+d}"
        )


# ------------------------------------------------------------------ #
# Convenience factories                                                #
# ------------------------------------------------------------------ #

def make_c54_solver(n_superh_max: int = N_SUPERH_MAX) -> PAHNetworkSolver:
    """Circumcoronene (C54H18): 6 solo + 12 duo = 18 H."""
    from .pah_h_state import C54H18_NH0, C54H18_SOLO, C54H18_DUO
    return PAHNetworkSolver(54, C54H18_NH0, C54H18_SOLO, C54H18_DUO, n_superh_max)


def make_c96_solver(n_superh_max: int = N_SUPERH_MAX) -> PAHNetworkSolver:
    """Circumcircumcoronene (C96H24): 0 solo + 24 duo = 24 H."""
    from .pah_h_state import C96H24_NH0, C96H24_SOLO, C96H24_DUO
    return PAHNetworkSolver(96, C96H24_NH0, C96H24_SOLO, C96H24_DUO, n_superh_max)


# ------------------------------------------------------------------ #
# Demo / smoke test                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from models.PAH_photophysics.pah_charge_utils import (
        afromNc,
        recombination_rate_Tielens21,
        attachment_rate_Carelli13,
    )

    # ── Environment ────────────────────────────────────────────────────
    Nc  = 54
    T   = 100.0   # K
    n_H = 1e3     # cm⁻³
    n_e = 1e-2    # cm⁻³
    G0  = 1e3     # Habing units
    a_cm = afromNc(Nc)

    # ── Charge rates ───────────────────────────────────────────────────
    # Photoionisation: Bakes & Tielens (1994) scaling ~5e-10 G0 s⁻¹.
    k_ion_0 = 5e-10 * G0
    k_ion = np.array([
        0.3 * k_ion_0,   # Z=-1 → 0  (photodetachment)
        k_ion_0,         # Z= 0 → +1 (first ionisation)
        0.5 * k_ion_0,   # Z=+1 → +2 (second ionisation)
        0.0,             # Z=+2: not in network
    ])

    # recombination_rate_Tielens21 already folds in n_e; pass ne=1 to get coeff
    k_rec_coeff = recombination_rate_Tielens21(Nc, T, 1.0)
    k_rec = np.array([0.0, 0.0, k_rec_coeff, k_rec_coeff])

    k_att_coeff = attachment_rate_Carelli13(T, 1.0)   # cm³/s

    # ── Build solver ───────────────────────────────────────────────────
    solver = make_c54_solver()
    k_Hloss  = np.zeros((solver.nZ, solver.nNh))   # no UV dissociation for demo
    k_H2loss = np.zeros((solver.nZ, solver.nNh))

    env = dict(n_H=n_H, n_e=n_e, T=T, a_pah_cm=a_cm)

    print(f"C{Nc}H18 network: {solver.N} species  "
          f"({solver.nZ} charge states × {solver.nNh} H states)")
    print(f"Environment: T={T} K  n_H={n_H:.0e}  n_e={n_e:.0e}  G0={G0:.0e}\n")

    # ── Mode 1: Steady-state equilibrium ──────────────────────────────
    for meth in ('direct', 'newton'):
        n_eq = solver.solve_equilibrium(
            k_Hloss, k_H2loss, k_ion, k_rec, k_att_coeff,
            **env, n_PAH_total=1.0, method=meth,
        )
        print(f"=== equilibrium  method={meth!r} ===")
        solver.print_distribution(n_eq)
        print()

    # ── Mode 2: Time evolution ─────────────────────────────────────────
    # Start with everything in the neutral normal state
    n0 = np.zeros((solver.nZ, solver.nNh))
    inh_normal = int(np.where(solver.Nh_vals == solver.Nh0)[0][0])
    n0[solver._iz[0], inh_normal] = 1.0

    for meth in ('expm', 'ode'):
        n_t = solver.solve_evolution(
            n0, dt=1e8,
            k_Hloss=k_Hloss, k_H2loss=k_H2loss,
            k_ion=k_ion, k_rec=k_rec, k_att=k_att_coeff,
            **env, method=meth,
        )
        print(f"=== evolution  dt=1e8 s  method={meth!r} ===")
        solver.print_distribution(n_t)
        print()
