"""Equilibrium (steady-state) solvers for the dust/PAH chemistry ODE.

Instead of time-integrating to find dy/dt ≈ 0, these solvers solve
the nonlinear root-finding problem F_dust(y_dust) = 0 directly, where
y_gas is derived at every evaluation via element mass conservation:

    y_gas_e = M0_e − Σ_d  A[e,d] · y_dust_d    (for each element e)

This **mass-conserving constraint** reduces the free variables to the
n_dust components of y_dust and eliminates the degenerate family of
trivial roots that exists in the unconstrained formulation (where
F ≡ 0 whenever y_dust ≡ 0 for arbitrary y_gas).

At a true steady state F_dust = 0 implies F_gas = 0 by mass conservation
(the model rates satisfy d/dt[y_gas_e + Σ A[e,d] y_dust_d] = 0 identically),
so solving the n_dust-dimensional constrained problem is equivalent to the
full (n_el + n_dust)-dimensional problem.

Two algorithms are provided:

``NewtonKrylovEquilibriumSolver``
    Newton outer loop with LGMRES as the inner Krylov solver.
    A diagonal (Jacobi) preconditioner M = diag(|∂F_d/∂y_d|⁻¹) is built
    at the initial point by forward finite differences and held fixed for
    all inner iterations.

``SparseNewtonEquilibriumSolver``
    Classic Newton iterations with a finite-difference full Jacobian
    (n_dust × n_dust, rebuilt at every Newton step), factorised with a
    sparse direct LU solve (``scipy.sparse.linalg.spsolve``), and a
    backtracking Armijo line search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import newton_krylov as _scipy_nk
from scipy.optimize import NoConvergence
from scipy.sparse.linalg import LinearOperator

from .chemistry_state import DustChemistryState
from .rhs import DustProcess


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EquilibriumSolverBase(ABC):
    """Abstract base for equilibrium (steady-state) solvers."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def find_equilibrium(
        self,
        state: "DustChemistryState",
        y_gas_0: np.ndarray,
        y_dust_0: np.ndarray,
        processes: List[DustProcess],
        t_eq_s: float = 1e30,
        verbose: bool = False,
    ):
        """Find the steady-state solution F(y) = 0 subject to mass conservation.

        Parameters
        ----------
        state :
            Fixed gas environment and grain-bin parameters.
        y_gas_0, y_dust_0 :
            Initial densities [g cm⁻³].  Used as starting guess and to
            compute the element totals that define the mass constraint.
        processes :
            Active physics processes from ``build_process_list``.
        t_eq_s :
            Nominal "equilibrium time" in seconds; only used to populate
            the second row of the synthetic history for output compatibility.
        verbose :
            Print per-iteration progress.

        Returns
        -------
        y_gas_eq : ndarray
        y_dust_eq : ndarray
        diag : dict
            Diagnostics dict whose layout matches ``integrate_dust_ode`` output.
        """
        ...


# ---------------------------------------------------------------------------
# Mass-conserving helpers
# ---------------------------------------------------------------------------

def _build_contribution_matrix(state: "DustChemistryState") -> np.ndarray:
    """Build A[n_el, n_dust_total] where A[e, d] = fraction of element e per
    unit mass of y_dust[d].

    Multiplying A @ y_dust gives the element mass currently locked in dust/PAH.
    """
    n_el    = state.n_elements
    n_total = state.npah + state.ndust
    A       = np.zeros((n_el, n_total))

    c_idx = state.el_names.index("C") if "C" in state.el_names else 2
    for pb in state.pah_bins:
        A[c_idx, pb.bin_index] = 1.0                   # all-carbon

    for db in state.dust_bins:
        d = state.npah + db.bin_index
        for e_idx, frac in zip(db.el_indices, db.el_mfractions):
            A[e_idx, d] = frac

    return A


def _make_rhs_conserving(
    state: "DustChemistryState",
    processes: List[DustProcess],
    M0_el: np.ndarray,
    A: np.ndarray,
):
    """Return F_dust(y_dust) with y_gas derived from mass conservation.

    The free variables are the n_dust components of y_dust.
    y_gas = max(M0_el − A @ y_dust, 0).

    Returns F_dust = [dy_dust_d/dt], an n_dust-dimensional vector.
    """
    n_el   = state.n_elements
    n_dust = state.npah + state.ndust

    def F_dust(y_dust_flat: np.ndarray) -> np.ndarray:
        y_dust = np.maximum(y_dust_flat, 0.0)

        # Enforce element budget: A @ y_dust must not exceed M0_el.
        # If any element would be over-drawn, scale all dust bins back
        # proportionally so the feasibility constraint is respected.
        # This prevents the solver from exploring unphysical states where
        # y_gas is clipped to 0 while y_dust grows unboundedly.
        el_in_dust = A @ y_dust
        scale = 1.0
        for e in range(n_el):
            if M0_el[e] > 0.0 and el_in_dust[e] > M0_el[e]:
                scale = min(scale, M0_el[e] / el_in_dust[e])
        if scale < 1.0:
            y_dust = y_dust * scale

        y_gas  = np.maximum(M0_el - A @ y_dust, 0.0)
        dydt_gas  = np.zeros(n_el)
        dydt_dust = np.zeros(n_dust)
        for proc in processes:
            proc.rate_fn(state, y_gas, y_dust, dydt_gas, dydt_dust)
        return dydt_dust

    return F_dust


def _project_to_feasible(
    y_dust: np.ndarray,
    M0_el: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Project y_dust to the feasible region (element budget not exceeded).

    Applies the same scale-back used inside F_dust so that the returned
    solution always satisfies A @ y_dust <= M0_el and y_dust >= 0.
    """
    y = np.maximum(y_dust, 0.0)
    el_in_dust = A @ y
    n_el = len(M0_el)
    scale = 1.0
    for e in range(n_el):
        if M0_el[e] > 0.0 and el_in_dust[e] > M0_el[e]:
            scale = min(scale, M0_el[e] / el_in_dust[e])
    if scale < 1.0:
        y = y * scale
    return y


# ---------------------------------------------------------------------------
# Generic finite-difference utilities
# ---------------------------------------------------------------------------

def _fd_jacobian_diag(
    F,
    y: np.ndarray,
    F0: np.ndarray | None = None,
    eps_rel: float = 1e-7,
) -> np.ndarray:
    """Forward finite-difference diagonal of J: n + 1 F evaluations."""
    if F0 is None:
        F0 = F(y)
    n = len(y)
    d = np.zeros(n)
    for i in range(n):
        h     = max(abs(y[i]) * eps_rel, 1e-200)
        y_p   = y.copy()
        y_p[i] += h
        d[i]  = (F(y_p)[i] - F0[i]) / h
    return d


def _fd_jacobian_full(
    F,
    y: np.ndarray,
    F0: np.ndarray | None = None,
    eps_rel: float = 1e-7,
) -> sp.csr_matrix:
    """Forward finite-difference full Jacobian as CSR: n + 1 F evaluations."""
    if F0 is None:
        F0 = F(y)
    n = len(y)
    J = np.zeros((n, n))
    for j in range(n):
        h      = max(abs(y[j]) * eps_rel, 1e-200)
        y_p    = y.copy()
        y_p[j] += h
        J[:, j] = (F(y_p) - F0) / h
    return sp.csr_matrix(J)


def _diag_preconditioner(J_diag: np.ndarray) -> LinearOperator:
    """M⁻¹ as LinearOperator: M = diag(|J_diag|), capped for stability."""
    n     = len(J_diag)
    scale = np.where(np.abs(J_diag) > 1e-200, 1.0 / np.abs(J_diag), 1.0)
    # Cap to avoid astronomically large corrections
    scale = np.clip(scale, 0.0, 1.0 / max(1e-200, np.max(np.abs(J_diag)) * 1e-20))
    return LinearOperator((n, n), matvec=lambda v: scale * v)


def _synthetic_history(
    y_gas_0:  np.ndarray,
    y_dust_0: np.ndarray,
    y_gas_eq: np.ndarray,
    y_dust_eq: np.ndarray,
    t_eq_s:   float,
) -> dict:
    """2-row history (t=0, t=t_eq) for output / plotting compatibility."""
    NaN = float("nan")
    return {
        "time_s": np.array([0.0, t_eq_s]),
        "y_gas":  np.vstack([y_gas_0,  y_gas_eq]),
        "y_dust": np.vstack([y_dust_0, y_dust_eq]),
        "h_s":    np.array([NaN, NaN]),
        "error":  np.array([NaN, NaN]),
    }


# ---------------------------------------------------------------------------
# Newton-Krylov solver
# ---------------------------------------------------------------------------

class NewtonKrylovEquilibriumSolver(EquilibriumSolverBase):
    """Newton-Krylov root-finding with a diagonal (Jacobi) preconditioner.

    Solves the **mass-conserving** n_dust-dimensional problem
    F_dust(y_dust) = 0 (y_gas derived from element conservation).

    Uses ``scipy.optimize.newton_krylov`` with LGMRES as the inner Krylov
    solver.  A diagonal preconditioner M = diag(|∂F_d/∂y_d|⁻¹) is
    computed once by forward finite differences at ``y_dust_0`` and held
    fixed for all inner iterations.

    Parameters
    ----------
    f_tol :
        Absolute tolerance on ``||F_dust||_∞``.
    f_rtol :
        Relative tolerance: converge when ``||F_dust|| < f_rtol · ||F(y0)||``.
        ``None`` disables the relative criterion.
    maxiter :
        Maximum outer Newton iterations.
    inner_maxiter :
        Maximum LGMRES iterations per Newton step.
    eps_fd :
        Relative step for finite-difference Jacobian diagonal.
    """

    def __init__(
        self,
        f_tol:         float       = 1e-40,
        f_rtol:        float | None = 1e-8,
        maxiter:       int          = 200,
        inner_maxiter: int          = 300,
        eps_fd:        float        = 1e-7,
    ) -> None:
        self.f_tol         = f_tol
        self.f_rtol        = f_rtol
        self.maxiter       = maxiter
        self.inner_maxiter = inner_maxiter
        self.eps_fd        = eps_fd

    @property
    def name(self) -> str:
        return "NewtonKrylov"

    def find_equilibrium(
        self,
        state,
        y_gas_0,
        y_dust_0,
        processes,
        t_eq_s: float = 1e30,
        verbose: bool = False,
    ):
        n_el   = state.n_elements
        n_dust = state.npah + state.ndust

        # --- Build mass-conservation constraint ---
        A      = _build_contribution_matrix(state)
        M0_el  = (np.asarray(y_gas_0, dtype=float)
                  + A @ np.asarray(y_dust_0, dtype=float))
        F_dust = _make_rhs_conserving(state, processes, M0_el, A)

        y_d0  = np.asarray(y_dust_0, dtype=float).copy()
        F0    = F_dust(y_d0)
        F_norm_0  = float(np.linalg.norm(F0))
        nfev      = [1]

        # --- Early exit: already at equilibrium ---
        if F_norm_0 <= self.f_tol:
            y_dust_eq = np.maximum(y_d0, 0.0)
            y_gas_eq  = np.maximum(M0_el - A @ y_dust_eq, 0.0)
            msg = f"already at equilibrium (||F0|| = {F_norm_0:.3e} <= f_tol = {self.f_tol:.3e})"
            diag = dict(
                solver_type   = "equilibrium",
                solver_name   = self.name,
                converged     = True,
                message       = msg,
                F_norm_init   = F_norm_0,
                F_norm_final  = F_norm_0,
                nfev          = nfev[0],
                naccepted     = 0,
                nrejected     = 0,
                icount        = 0,
                nincreased    = 0,
                nreduced      = 0,
                h_min_used    = float("nan"),
                h_max_used    = float("nan"),
                h_mean_used   = float("nan"),
                err_min       = float("nan"),
                err_max       = float("nan"),
                err_mean      = float("nan"),
                history       = _synthetic_history(
                    y_gas_0, y_dust_0, y_gas_eq, y_dust_eq, t_eq_s
                ),
            )
            return y_gas_eq, y_dust_eq, diag

        def F_counted(y):
            nfev[0] += 1
            return F_dust(y)

        # --- Diagonal preconditioner at y_d0 ---
        J_diag = _fd_jacobian_diag(F_dust, y_d0, F0, eps_rel=self.eps_fd)
        nfev[0] += n_dust
        M = _diag_preconditioner(J_diag)

        # --- Effective absolute tolerance ---
        # Use max(f_tol, f_rtol * F_norm_0): the OR-combination means the
        # solver stops as soon as the easier criterion is satisfied.
        # (Using min would demand *both*, making the tolerance impossibly
        # tight when F_norm_0 is already small.)
        if self.f_rtol is not None and F_norm_0 > 0.0:
            f_tol_eff = max(self.f_tol, self.f_rtol * F_norm_0)
        else:
            f_tol_eff = self.f_tol

        conv  = False
        msg   = "did not converge"
        y_sol = y_d0.copy()

        try:
            y_sol = _scipy_nk(
                F_counted,
                y_d0,
                f_tol=f_tol_eff,
                maxiter=self.maxiter,
                method="lgmres",
                inner_maxiter=self.inner_maxiter,
                inner_M=M,
                verbose=verbose,
            )
            conv = True
            msg  = f"converged after {nfev[0]} F evaluations"
        except NoConvergence as exc:
            y_sol = np.asarray(exc.args[0]) if exc.args else y_sol
            msg   = f"no convergence after {nfev[0]} F evaluations"
        except Exception as exc:
            msg = str(exc)

        # Project y_sol back to the feasible region (A @ y_dust <= M0_el).
        # The Newton-Krylov solver can wander into the infeasible zone because
        # F_dust internally scales y_dust down (flat plateau in F outside the
        # budget boundary).  Projecting gives the canonical physical state.
        y_dust_eq = _project_to_feasible(y_sol, M0_el, A)
        y_gas_eq  = np.maximum(M0_el - A @ y_dust_eq, 0.0)
        F_final   = F_dust(y_dust_eq)
        nfev[0]  += 1

        diag = dict(
            solver_type   = "equilibrium",
            solver_name   = self.name,
            converged     = conv,
            message       = msg,
            F_norm_init   = F_norm_0,
            F_norm_final  = float(np.linalg.norm(F_final)),
            nfev          = nfev[0],
            naccepted     = nfev[0],
            nrejected     = 0,
            icount        = nfev[0],
            nincreased    = 0,
            nreduced      = 0,
            h_min_used    = float("nan"),
            h_max_used    = float("nan"),
            h_mean_used   = float("nan"),
            err_min       = float("nan"),
            err_max       = float("nan"),
            err_mean      = float("nan"),
            history       = _synthetic_history(
                y_gas_0, y_dust_0, y_gas_eq, y_dust_eq, t_eq_s
            ),
        )

        return y_gas_eq, y_dust_eq, diag


# ---------------------------------------------------------------------------
# Sparse direct Newton solver
# ---------------------------------------------------------------------------

class SparseNewtonEquilibriumSolver(EquilibriumSolverBase):
    """Newton iterations with a sparse direct LU solve at each step.

    Solves the **mass-conserving** n_dust-dimensional problem
    F_dust(y_dust) = 0 (y_gas derived from element conservation).

    At each iteration the n_dust × n_dust Jacobian is approximated by
    forward finite differences (n_dust + 1 RHS evaluations), converted
    to CSR, and factorised with ``scipy.sparse.linalg.spsolve``.
    A simple backtracking line search (Armijo condition) prevents
    divergence.

    Parameters
    ----------
    rtol :
        Relative convergence: ``||F_dust|| < rtol · ||F(y_dust_0)||``.
    atol :
        Absolute convergence: ``||F_dust|| < atol``.
    maxiter :
        Maximum Newton iterations.
    eps_fd :
        Relative perturbation for finite-difference Jacobian columns.
    alpha_min :
        Minimum backtracking step-length before accepting the full step.
    """

    def __init__(
        self,
        rtol:      float = 1e-8,
        atol:      float = 1e-40,
        maxiter:   int   = 50,
        eps_fd:    float = 1e-7,
        alpha_min: float = 1e-4,
    ) -> None:
        self.rtol      = rtol
        self.atol      = atol
        self.maxiter   = maxiter
        self.eps_fd    = eps_fd
        self.alpha_min = alpha_min

    @property
    def name(self) -> str:
        return "SparseNewton"

    def find_equilibrium(
        self,
        state,
        y_gas_0,
        y_dust_0,
        processes,
        t_eq_s: float = 1e30,
        verbose: bool = False,
    ):
        # --- Build mass-conservation constraint ---
        A     = _build_contribution_matrix(state)
        M0_el = (np.asarray(y_gas_0, dtype=float)
                 + A @ np.asarray(y_dust_0, dtype=float))
        F_fn  = _make_rhs_conserving(state, processes, M0_el, A)

        y      = np.asarray(y_dust_0, dtype=float).copy()
        nfev   = [0]

        def F(yy):
            nfev[0] += 1
            return F_fn(yy)

        F0       = F(y)
        F_norm_0 = float(np.linalg.norm(F0))
        F_norm   = F_norm_0
        conv     = False
        msg      = f"did not converge in {self.maxiter} iterations"
        iter_num = 0

        # --- Early exit: already at equilibrium ---
        if F_norm_0 <= self.atol:
            y_dust_eq = np.maximum(y, 0.0)
            y_gas_eq  = np.maximum(M0_el - A @ y_dust_eq, 0.0)
            msg = f"already at equilibrium (||F0|| = {F_norm_0:.3e} <= atol = {self.atol:.3e})"
            diag = dict(
                solver_type   = "equilibrium",
                solver_name   = self.name,
                converged     = True,
                message       = msg,
                F_norm_init   = F_norm_0,
                F_norm_final  = F_norm_0,
                n_iter        = 0,
                nfev          = nfev[0],
                naccepted     = 0,
                nrejected     = 0,
                icount        = 0,
                nincreased    = 0,
                nreduced      = 0,
                h_min_used    = float("nan"),
                h_max_used    = float("nan"),
                h_mean_used   = float("nan"),
                err_min       = float("nan"),
                err_max       = float("nan"),
                err_mean      = float("nan"),
                history       = _synthetic_history(
                    y_gas_0, y_dust_0, y_gas_eq, y_dust_eq, t_eq_s
                ),
            )
            return y_gas_eq, y_dust_eq, diag

        for iter_num in range(1, self.maxiter + 1):
            F_cur  = F(y)
            F_norm = float(np.linalg.norm(F_cur))

            if verbose:
                print(f"  SparseNewton iter {iter_num:3d}:  ||F_dust|| = {F_norm:.3e}")

            if F_norm < self.atol or (F_norm_0 > 0.0 and F_norm < self.rtol * F_norm_0):
                conv = True
                msg  = f"converged in {iter_num} iterations ({nfev[0]} F evals)"
                break

            # Finite-difference Jacobian (n_dust × n_dust) → CSR
            J_sp = _fd_jacobian_full(F, y, F_cur, eps_rel=self.eps_fd)

            # Sparse LU: J δy = −F
            try:
                delta = spla.spsolve(J_sp, -F_cur)
            except Exception as exc:
                msg = f"spsolve failed at iter {iter_num}: {exc}"
                break

            if not np.all(np.isfinite(delta)):
                msg = f"non-finite Newton step at iter {iter_num}; stopping"
                break

            # Backtracking Armijo line search
            alpha = 1.0
            while alpha > self.alpha_min:
                y_trial = np.maximum(y + alpha * delta, 0.0)
                if float(np.linalg.norm(F(y_trial))) < F_norm:
                    break
                alpha *= 0.5

            y = np.maximum(y + alpha * delta, 0.0)

        F_final    = F(y)
        # Project back to feasible region (same scale-back as inside F_dust)
        y_dust_eq  = _project_to_feasible(y, M0_el, A)
        y_gas_eq   = np.maximum(M0_el - A @ y_dust_eq, 0.0)

        diag = dict(
            solver_type   = "equilibrium",
            solver_name   = self.name,
            converged     = conv,
            message       = msg,
            F_norm_init   = F_norm_0,
            F_norm_final  = float(np.linalg.norm(F_final)),
            n_iter        = iter_num,
            nfev          = nfev[0],
            naccepted     = iter_num,
            nrejected     = 0,
            icount        = iter_num,
            nincreased    = 0,
            nreduced      = 0,
            h_min_used    = float("nan"),
            h_max_used    = float("nan"),
            h_mean_used   = float("nan"),
            err_min       = float("nan"),
            err_max       = float("nan"),
            err_mean      = float("nan"),
            history       = _synthetic_history(
                y_gas_0, y_dust_0, y_gas_eq, y_dust_eq, t_eq_s
            ),
        )

        return y_gas_eq, y_dust_eq, diag


