"""Grain relative velocities: the collapse matrices, the solver's scalar
function they must agree with, and benchmark M4 (Marchand+21 Fig. 4).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycalima.collapse import velocities as vel
from pycalima.collapse.coagulation import intermediate_turbulence_kernel
from pycalima.solvers.grain_dynamics import SIGMA_NEUTRAL, grain_relative_velocity

KB, MH = 1.380649e-16, 1.6735575e-24


def test_brownian_velocity_is_the_maxwellian_mean_relative_speed():
    """<|v1 - v2|> = (8 k T / pi m_red)^1/2, for the solver and the matrices."""
    T, m = 50.0, np.array([1e-17, 4e-15])
    expect = math.sqrt(8 * KB * T / (math.pi * m[0] * m[1] / m.sum()))
    assert vel.brownian_dv(T, m)[0, 1] == pytest.approx(expect, rel=1e-12)
    v = grain_relative_velocity("Ormel2007", T, 1e-20, 1e4, 0.0, 2.3, 0.0,
                                1e-6, 3.0, m[0], 1e-5, 3.0, m[1])
    assert v == pytest.approx(expect, rel=1e-6)


@pytest.mark.parametrize("nH,T", [(1e2, 100.0), (1e6, 10.0), (1e10, 300.0)])
def test_matrices_reproduce_the_solver_velocity(nH, T):
    """Same driving: v_g = v_turb, tau_L = L / v_turb, tau_eta from the neutral
    viscosity. A 1 nm - 1 mm grid crosses at least two Ormel & Cuzzi regimes
    at every density, and all three at the lowest."""
    mu, rho_s, v_turb, L = 2.3, 3.0, 3e4, 3e17
    rho_g = mu * MH * nH
    a = np.geomspace(1e-7, 1e-1, 25)
    m = 4.0 / 3.0 * math.pi * rho_s * a ** 3
    v_th = vel.thermal_speed(T, mu)
    Re = max(3.0 * v_turb * L * nH * SIGMA_NEUTRAL / v_th, 1.0)   # neutral gas: nu = v_th / (3 n sigma)
    tau_L = L / v_turb
    t_s = vel.epstein_stopping_time(a, rho_s, T, rho_g, mu)
    regimes = {int(r) for r in np.digitize(t_s, [tau_L / math.sqrt(Re), tau_L])}
    dv = np.sqrt(vel.brownian_dv(T, m) ** 2
                 + vel.ormel_cuzzi_dv(t_s, v_turb, tau_L, tau_L / math.sqrt(Re), y_star="approx") ** 2)
    for i in range(0, 25, 3):
        for j in range(0, 25, 4):
            ref = grain_relative_velocity("Ormel2007", T, rho_g, nH, v_turb, mu, L,
                                          a[i], rho_s, m[i], a[j], rho_s, m[j])
            assert dv[i, j] == pytest.approx(ref, rel=2e-3)   # differing constants only
    assert len(regimes) >= 2 and (nH > 1e3 or regimes == {0, 1, 2})


def _oc(St1, St2, Re=1e8, y_star="exact"):
    """dV / V_g for two Stokes numbers (V_g**2 = 3/2 v_g**2, so v_g = (2/3)**1/2)."""
    return vel.ormel_cuzzi_dv(np.array([St1, St2]), math.sqrt(2.0 / 3.0), 1.0, Re ** -0.5, y_star)[0, 1]


@pytest.mark.parametrize("y_star", ["exact", "approx"])
@pytest.mark.parametrize("ratio", [0.0, 1.0 / 3.0, 1.0])
def test_turbulent_velocity_is_continuous_across_the_regimes(y_star, ratio):
    """Ormel & Cuzzi (2007) Eqs. 16-18 are continuous: halving the step halves
    the largest step in ln dV, across t_eta (St = 1e-4 here) and t_L (St = 1).
    Switching between their limit formulas instead jumps by up to x2.4. The
    steepest part is the onset of class II eddies at St1 = Re^-1/2 / y*, the
    "turning point" near t_eta of their Fig. 3."""
    steps = []
    for n in (3000, 6000):
        St = np.geomspace(2e-5 if ratio < 1 else 2e-4, 30.0, n)
        v = np.array([_oc(s, max(ratio * s, 1e-14), y_star=y_star) for s in St])
        steps.append(np.max(np.abs(np.diff(np.log(v)))))
    assert steps[1] == pytest.approx(steps[0] / 2.0, rel=0.1)
    assert steps[0] < 0.05      # at d ln St = 0.005: a kink, not a jump


def test_turbulent_velocity_limits_are_their_regime_formulas():
    Re = 1e8
    # tightly coupled, Eq. 27: dV^2 = V_g^2 (t_L / t_eta) (St1 - St2)^2
    assert _oc(1e-6, 3e-7) ** 2 == pytest.approx(Re ** 0.5 * (1e-6 - 3e-7) ** 2, rel=0.02)
    # fully intermediate, Eq. 28 with y_a = 1.6: ~3.0 St1 (t2 << t1), 2.0 St1 (t1 = t2)
    St, Re_big = 1e-4, 1e16
    assert _oc(St, 1e-14, Re_big, "approx") ** 2 / St == pytest.approx(3.0, rel=0.03)
    assert _oc(St, St, Re_big, "approx") ** 2 / St == pytest.approx(2.0, rel=0.03)
    # heavy, Eq. 29: dV^2 = V_g^2 (1/(1+St1) + 1/(1+St2))
    assert _oc(50.0, 3.0) ** 2 == pytest.approx(1 / 51 + 1 / 4, rel=1e-6)


def test_turbulent_velocity_matches_their_fig3():
    """Ormel & Cuzzi (2007) Fig. 3 (Re = 1e8): dV / V_g ~ 1e-3 at St1 = 1e-5 and
    ~1e-2 at 1e-4 (t1 >> t2); peaks of ~1.2 (t1 >> t2) and 1.0 (t1 = t2) at St1 ~ 1."""
    assert _oc(1e-5, 1e-14) == pytest.approx(1e-3, rel=0.1)
    assert _oc(1e-4, 1e-14) == pytest.approx(1.2e-2, rel=0.1)
    St = np.geomspace(0.1, 10, 201)
    assert max(_oc(s, 1e-14) for s in St) == pytest.approx(1.22, rel=0.05)
    assert max(_oc(s, s) for s in St) == pytest.approx(1.0, rel=0.05)


def test_solver_velocity_is_symmetric_in_target_and_projectile():
    args = ("Ormel2007", 30.0, 2.3 * MH * 1e5, 1e5, 2e4, 2.3, 1e17)
    g1, g2 = (3e-6, 3.0, 4 / 3 * math.pi * 3.0 * 3e-6 ** 3), (2e-4, 3.0, 4 / 3 * math.pi * 3.0 * 2e-4 ** 3)
    assert grain_relative_velocity(*args, *g1, *g2) == pytest.approx(grain_relative_velocity(*args, *g2, *g1), rel=1e-14)


def test_ambipolar_drift_separates_charged_grains_only():
    t_s, m = np.array([1e9, 1e10, 1e10]), np.array([1e-16, 1e-14, 1e-14])
    dv = vel.ambipolar_dv(t_s, m, np.array([0.0, 0.0, -3.0]), B=3e-5, V_AD=2e3)
    assert dv[0, 1] == 0.0               # neutral grains move with the neutrals
    assert dv[0, 2] > 0.0
    assert dv[0, 2] <= 2e3 * math.sqrt(2.0)


def test_coulomb_factor_limits():
    a, m = np.array([1e-6, 1e-6]), np.array([1e-17, 1e-17])
    fast = vel.coulomb_factor(a, m, np.array([-1.0, -1.0]), np.full((2, 2), 1e6))
    slow = vel.coulomb_factor(a, m, np.array([-1.0, -1.0]), np.full((2, 2), 1.0))
    attract = vel.coulomb_factor(a, m, np.array([-1.0, 1.0]), np.full((2, 2), 1e3))
    assert fast[0, 1] == pytest.approx(1.0, abs=1e-3)
    assert slow[0, 1] == 0.0
    assert attract[0, 1] > 1.0


def test_fig4_fragmentation_boundary_scales_as_their_map():
    """Marchand+21 Fig. 4: E_kin = 5 E_roll along a ∝ n_H^(1/8) (from dv ∝ n^-1/4 a^1/2,
    E_kin ∝ a^4 n^-1/2). Their white (ratio = 1) band, read off the figure, sits at
    0.45, 0.8, 1.4, 2.4, 4.3 um for n_H = 1e4 ... 1e12: the same slope, at a 1.25x
    higher a than equal-mass partners give. Their partner mass is not stated; a
    partner ~4x lighter than the larger grain gives their 2.4x smaller E_kin."""
    E5 = 5.0 * vel.rolling_energy(370.0, 0.05e-4)
    a = np.geomspace(1e-5, 1e-3, 4001)
    m = 4.0 / 3.0 * math.pi * 2.3 * a ** 3
    bounds = []
    for nH in 10.0 ** np.arange(4, 13, 2):
        K = intermediate_turbulence_kernel(nH, 10.0, 2.3, mu=2.3, T_power=-0.25)(None, a)
        dv = np.diag(K) / (math.sqrt(8 / (3 * math.pi)) * math.pi * (2 * a) ** 2)
        bounds.append(a[np.argmin(np.abs(np.log(0.5 * (m / 2) * dv ** 2 / E5)))])
    bounds = np.array(bounds) * 1e4
    theirs = np.array([0.45, 0.8, 1.4, 2.4, 4.3])
    slope = np.polyfit(np.arange(4, 13, 2), np.log10(bounds), 1)[0]
    assert slope == pytest.approx(1.0 / 8.0, rel=0.02)
    np.testing.assert_allclose(theirs / bounds, 1.25, rtol=0.1)


# ---------------------------------------------------------------------------
# collision speeds with a velocity distribution (Guillet+20, Eqs. 18-19, App. C)
# ---------------------------------------------------------------------------

def _numeric_mean_speed(dV, V, mu):
    from scipy.integrate import quad
    a = math.sqrt(1.5) / dV
    if mu == 0.0:
        f = lambda v: 4.0 / math.sqrt(math.pi) * a ** 3 * v * v * math.exp(-(a * v) ** 2)
    else:
        f = lambda v: a / (math.sqrt(math.pi) * mu) * v * (math.exp(-(a * (v - mu)) ** 2)
                                                           - math.exp(-(a * (v + mu)) ** 2))
    top = V if np.isfinite(V) else mu + 20.0 * dV
    return quad(lambda v: v * f(v), 0.0, top, limit=400)[0]


@pytest.mark.parametrize("mu", [0.0, 1e-4, 0.1, 1.0, 3.0])
@pytest.mark.parametrize("V", [0.3, 1.0, 3.0, np.inf])
def test_mean_collision_speed_matches_quadrature_of_the_skewed_maxwellian(mu, V):
    ours = vel.mean_collision_speed(1.0, V, mu)[()]
    assert ours == pytest.approx(_numeric_mean_speed(1.0, V, mu), rel=1e-6, abs=1e-12)


def test_mean_collision_speed_limits():
    """No limit, no drift: the Maxwellian mean speed (8/3 pi)^1/2 dV; their Eq. 18
    with a limit; and the drift-dominated limit mu + dV^2 / (3 mu)."""
    assert vel.mean_collision_speed(2.0)[()] == pytest.approx(math.sqrt(8 / (3 * math.pi)) * 2.0, rel=1e-12)
    chi = math.sqrt(1.5) * 0.7
    eq18 = math.sqrt(8 / (3 * math.pi)) * (1 - (1 + chi ** 2) * math.exp(-chi ** 2))
    assert vel.mean_collision_speed(1.0, 0.7)[()] == pytest.approx(eq18, rel=1e-12)
    assert vel.mean_collision_speed(1.0, np.inf, 30.0)[()] == pytest.approx(30 + 1 / 90, rel=1e-6)


def test_brownian_rms_and_mean_are_consistent():
    """Combining rms in quadrature and taking the mean speed recovers the
    Brownian mean speed when there is no other motion."""
    m = np.array([1e-17, 3e-16])
    mean = vel.mean_collision_speed(vel.brownian_rms(30.0, m))
    np.testing.assert_allclose(mean, vel.brownian_dv(30.0, m), rtol=1e-12)
