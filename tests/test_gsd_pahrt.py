"""gsd.pahrt, the per-cell PAH equilibrium, on a synthetic molecule (no pahchem): invariants of the solver."""
import numpy as np
import pytest

from pycalima.gsd.pahrt import MoleculeSolver, PAHrtBin, PAHrtCell
from pycalima.pahchem.pahrt_kernels import OBSERVABLES, MoleculeKernels

pytestmark = pytest.mark.gsd_quick

PROCS = [("ionisation (dZ +1)", 1, 0, 0), ("electron capture (dZ -1)", -1, 0, 0), ("H loss", 0, -1, 0),
         ("H addition", 0, 1, 0), ("H2 loss", 0, -2, 1)]
G0 = np.array([1.0, 10.0, 100.0])
F_UNIT = 1e8


def toy(nc=30, h_scale=1.0, multiphoton=False):
    """Charges -1, 0, +1 x N_H 0, 1, 2 (N_H0 = 1), one group, two slopes, collisions with e, H and C+."""
    nz, nn, npr, no = 3, 3, len(PROCS), len(OBSERVABLES)
    pk = np.zeros((1, 2, G0.size, npr, nz, nn))
    pw = np.zeros((1, 2, G0.size, no, nz, nn))
    for i, g0 in enumerate(G0):
        pk[0, :, i, 0, :2, :] = 1e-17                                   # ionisation out of -1 and 0
        mp = (g0 / G0[0]) if multiphoton else 1.0                        # kappa ~ F: rate ~ F^2
        pk[0, :, i, 2, :, 1:] = 1e-20 * h_scale * mp                     # H loss from N_H >= 1
        pk[0, :, i, 4, :, 2] = 1e-21 * h_scale                           # H2 loss from N_H = 2
        pw[0, :, i, OBSERVABLES.index("absorbed")] = 1e-17 * 10.0 * 1.602e-12
        pw[0, :, i, OBSERVABLES.index("heating")] = 1e-17 * 1.602e-12
        pw[0, :, i, OBSERVABLES.index("carbon")][:, 0] = 1e-19 * mp
    lt = np.array([1.0, 2.0, 3.0, 4.0])
    ck = np.zeros((3, lt.size, npr, nz, nn))
    cw = np.zeros((3, lt.size, no, nz, nn))
    ck[0, :, 1, 1:, :] = 1e-6                                           # e: captures by 0 and +1
    cw[0, :, OBSERVABLES.index("cooling"), 1:, :] = 1e-6 * 1.602e-12
    ck[1, :, 3, :, :2] = 1e-10 * h_scale                                 # H: addition below N_H = 2
    ck[2, :, 0, 0, :] = 1e-7                                             # C+: mutual neutralisation of -1
    return MoleculeKernels(f"toy{nc}", nc, 1, (-1, 0, 1), (0, 1, 2), list(PROCS), np.array([5.6]), np.array([13.6]),
                           np.array([True]), np.array([1.0, 3.0]), G0, np.array([[10.0, 8.0]]),
                           np.full((1, 2), F_UNIT), pk, pw, lt, ("e", "H", "C+"), ck, cw)


def cell(F=1e9, n_e=0.1, n_h=100.0, n_cp=0.05):
    return PAHrtCell(np.array([F]), np.array([9.0]), 100.0, n_e, n_h, (("C+", n_cp),))


def test_steady_state_normalised_and_nonnegative():
    for method in ("full", "nested"):
        F, _ = MoleculeSolver(toy()).solve(cell(), method)
        assert F.min() >= 0.0
        assert F.sum() == pytest.approx(1.0, rel=1e-12)


def test_nested_equals_full_when_charging_is_fast():
    ms = MoleculeSolver(toy(h_scale=1e-6))
    Ff, of = ms.solve(cell(), "full")
    Fn, on = ms.solve(cell(), "nested")
    np.testing.assert_allclose(Fn, Ff, rtol=1e-4, atol=1e-12)
    assert on["z_mean"] == pytest.approx(of["z_mean"], abs=1e-6)


def test_photon_rates_linear_and_multiphoton_quadratic():
    lin, mp = MoleculeSolver(toy()), MoleculeSolver(toy(multiphoton=True))
    for F in (2e8, 2e9):
        k1, _ = lin.photon(cell(F))
        k2, _ = lin.photon(cell(10 * F))
        np.testing.assert_allclose(k2[0], 10 * k1[0], rtol=1e-12)               # ionisation ~ F
        m1, _ = mp.photon(cell(F))
        m2, _ = mp.photon(cell(10 * F))
        np.testing.assert_allclose(m2[2][:, 1:], 100 * m1[2][:, 1:], rtol=1e-9)  # kappa ~ F: H loss ~ F^2


def test_electron_source_balances_ion_recombination():
    """In steady state the charge is constant: the net electrons freed equal the charge the ions take."""
    n_cp = 0.05
    for method in ("full", "nested"):
        _, o = MoleculeSolver(toy()).solve(cell(n_cp=n_cp), method)
        assert o["alpha(C+)"] > 0
        assert o["e_source"] == pytest.approx(-n_cp * o["alpha(C+)"], rel=1e-8)


def test_bin_weights_and_outputs_linear_in_them():
    b = PAHrtBin([MoleculeSolver(toy(nc)) for nc in (25, 30, 38)], 20.0, 42.0)
    for s in (0.0, 0.3, 0.5, 1.0):
        w = b.weights(s)
        assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0, rel=1e-12)
        mean, outs, w = b.solve(cell(), s)
        assert mean["z_mean"] == pytest.approx(sum(wi * o["z_mean"] for wi, o in zip(w, outs, strict=True)),
                                               rel=1e-12)
    assert b.weights(1.0)[-1] > b.weights(0.0)[-1]                     # larger s: more weight on the heavy end


def test_kernels_round_trip(tmp_path):
    mk = toy()
    mk.save(tmp_path / "toy.npz")
    back = MoleculeKernels.load(tmp_path / "toy.npz")
    assert back.procs == [tuple(p) for p in mk.procs] and back.charges == mk.charges
    np.testing.assert_array_equal(back.photon_k, mk.photon_k)
    np.testing.assert_array_equal(back.coll_k, mk.coll_k)
