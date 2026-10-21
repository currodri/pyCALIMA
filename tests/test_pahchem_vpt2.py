"""VPT2 anharmonic constants (pahchem.vpt2): analytic limits on synthetic force fields, then the Mackie+18 force
fields against PAHdb anharmonic pyrene and the Demyk+26 lab temperature laws (skipped without those datasets)."""

import numpy as np
import pytest

from pycalima.pahchem import vpt2


def _ff(omega, cubic=(), quartic=()):
    n = len(omega)
    c = np.zeros((n, n, n))
    for (i, j, k), v in cubic:
        for a, b, d in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            c[a, b, d] = v
    q = np.zeros((n, n))
    for (i, j), v in quartic:
        q[i, j] = q[j, i] = v
    return vpt2.ForceField(np.asarray(omega, float), c, q)


def test_quartic_terms_alone():
    chi = vpt2.anharmonic_constants(_ff([1000.0, 1700.0], quartic=[((0, 0), 8.0), ((0, 1), 12.0)]))
    assert chi[0, 0] == pytest.approx(8.0 / 16) and chi[0, 1] == pytest.approx(12.0 / 4) and chi[1, 1] == 0.0


def test_a_stiff_mode_eliminated_adiabatically_lowers_the_coupling():
    """omega_k >> omega_i, omega_j: chi_ij -> -phi_ijk^2 / (2 omega_k) (minimising V over q_k)."""
    phi, wk = 30.0, 1e6
    chi = vpt2.anharmonic_constants(_ff([700.0, 1100.0, wk], cubic=[((0, 1, 2), phi)]))
    assert chi[0, 1] == pytest.approx(-phi**2 / (2 * wk), rel=1e-5)


def test_cubic_terms_match_the_closed_form_away_from_resonance():
    w = np.array([800.0, 1300.0, 3100.0])
    p_iik, p_ijk = 25.0, 18.0
    chi = vpt2.anharmonic_constants(_ff(w, cubic=[((0, 0, 2), p_iik), ((0, 1, 2), p_ijk)]), delta_hi=1.0, delta_lo=1.0)
    wi, wj, wk = w
    D = (wi + wj + wk) * (wi - wj - wk) * (-wi + wj - wk) * (-wi - wj + wk)
    assert chi[0, 1] == pytest.approx(-0.5 * p_ijk**2 * wk * (wk**2 - wi**2 - wj**2) / D)
    assert chi[0, 0] == pytest.approx(-p_iik**2 * (8 * wi**2 - 3 * wk**2) / (wk * (4 * wi**2 - wk**2)) / 16)


def test_a_fermi_resonance_is_removed_not_divergent():
    w = [1500.0, 1600.0, 3100.5]                       # 1500 + 1600 ~ 3100.5: type 2
    ff = _ff(w, cubic=[((0, 1, 2), 40.0)])
    chi = vpt2.anharmonic_constants(ff)
    assert np.all(np.isfinite(chi)) and abs(chi[0, 1]) < 1.0
    assert ("2", 0, 1, 2) in ff.resonances


@pytest.fixture(scope="module")
def mackie():
    from pycalima._datasets import get_dataset

    root = get_dataset("mackie2018-supplement").locate()
    if root is None:
        pytest.skip("Mackie+18 supplement not imported (calima-fetch-data import mackie2018-supplement ...)")
    return root


def test_pyrene_isolated_fundamentals_are_pahdb_anharmonic(mackie):
    """Away from the CH stretches (whose polyads PAHdb includes), the VPT2 fundamentals sit on PAHdb's lines."""
    from pycalima._datasets import get_dataset
    from pycalima.pahchem import ames_suite

    if get_dataset("pahdb-anharmonic-v1-00").locate() is None:
        pytest.skip("PAHdb anharmonic v1.00 not installed")
    ff = vpt2.mackie2018_force_field("pyrene", mackie)
    nu = vpt2.fundamentals(ff.omega, vpt2.anharmonic_constants(ff))
    rec = next(r for r in ames_suite.read_xml(ames_suite.xml_path("pahdb-anharmonic-v1-00"))
               if r["formula"] == "C16H10" and int(r["charge"]) == 0)
    f, inten = np.asarray(rec["freqs_cm"]), np.asarray(rec["intensity"])
    strong = f[np.argsort(-inten)[:30]]
    strong = strong[strong < 1500.0]
    d = np.array([np.min(np.abs(nu - x)) for x in strong])
    assert np.median(d) < 1.0 and np.mean(d < 2.5) > 0.6


def test_6h_pyrene_ch2_thermal_shift_is_the_demyk26_law(mackie):
    from pycalima._datasets import get_dataset
    from pycalima.pahchem import cosmicpah

    if get_dataset(cosmicpah.DATASET).locate() is None:
        pytest.skip("CosmicPAH-IRDB not fetched")
    ff = vpt2.mackie2018_force_field("123678-hexahydropyrene", mackie)
    chi = vpt2.anharmonic_constants(ff)
    kb = 0.6950348

    def shift(t, k):
        n = 1.0 / np.expm1(ff.omega / (kb * t))
        return chi[k] @ n + chi[k, k] * n[k]

    nu = vpt2.fundamentals(ff.omega, chi)
    ch2 = [k for k in range(ff.omega.size) if 2915 < nu[k] < 2935]       # the asymmetric CH2 stretches
    slopes = [np.mean([(shift(b, k) - shift(a, k)) / (b - a) for k in ch2]) for a, b in ((423, 523), (523, 673))]
    lab = [p.c1 for p in cosmicpah.band("6h-pyrene", 2942.7).position_pieces]
    assert slopes == pytest.approx(lab, rel=0.25)
