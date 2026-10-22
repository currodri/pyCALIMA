"""PAH group tables for RAMSES (``pahchem.group_tables``) and the grid-general yields behind them.

Need PAHdb v4.00 (the only library with -1..+2 states) and skip without it.
"""
import numpy as np
import pytest
from test_pahchem_population import _v400

from pycalima.pahchem import fields, photo, population
from pycalima.pahchem import group_tables as G
from pycalima.pahchem.baseline import Params

L0, L1 = (5.6, 11.2, 13.6), (11.2, 13.6, 24.59)


def _molecule(nc=54):
    _, lib = _v400()
    return min((m for m in lib if G.complete(m)), key=lambda m: abs(m.nc - nc))


def _table(mols, params=Params()):
    ks = [G.molecule_kernels((m, params, 0.46, L0, L1, G.N_E_PER_GROUP, G.LT[::10])) for m in mols]
    return G.combine(ks, np.ones(len(ks)), L0, L1, lT=G.LT[::10])


@pytest.mark.parametrize("detachment", ["jochims", "statistical", "unity"])
def test_yields_on_another_grid_agree_with_E_PH_at_shared_nodes(detachment):
    mol, p = _molecule(), Params(detachment=detachment)
    E = np.concatenate([np.linspace(1.0, 4.9, 40), population.E_PH[::7]])
    shared = np.isin(E, population.E_PH)
    idx = np.searchsorted(population.E_PH, E[shared])
    for z in G.CHARGES:
        y = population._yield(mol, z, p, energy_ev=E)
        assert np.array_equal(y[shared], population._yield(mol, z, p)[idx])
        ek = population.electron_energy(mol, z, p, energy_ev=E)
        assert np.allclose(ek[shared], population.electron_energy(mol, z, p)[idx], rtol=0, atol=1e-12)


@pytest.mark.parametrize("detachment", ["jochims", "statistical"])
def test_photons_between_the_electron_affinity_and_5_ev_detach_anions(detachment):
    mol = _molecule()
    E = np.linspace(0.5, 4.9, 200)
    y = population._yield(mol, -1, Params(detachment=detachment), energy_ev=E)
    assert (y[E < mol.ionisation_ev[-1]] == 0).all()
    assert y[E > mol.ionisation_ev[-1] + 1.0].max() > 0.1


def test_group_photo_rate_is_the_blackbody_integral():
    """At a tabulated T_s, F_g K_g is int sigma Y F dE over the group for a blackbody of that T_s."""
    mol = _molecule()
    tab = _table([mol])
    ts = 10 ** tab["log10_Ts"][12]
    bb = fields.blackbody(ts, 1.0, e_min=L0[0], e_max=L1[-1], n=40001)
    F, egy = G.group_fluxes(bb, L0, L1)
    gt = G.PAHGroupTable(tab, egy=egy)
    p = Params()
    for i, z in enumerate(G.CHARGES):
        for g, (lo, hi) in enumerate(zip(L0, L1)):
            E = np.geomspace(lo, hi, 4001)
            direct = np.trapezoid(photo.sigma_li_draine(mol.nc, z, E) * population._yield(mol, z, p, energy_ev=E)
                                  * bb.flux_at(E), E)
            assert gt.Kg[i, g] * F[g] == pytest.approx(direct, rel=2e-3, abs=1e-30)


def test_steady_state_passes_the_ions_charge_to_the_electrons():
    """Electrons made minus captured equals minus the ions neutralised (the chain's fluxes balance)."""
    _, lib = _v400()
    mols = [m for m in lib if G.complete(m) and 50 <= m.nc < 60][:3]
    gt = G.PAHGroupTable(_table(mols))
    ions = (("C+", 0.02), ("H+", 0.01), ("S+", 1e-3))
    r = gt.solve([1e7, 1e6, 1e5], 100.0, 0.03, n_h=30.0, ions=ions, G0_bg=1.7)
    assert r["f"].sum() == pytest.approx(1.0)
    neutralised = sum(n * (r["alpha_mn"][i] + r["alpha_ct"][i]) for i, n in ions)
    assert r["electron_source"] == pytest.approx(-neutralised, rel=1e-9)


def test_bin_table_survives_a_round_trip(tmp_path):
    _, lib = _v400()
    tab = _table([m for m in lib if G.complete(m) and 30 <= m.nc < 34][:2])
    back = G.read_bin_table(G.write_bin_table(tab, tmp_path / "t.dat"))
    for key in ("A", "K", "H", "egy", "A_bg", "K_bg", "H_bg", "alpha", "L", "k_mn", "k_ct", "lT", "log10_Ts"):
        assert np.allclose(back[key], tab[key], rtol=1e-11, atol=0), key
    assert back["ions"] == tab["ions"] and back["nc"] == pytest.approx(tab["nc"])
