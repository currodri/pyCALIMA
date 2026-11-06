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


@pytest.mark.parametrize("detachment", ["jochims", "statistical", "valence", "unity"])
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


@pytest.mark.parametrize("detachment", ["jochims", "statistical", "valence", "unity"])
def test_yields_follow_the_order_of_an_unsorted_grid(detachment):
    """molecule_kernels passes the group nodes (to 500 eV) and then BG_GRID (1-13.6 eV), so the grid is unsorted."""
    mol, p = _molecule(), Params(detachment=detachment)
    E = population.E_PH[::5]
    perm = np.random.default_rng(3).permutation(E.size)
    for z in G.CHARGES:
        assert np.array_equal(population._yield(mol, z, p, energy_ev=E[perm]),
                              population._yield(mol, z, p, energy_ev=E)[perm])
        assert np.allclose(population.electron_energy(mol, z, p, energy_ev=E[perm]),
                           population.electron_energy(mol, z, p, energy_ev=E)[perm], rtol=0, atol=1e-12)


@pytest.mark.parametrize("yield_model", ["jochims", "lepage"])
def test_yields_never_fall_with_photon_energy(yield_model):
    """Le Page's law is a bump about 14.9 eV; above 13.6 eV the population ramps it to unity instead."""
    mol, p = _molecule(), Params(yield_model=yield_model, yield_cation="jochims")
    for z in (0, 1):
        assert np.all(np.diff(population._yield(mol, z, p)) >= -1e-12)


def test_valence_detachment_adds_a_channel_and_conserves_the_excess():
    """The valence channel adds prompt detachment above ``anion_valence_threshold`` (>= EA): the yield is a
    probability at least the statistical one, and no electron takes more than E - EA."""
    mol = _molecule(96)
    E = population.E_PH
    y_s, e_s = population.statistical_detachment(mol)
    y_v, e_v = population.statistical_detachment(mol, valence=True)
    ea = mol.ionisation_ev[-1]
    assert population.anion_valence_threshold(mol) >= ea
    assert np.all((y_v >= y_s - 1e-12) & (y_v <= 1 + 1e-12))
    assert np.all(e_v <= np.clip(E - ea, 0, None) + 1e-9) and np.all(e_v >= 0)
    mid = (E >= 5.0) & (E <= 9.0)          # below ~10 eV a large anion detaches only through the valence channel
    assert y_v[mid].min() > 0.15 and y_s[mid].max() < 0.1


def test_the_dication_is_the_top_state_of_every_molecule():
    """Molecules with a +3 state in PAHdb ionise out of +2 in the tables only above Y_UNITY_EV, as the rest do."""
    _, lib = _v400()
    mol = next(m for m in lib if G.complete(m) and max(m.charges) > 2)
    k = G.molecule_kernels((mol, Params(), 0.46, L0, L1, G.N_E_PER_GROUP, G.LT[::10]))
    nodes = np.concatenate(G._group_nodes(L0, L1, G.N_E_PER_GROUP)).reshape(len(L0), -1)
    assert np.all(k["K"][-1][nodes < population.Y_UNITY_EV] == 0.0) and np.all(k["K_bg"][-1] == 0.0)
    assert np.any(population._yield(mol, 2, Params(), energy_ev=nodes.ravel()) > 0)      # Population follows +3


@pytest.mark.parametrize("detachment", ["jochims", "statistical", "valence"])
def test_photons_between_the_electron_affinity_and_5_ev_detach_anions(detachment):
    mol = _molecule()
    E = np.linspace(0.5, 4.9, 200)
    y = population._yield(mol, -1, Params(detachment=detachment), energy_ev=E)
    assert (y[E < mol.ionisation_ev[-1]] == 0).all()
    assert y[E > mol.ionisation_ev[-1] + 1.0].max() > 0.1


def test_group_photo_rate_is_the_blackbody_integral():
    """At a tabulated T_s, F_g K_g is int sigma Y F dE over the group for a blackbody of that T_s (with the
    table's chain, +2 the top state)."""
    mol = _molecule()
    tab = _table([mol])
    mol = G._chain(mol)
    ts = 10 ** tab["log10_Ts"][12]
    bb = fields.blackbody(ts, 1.0, e_min=L0[0], e_max=L1[-1], n=40001)
    F, egy = G.group_fluxes(bb, L0, L1)
    gt = G.PAHGroupTable(tab, egy=egy)
    p = Params()
    for i, z in enumerate(G.CHARGES):
        for g, (lo, hi) in enumerate(zip(L0, L1)):
            E = np.geomspace(lo, hi, 4001)
            direct = np.trapezoid(photo.sigma_uv(mol.nc, z, E) * population._yield(mol, z, p, energy_ev=E)
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
    assert r["valid"] and not gt.solve([1e7, 1e6, 1e5], 1e6, 0.03)["valid"]       # no hot-gas collisions (audit m12)


def test_bin_table_survives_a_round_trip(tmp_path):
    _, lib = _v400()
    tab = _table([m for m in lib if G.complete(m) and 30 <= m.nc < 34][:2])
    path = G.write_bin_table(tab, tmp_path / "t.dat")
    back = G.read_bin_table(path)
    assert f"valid to T = {G.T_VALID_MAX_K:g} K" in path.read_text().splitlines()[0]
    for key in ("A", "K", "H", "egy", "A_bg", "K_bg", "H_bg", "alpha", "L", "k_mn", "k_ct", "lT", "log10_Ts"):
        assert np.allclose(back[key], tab[key], rtol=1e-11, atol=0), key
    assert back["ions"] == tab["ions"] and back["nc"] == pytest.approx(tab["nc"])
