"""Population-weighted IR emission of a PAH family (pahchem.family_emission), M7b.

Uses the PAHdb modes of each charge of the parent in model_data/PAH_states
(C54H18_<Z>.dat for Z = -1..+2). Measured values are recorded in the docstrings.
"""

import numpy as np
import pytest

from pycalima.pahchem.baseline import BaselineModel, Environment, NGC7023_EDGE
from pycalima.pahchem.family_emission import FamilyEmission, band_fluxes
from pycalima.pahchem.molecule import PAHSpecies

GRID = np.arange(500.0, 3500.0, 0.5)


_CACHE = {}


@pytest.fixture
def c54(model_data):
    if "c54" not in _CACHE:
        files = {z: model_data / "PAH_states" / f"C54H18_{z}.dat" for z in (-1, 0, 1, 2)}
        if not all(p.exists() for p in files.values()):
            pytest.skip("model_data/PAH_states/ lacks the C54H18 charge ladder (-1..+2)")
        species = {z: PAHSpecies.from_states_file(p) for z, p in files.items()}
        _CACHE["c54"] = FamilyEmission(BaselineModel("C54", species[0], anion=species[-1]), species)
    return _CACHE["c54"]


def test_family_emits_what_it_absorbs(c54):
    """sum_states f sum_i h nu_i R_i = sum_states f r_heat at the NGC 7023 edge (measured
    6e-16): every state's steady state radiates its absorbed heating power."""
    fr = c54.fractions(NGC7023_EDGE)
    states = c54.state_emission(NGC7023_EDGE)
    emitted = sum(w * states[k][0] @ states[k][1] for k, w in fr.items())
    absorbed = sum(w * states[k][2] for k, w in fr.items())
    assert emitted == pytest.approx(absorbed, rel=2e-3)
    assert sum(fr.values()) == pytest.approx(1.0)


def test_charge_sets_the_band_ratios(c54):
    """Each charge alone at the NGC 7023 edge (fully hydrogenated): 6.2/11.2 = 0.34
    (neutral), 5.2 (cation), 9.5 (dication), 20 (anion); 3.3/11.2 falls from 0.72
    (neutral) to 0.28 (cation) and 0.07 (dication). The PAHdb ion spectra carry
    the C-C bands; this is what makes 6.2/11.2 trace ionisation."""
    r = {}
    for z in (-1, 0, 1, 2):
        one = {(zz, n): float(zz == z and n == c54.model.nh0) for zz in c54.model.charges for n in c54.model.nh_values}
        b = band_fluxes(GRID, c54.spectrum(NGC7023_EDGE, GRID, one)[0])
        r[z] = (b["6.2"] / b["11.2"], b["3.3"] / b["11.2"])
    assert r[0][0] < 0.5 and r[1][0] > 10 * r[0][0] and r[2][0] > r[1][0]
    assert r[0][1] > r[1][1] > r[2][1]


def test_6p2_over_11p2_rises_with_g0(c54):
    """At n_H = 1e4, n_e = 3, T = 300 K the family 6.2/11.2 rises with G0: measured
    0.57, 0.69, 3.0 at G0 = 1e2, 1e3, 1e4 as the cation fraction goes 0.01, 0.15,
    0.72 (charge), then 249 at 1e5, where the family is dehydrogenated and the
    stripped modes lose the C-H out-of-plane band (dehydrogenation, not charge)."""
    ratios = []
    for g0 in (1e2, 1e3, 1e4, 1e5):
        env = Environment("t", g0, 1e4, 3.0, 300.0, teff_k=None)
        b = band_fluxes(GRID, c54.spectrum(env, GRID)[0])
        ratios.append(b["6.2"] / b["11.2"])
    assert np.all(np.diff(ratios) > 0), ratios
