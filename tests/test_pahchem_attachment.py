"""Statistical electron sticking on neutral PAHs (pahchem.attachment).

Gate 1 is exact: the autodetachment rate is built from the capture cross
section by detailed balance, so its thermal average must equal the capture
rate times the Saha-like equilibrium constant. Gate 2 records the values for
coronene and circumcoronene with PAHdb anion modes (model_data/PAH_states).
"""

import numpy as np
import pytest

from pycalima.pahchem import dos, energy_me, fields, photo
from pycalima.pahchem.attachment import (ERG_PER_CM, HBAR_CGS, ME_G, Attachment,
                                         capture_cross_section)
from pycalima.pahchem.molecule import PAHSpecies

KB_CGS = 1.380649e-16


@pytest.mark.parametrize("ea_ev, temperature_k", [(0.3, 800.0), (0.5, 1500.0)])
def test_autodetachment_obeys_detailed_balance(ea_ev, temperature_k):
    """<k_d>_anion(T) / <sigma_c v>(T) = (2 g_n/g_a) (2 pi m_e kT/h^2)^1.5 (Q_n/Q_a) e^{-EA/kT}
    for a 30-mode molecule; measured to 6e-4."""
    rng = np.random.default_rng(1)
    fn = np.sort(rng.uniform(150, 1600, 30)).round()
    neutral = PAHSpecies(12, 0, 0, fn, np.ones(30))
    anion = PAHSpecies(12, 0, -1, (0.97 * fn).round(), np.ones(30))
    at = Attachment(neutral, anion, ea_ev, 3e-8, e_excess_cm=60000.0, grain_cm=1.0)
    kt = dos.KB_CM * temperature_k
    _, ca = dos.beyer_swinehart(anion.freqs_cm, at.energy_cm[-1], 1.0)
    En, cn = dos.beyer_swinehart(neutral.freqs_cm, at.energy_cm[-1], 1.0)
    boltz = ca * np.exp(-at.energy_cm / kt)
    kd_mean = at.k_detach @ boltz / boltz.sum()
    e = np.arange(1, 40 * int(kt)) * ERG_PER_CM
    maxwell = np.sqrt(e) * np.exp(-e / (KB_CGS * temperature_k))
    k_cap = capture_cross_section(e / ERG_PER_CM, 3e-8) * np.sqrt(2 * e / ME_G) @ maxwell / maxwell.sum()
    h = 2 * np.pi * HBAR_CGS
    k_eq = ((2 * 1 / 2) * (2 * np.pi * ME_G * KB_CGS * temperature_k / h**2) ** 1.5
            * (cn @ np.exp(-En / kt)) / boltz.sum() * np.exp(-ea_ev * dos.CM_PER_EV / kt))
    assert kd_mean / k_cap == pytest.approx(k_eq, rel=2e-3)


def test_no_detachment_channel_means_unit_sticking():
    """With EA beyond any capture energy the anion cannot detach: s_e = 1."""
    fn = np.sort(np.random.default_rng(2).uniform(100, 3000, 30)).round()
    sp = PAHSpecies(12, 0, 0, fn, np.ones(30))
    at = Attachment(sp, PAHSpecies(12, 0, -1, fn, np.ones(30)), 5.0, 3e-8, e_excess_cm=3000.0)
    assert at.sticking(100.0, [0.0], [1.0]) > 0.999


def _pair(model_data, formula):
    states = model_data / "PAH_states"
    paths = [states / f"{formula}_{z}.dat" for z in (0, -1)]
    if not all(p.exists() for p in paths):
        pytest.skip(f"model_data/PAH_states/ lacks {formula} neutral or anion modes")
    return [PAHSpecies.from_states_file(p) for p in paths]


def test_coronene_sticking_in_the_ism(model_data):
    """C24H12 (EA 0.47 eV), neutral at its radiative P(E): s_e = 4.2e-3, 2.5e-3,
    2.7e-4 at 30, 100, 1000 K (G0 = 1). Autodetachment wins over IR emission, so
    s_e << 1 and falls with T; P(E) at G0 = 1e4 lowers it by 9 %. A 300 K thermal
    neutral (a laboratory cell) gives 4.6e-6, as the "statistical" 1e-5 of Andrews+16."""
    from pycalima.pahchem.chemistry import disc_radius_cm

    neutral, anion = _pair(model_data, "C24H12")
    at = Attachment(neutral, anion, 0.47, disc_radius_cm(24), e_excess_cm=30000.0)
    cm = energy_me.CascadeModel(neutral.freqs_cm, neutral.einstein_a, e_max_cm=16 * dos.CM_PER_EV,
                                de_cm=10.0, grain_cm=1.0)
    sigma = photo.sigma_li_draine(24, 0, photo_grid := np.linspace(5.0, 13.6, 345))
    s = {}
    for g0 in (1.0, 1e4):
        h = photo.heating_spectrum(fields.draine1978(g0=g0), sigma, photo.yield_jochims1996(photo_grid, 7.2),
                                   photo_grid)
        p = cm.steady_state(photo_grid, h, {}).probability
        s[g0] = [at.sticking(T, cm.energy_cm, p) for T in (30.0, 100.0, 1000.0)]
    assert s[1.0] == pytest.approx([4.2e-3, 2.5e-3, 2.7e-4], rel=0.05)
    assert np.all(np.array(s[1e4]) / s[1.0] > 0.88)
    assert at.sticking(300.0) == pytest.approx(4.6e-6, rel=0.1)


def test_circumcoronene_sticks_every_electron(model_data):
    """C54H18 (EA 1.44 eV): the anion's state density at EA is so large that
    autodetachment is negligible, s_e = 1 to 1e-3 up to 3000 K. This replaces the
    assumed s_e = 1 of the baseline card with a calculation."""
    from pycalima.pahchem.chemistry import disc_radius_cm

    neutral, anion = _pair(model_data, "C54H18")
    at = Attachment(neutral, anion, 1.44, disc_radius_cm(54), e_excess_cm=8000.0)
    assert min(at.sticking(T, [0.0], [1.0]) for T in (10.0, 300.0, 3000.0)) > 0.999
