"""Statistical electron sticking on neutral PAHs (pahchem.attachment).

Gate 1 is exact: the autodetachment rate is built from the capture cross
section by detailed balance, so its thermal average must equal the capture
rate times the Saha-like equilibrium constant. Gate 2 records the values for
coronene and circumcoronene with PAHdb anion modes (model_data/PAH_states).
"""

import numpy as np
import pytest

from pycalima.pahchem import cooling, dos, energy_me, fields, photo
from pycalima.pahchem.attachment import (ERG_PER_CM, HBAR_CGS, ME_G, Attachment,
                                         capture_cross_section, capture_flux_tail, grid_excess_cm)
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


def test_capture_fraction_scales_autodetachment_and_raises_sticking():
    """f sigma_c: detailed balance scales k_d by f, so fewer anions autodetach (s_e rises) while f s_e falls."""
    rng = np.random.default_rng(3)
    fn = np.sort(rng.uniform(150, 1600, 30)).round()
    neutral = PAHSpecies(12, 0, 0, fn, np.ones(30))
    anion = PAHSpecies(12, 0, -1, (0.97 * fn).round(), np.ones(30))
    full, part = (Attachment(neutral, anion, 0.5, 3e-8, e_excess_cm=8000.0, capture_fraction=f) for f in (1.0, 0.01))
    assert np.allclose(part.k_detach, 0.01 * full.k_detach)
    s1, sf = full.sticking(100.0), part.sticking(100.0)
    assert s1 <= sf <= 1.0 and 0.01 * sf < s1
    with pytest.raises(ValueError):
        Attachment(neutral, anion, 0.5, 3e-8, e_excess_cm=8000.0, capture_fraction=0.0)


def test_capture_flux_tail_is_the_integral_of_the_capture_flux():
    kt, x = dos.KB_CM * 3000.0, 1500.0
    eps = np.linspace(x, x + 60 * kt, 2_000_001)
    direct = np.trapezoid(capture_cross_section(eps, 3e-8) * eps * np.exp(-eps / kt), eps)
    assert capture_flux_tail(x, kt, 3e-8) == pytest.approx(direct, rel=1e-6)


def test_sticking_does_not_depend_on_the_grid_ceiling():
    """The Maxwellian above the grid stays in the flux normalisation: once P_stab ~ 0 at the top, raising
    ``e_excess_cm`` changes nothing, also at temperatures where most electrons are above the grid."""
    rng = np.random.default_rng(5)
    fn = np.sort(rng.uniform(150, 1600, 30)).round()
    neutral = PAHSpecies(12, 0, 0, fn, np.ones(30))
    anion = PAHSpecies(12, 0, -1, (0.97 * fn).round(), np.ones(30))
    short, tall = (Attachment(neutral, anion, 0.5, 3e-8, e_excess_cm=x) for x in (8000.0, 40000.0))
    for t in (100.0, 3000.0, 1e4, 1e5):
        g = (np.array([0.0]), np.array([1.0]))
        assert short.sticking(t, *g) == pytest.approx(tall.sticking(t, *g), rel=1e-6)


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


def test_hot_electrons_fall_off_a_large_pah_on_a_grid_sized_to_kt(model_data):
    """C54H18 (EA 1.2 eV) sticks every electron at 1e3 K but not at 1e4 and 1e5 K, where the electrons
    reach energies at which autodetachment wins; an 8000 cm^-1 grid, whose top bin still has P_stab = 1, gave
    ~1 there. The grid of ``grid_excess_cm`` is converged (a twice taller one agrees)."""
    from pycalima.pahchem.chemistry import disc_radius_cm

    neutral, anion = _pair(model_data, "C54H18")
    ground = (np.array([0.0]), np.array([1.0]))
    s = []
    for t in (1e3, 1e4, 1e5):
        x = grid_excess_cm(t)
        at, tall = (Attachment(neutral, anion, 1.2, disc_radius_cm(54), e_excess_cm=e) for e in (x, 2 * x))
        s.append(at.sticking(t, *ground))
        assert s[-1] == pytest.approx(tall.sticking(t, *ground), rel=1e-6)
    assert s[0] > 0.999 and 0.6 < s[1] < 0.9 and s[2] < 0.1


def test_circumcoronene_sticks_every_electron(model_data):
    """C54H18 (EA 1.44 eV): the anion's state density at EA is so large that
    autodetachment is negligible, s_e = 1 to 1e-3 up to 3000 K. This replaces the
    assumed s_e = 1 of the baseline card with a calculation."""
    from pycalima.pahchem.chemistry import disc_radius_cm

    neutral, anion = _pair(model_data, "C54H18")
    at = Attachment(neutral, anion, 1.44, disc_radius_cm(54), e_excess_cm=8000.0)
    assert min(at.sticking(T, [0.0], [1.0]) for T in (10.0, 300.0, 3000.0)) > 0.999


def test_stabilisation_kernel_matches_the_numpy_loop():
    """P_stab through the numba recursion (when installed) equals the NumPy loop to rounding."""
    from pycalima.pahchem import attachment

    rng = np.random.default_rng(4)
    fn = np.sort(rng.uniform(150, 1600, 30)).round()
    fa = (0.97 * fn).round()
    at = Attachment(PAHSpecies(12, 0, 0, fn, np.ones(30)), PAHSpecies(12, 0, -1, fa, np.ones(30)), 0.5, 3e-8,
                    e_excess_cm=8000.0)
    _, counts = dos.beyer_swinehart(fa, at.energy_cm[-1], at.grain)
    k_modes = cooling.mode_emission_rates(at.energy_cm, counts, fa, np.ones(30), window_cm=20.0)
    steps = np.maximum(np.rint(fa / at.grain).astype(np.int64), 1)
    slow = attachment._stabilisation_numpy(k_modes, at.k_ir + at.k_detach, steps, at.i_ea)
    assert np.max(np.abs(at.p_stab - slow)) < 1e-13


def test_delayed_detachment_is_a_probability_with_slow_electrons():
    """1 - P_stab: zero at and below EA, rising with the internal energy to 1; the autodetached electron's
    mean energy is positive and far below E - EA (thermionic); the numba and NumPy cascades agree."""
    from pycalima.pahchem import attachment

    rng = np.random.default_rng(5)
    fn = np.sort(rng.uniform(150, 1600, 30)).round()
    fa = (0.97 * fn).round()
    at = Attachment(PAHSpecies(12, 0, 0, fn, np.ones(30)), PAHSpecies(12, 0, -1, fa, np.ones(30)), 0.5, 3e-8,
                    e_excess_cm=40000.0, grain_cm=5.0)
    E = np.linspace(0.0, at.energy_cm[-1], 400)
    p, eps = at.delayed_detachment(E)
    ea = 0.5 * dos.CM_PER_EV
    assert np.all((p >= 0) & (p <= 1 + 1e-12)) and np.all(p[E <= ea] < 1e-12)
    assert np.all(np.diff(p) > -1e-9) and p[-1] > 0.99
    hot = p > 1e-3
    assert np.all(eps[hot] > 0) and np.all(eps[hot] < E[hot] - ea)
    far = E - ea > 2.0 * dos.CM_PER_EV
    assert np.all(eps[far] < 0.2 * (E[far] - ea))                     # thermionic: a small share of the excess
    _, counts = dos.beyer_swinehart(fa, at.energy_cm[-1], at.grain)
    k_modes = cooling.mode_emission_rates(at.energy_cm, counts, fa, np.ones(30), window_cm=20.0)
    steps = np.maximum(np.rint(fa / at.grain).astype(np.int64), 1)
    slow = attachment._cascade_yield_numpy(k_modes, at.k_ir + at.k_detach, steps, at.i_ea, at.k_detach * at.eps_detach)
    assert np.max(np.abs(at.w_detach - slow)) <= 1e-10 * max(np.max(np.abs(slow)), 1e-300)


@pytest.mark.parametrize("ea_cm", [8232.4, 8237.6])
def test_delayed_detachment_conserves_energy_within_a_grain_of_the_ea(ea_cm):
    """Energy conservation near the threshold, on the population's 10 cm^-1 grid: no detachment at or below the EA,
    and an electron that never takes more than the excess E - EA. The grid rounds the EA (8232.4 -> 8230, 8237.6 ->
    8240 cm^-1), so its excess is off by up to half a grain, and the window averaging adds the next bins' energies;
    within a few grains of the EA both pushed the mean energy past the excess (here by up to 16 cm^-1; C30H14 (7277)-
    by 3.6 cm^-1 at 3.5 grains, a negative internal energy after detachment), and below a down-rounded EA P_det
    reached 0.18."""
    rng = np.random.default_rng(6)
    fn = np.sort(rng.uniform(150, 1600, 30)).round()
    fa = (0.97 * fn).round()
    at = Attachment(PAHSpecies(12, 0, 0, fn, np.ones(30)), PAHSpecies(12, 0, -1, fa, np.ones(30)),
                    ea_cm / dos.CM_PER_EV, 3e-8, e_excess_cm=3000.0, grain_cm=10.0)
    E = at.ea_cm + np.linspace(-20.0, 200.0, 2201)
    p, eps = at.delayed_detachment(E)
    above = E > at.ea_cm
    assert np.all(p[~above] == 0.0) and p[above].max() > 0.5                # it does detach above the EA
    assert np.all(eps[above] <= E[above] - at.ea_cm) and np.all(eps[~above] == 0.0)
