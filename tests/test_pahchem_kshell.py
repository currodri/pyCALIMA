"""C 1s (K-shell) absorption (``Params.kshell``): the cross-section split and the per-absorption branching."""

import numpy as np
import pytest

from pycalima.models.dust_charge import wdb06_atomic
from pycalima.pahchem import photo, uvlib
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Params


def _library_or_skip():
    if not uvlib.default_path().exists():
        pytest.skip("external_data/pahchem/pah_uv_library.npz is not bundled")


def test_kshell_cross_section_is_the_c1s_part_of_the_library():
    """The library less its C 1s part stays >= 0 (the valence remainder); above the measured edge's blend the part is the
    atomic C 1s, ~96 % of the whole above 500 eV."""
    _library_or_skip()
    E = np.geomspace(270.0, 1e4, 3000)
    for nc, z in ((24, 0), (54, 1), (216, 2)):
        total = photo.sigma_uv(nc, z, E, "octopus_library")
        k = uvlib.sigma_k_shell(nc, E)
        assert np.all(total - k >= -1e-12 * total.max())
        far = E > uvlib.K_BLEND_END_EV
        assert np.allclose(k[far], nc * wdb06_atomic.photoionisation_cross_section("C", "1s", E[far]))
        assert np.all((k / total)[E > 500.0] > 0.95)


_CACHE = {}


def _mol():
    """C54H18 (PAHdb v4.00) on the cold-limit ladder."""
    if "mol" in _CACHE:
        return _CACHE["mol"]
    import v400_records

    recs = v400_records.records()
    _CACHE["mol"] = P.extend_ladder(next(m for m in P.library_v400(recs) if m.name == "C54H18 (37)"))
    return _CACHE["mol"]


@pytest.mark.parametrize("escape", ["none", "window", "y2"])
def test_kshell_branching_sums_to_one_and_follows_the_edge(escape):
    mol = _mol()
    p = Params(coulomb_escape=escape)
    E = np.linspace(282.0, 600.0, 2000)
    for z in (0, 3, 8):
        prob, e_k, depth = P.kshell_branching(mol, z, p, E)
        assert np.allclose(prob.sum(axis=0), 1.0) and np.all(prob >= 0.0)
        w, f = P.K_FLUORESCENCE, P.K_DOUBLE_AUGER
        above = E >= e_k
        assert np.allclose(prob[2][above], (1 - w) * (1 - f)) and np.allclose(prob[3][above], (1 - w) * f)
        resonance = E < e_k - depth                # 1s -> pi*: the electron stays, one Auger electron leaves
        assert np.allclose(prob[1][resonance], (1 - w) * (1 - f))
        assert e_k == pytest.approx(P.K_EDGE_EV + P._ladder_ip(mol, z) - mol.ionisation_ev[0])
        window = ~above & ~resonance
        if escape == "none":
            assert np.allclose(prob[2][window], (1 - w) * (1 - f))
        elif escape == "y2":
            assert np.allclose(prob[1][window], (1 - w) * (1 - f))


def test_xray_only_field_is_scaled_by_its_energy_flux():
    """No flux at 6-13.6 eV: Habing G0 is 0, and the field takes its scale from its whole energy flux in Habing units."""
    from pycalima.pahchem import fields

    f = fields.powerlaw(1.0, 250.0, e_min=300.0, e_max=1e4)
    assert f.g0() == 0.0 and f.intensity() == pytest.approx(250.0)
    total = np.trapezoid(f.energy_ev * f.photon_flux, f.energy_ev) * fields.EV_ERG
    assert total == pytest.approx(250.0 * fields.HABING_FLUX)
    uv = fields.powerlaw(1.0, 250.0)                       # with the far UV: the Habing band sets it, as before
    assert uv.g0() == pytest.approx(250.0) and uv.intensity() == uv.g0()


def test_deposit_response_conserves_energy_and_strips_more_with_the_charge():
    """One deposit's IR is the deposit; the loss per capture (H, H2, C2H2, protons) grows with the energy it deposits (IP
    rises with Z)."""
    from pycalima.pahchem import dos

    mol = _mol()
    p = Params()
    probs = []
    for z in (1, 5, 9):
        responses = []
        for which in P.RESPONSE_SETS:                         # a capture; the Auger excitation plus each heat node
            inp = P._response_inputs(mol, z, 100.0, 45.0 * dos.CM_PER_EV, p, which)
            responses += list(zip(inp[-1], P.deposit_response(*inp)))
        rec = responses[0][1]
        for e, members in responses:
            for em, *prob in members:
                assert em.sum() / dos.CM_PER_EV == pytest.approx(e, rel=2e-3)
                assert all(0.0 <= x <= 1.0 for x in prob) and sum(prob) <= 1.0 + 1e-9
        probs.append(sum(rec[0][1:]))                          # every loss: H, H2, C2H2, protons (the highest charges)
    assert probs[0] < probs[1] < probs[2]


def test_round_off_guard_clips_ulps_and_raises_on_real_excursions():
    """The probability and rate clips take off round-off only (1-2 ulp measured), never a real excursion."""
    assert P._round_off(1.0 + 4.4e-16, 0.0, 1.0, "p") == 1.0
    assert P._round_off(-2e-16, 0.0, 1.0, "p") == 0.0
    with pytest.raises(ValueError):
        P._round_off(1.0 + 1e-9, 0.0, 1.0, "p")
    with pytest.raises(ValueError):
        P._round_off(np.nan, 0.0, 1.0, "p")                                    # NaN fails every comparison: caught too
    with pytest.raises(ValueError):
        photo.yield_coulomb_window(np.array([15.0]), 10.0, 5.0, u_c=-1.0)      # outside the domain of p(u)
