"""Grackle (Chiaki & Wise 2019 chemistry) as a reference for the collapse model.

The bundled tables come from ``diagnostics/collapse/make_grackle_tables.py``:
dust-free Grackle tracks for the N07 CCSN pathways (reverse shock into
n_amb = 1 cm^-3), the pathways' grain moments and Planck-mean opacities, and
Grackle's own [Z_cr] with its full chemistry (``data/grackle_ccsn_n1_zcr.txt``).

Grackle reproduces Chiaki+15's static [Z_cr] (n_amb = 1: -4.94, -4.42, -4.53,
-3.88 against -4.9, -4.3, -4.7, -4.1 for 13-30 Msun); its grain growth barely
acts (<= 0.05 dex). Ours, on the Omukai+05 Z = 0 track, is 0.2-0.4 dex higher.
Of that, Grackle's gas (its dust-free track at the trial Z; its Z = 0 gas is
1.4-1.8x hotter than Omukai's) accounts for 0.2-0.3 dex. The rest is our energy
equation: it carries the gas chemistry as the track's implied cooling, so when
dust cooling pulls the gas off the track (from 1e11 cm^-3 here) the H2 formation
heating does not respond. Grackle's dusty gas stays near its dust-free track to
1e13 cm^-3, while three-body H2 formation completes, then cools steeply and
fragments; ours cools earlier and more gently (gamma stays above 0.8). With
Grackle's grains and gas we remain 0.0-0.4 dex above Grackle (13: 0.19, 20: 0.37,
25: 0.01 dex); Grackle's opacities (Si alone or all species) do not close it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pycalima import _paths
from pycalima.collapse.grid import bin_means
from pycalima.collapse.ics import grackle_grains, nozawa_ccsn
from pycalima.collapse.integrator import critical_dust_to_gas, fragmentation, run_collapse
from pycalima.collapse.materials import get_material
from pycalima.collapse.optics import planck_table
from pycalima.collapse.tracks import Track, _grackle_table

ZCR = Path(__file__).parent / "data" / "grackle_ccsn_n1_zcr.txt"


def _S(p):
    g, x = p.grid, p.x
    b = g.slopes(x)
    a2 = g.a_edges[:-1] ** 2 * bin_means(np.ones(g.n_bins), g.eta, b, 2.0 / 3.0)
    return np.sum(math.pi * g.number_density(x, b) * a2) / x.sum()


def test_grackle_track_interpolates_in_metallicity():
    path = str(_paths.get_external_data_path("collapse_tracks", "grackle_ccsn20_n1_dustfree.txt"))
    d, lz = _grackle_table(path)
    k = int(np.argmin(np.abs(lz + 4.0)))
    on = Track.grackle(20, 10.0 ** lz[k])
    assert np.allclose(on.log_T, d[np.isfinite(d[:, 1 + k]), 1 + k])
    mid = Track.grackle(20, 10.0 ** (0.5 * (lz[k] + lz[k + 1])))
    assert np.allclose(mid.log_T, 0.5 * (d[:, 1 + k] + d[:, 2 + k]))


@pytest.mark.parametrize("lz", [-6.0, -4.0, -2.5])
def test_grackle_dust_free_tracks_do_not_fragment(lz):
    """Without dust no sub-solar fragment, at any metallicity of the table."""
    h, _ = run_collapse(Track.grackle(20, 10.0 ** lz), [], n0=1e-1, n_end=1e16, dlnn=0.1, energy=True)
    assert fragmentation(h) == []


@pytest.mark.parametrize("M", [13, 20, 25, 30])
def test_grackle_grains_have_grackle_cross_sections(M, collapse_optics):
    pops, _, info = nozawa_ccsn(M, 1e-5, n_amb=1, grains="grackle")
    gg = grackle_grains()[float(M)]
    assert info["f_dep"] == pytest.approx(sum(v[0] for v in gg.values()), rel=1e-6)
    for p in pops:
        frac, _, r2, r3 = gg[p.material]
        # broad distributions lose a little to the 1 A grid edge
        assert _S(p) == pytest.approx(3 * r2 / (4 * get_material(p.material).rho_s * r3), rel=0.03)


def test_grackle_silicon_is_as_transparent_as_ours(collapse_optics):
    """Nozawa+08 / Chiaki+15 silicon (Piller 1985) absorbs like ours (Franta+17)
    to a factor of a few, about 1e3 below carbon: the low far-IR emissivity of pure Si is
    not an artefact of our optical constants."""
    path = _paths.get_external_data_path("collapse_ics", "grackle_ccsn_n1_planck_opacity.txt")
    with open(path) as fh:
        cols = [ln for ln in fh if ln.startswith("# columns")][0].split("then")[-1].split()
    d = np.loadtxt(path)
    lT = d[:, 0]
    ratio = {}
    for sp in ("C", "Si"):
        g = d[:, 1 + cols.index(f"M20:{sp}")]
        t = planck_table(sp)
        T = 10.0 ** lT[(lT >= 1.0) & (lT <= 2.5)]
        ratio[sp] = np.interp(np.log10(T), lT, g) / (t(np.full(T.size, 1e-6), T) / 1e-6)
    cal = ratio["C"]
    assert np.allclose(cal, 7.85e15, rtol=0.01)              # one constant: the calibration
    assert np.all((ratio["Si"] / cal > 1.0) & (ratio["Si"] / cal < 5.0))
    qa = lambda sp, T: float(np.squeeze(planck_table(sp)(np.array([1e-6]), T))) / 1e-6
    assert qa("Si", 100.0) < 1e-3 * qa("C", 100.0)


def _ref():
    return {round(r[0]): r[1] for r in np.loadtxt(ZCR)}


def _zcr(M, track):
    make = lambda Z: nozawa_ccsn(M, Z, n_amb=1, grains="grackle")[:2]
    return math.log10(critical_dust_to_gas(make, track, criterion="frag", lo=10 ** -6.0, hi=10 ** -3.0,
                                           scan_dex=0.25, rtol=0.05, dlnn=0.1, n0=1e-1))


@pytest.mark.parametrize("M", [13, 20])
def test_grackle_gas_brings_critical_metallicity_towards_grackle(M, collapse_optics):
    """Static dust with Grackle's grains: Grackle's gas lowers [Z_cr] by 0.15-0.35 dex
    from the Omukai+05 Z = 0 track, and leaves it at most 0.45 dex above Grackle's
    own (the energy-equation limitation in the module docstring; 0.19 and 0.37 now)."""
    omukai = _zcr(M, Track.omukai2005("Z0"))
    grackle = _zcr(M, lambda Z: Track.grackle(M, Z))
    assert 0.15 <= omukai - grackle <= 0.35
    assert 0.0 <= grackle - _ref()[M] <= 0.45
